from collections.abc import Sequence
from typing import Any, Literal

import numpy as np
import pandas as pd

from endrs.data.data_info import IdConverter
from endrs.feature.building import (
    build_dense_arrays,
    build_hash_sparse_arrays,
    build_multi_sparse_arrays,
    build_sparse_arrays,
)
from endrs.feature.config import FeatureConfig
from endrs.feature.extraction import (
    extract_all_multi_sparse_values,
    extract_all_sparse_values,
)
from endrs.feature.feat_info import FeatInfo
from endrs.feature.mapping import (
    build_hash_mapping,
    build_sequential_mapping,
    update_hash_mapping,
)
from endrs.utils.hashing import Hasher


def map_and_reindex_feat_data(
    feat_data: pd.DataFrame,
    col_name: str,
    mapping: dict[Any, int],
    name: Literal["user", "item"],
    unique_ids: np.ndarray | None = None,
    require_complete_coverage: bool = False,
) -> pd.DataFrame:
    """Map raw user/item IDs to internal indices.

    Feature array alignment is performed by `build_*_arrays` functions using
    the entity ID column from the returned DataFrame. This design supports
    partial feature coverage where missing entities retain OOV default values.

    Parameters
    ----------
    feat_data : pd.DataFrame
        Feature data containing entity ID column.
    col_name : str
        Name of the ID column (e.g., 'user_id', 'item_id').
    mapping : dict[Any, int]
        Mapping from raw IDs to internal indices.
    name : {'user', 'item'}
        Entity name for error messages ('user' or 'item').
    unique_ids : np.ndarray or None
        All unique IDs from interaction data. Required if require_complete_coverage=True.
    require_complete_coverage : bool
        If True, validate that all unique_ids exist in feat_data.
        Set to False to allow partial feature coverage.

    Returns
    -------
    pd.DataFrame
        Feature data with IDs mapped to internal indices.

    Raises
    ------
    ValueError
        If feat_data has duplicate IDs or missing required IDs.

    Notes
    -----
    OOV (Out-Of-Vocabulary) Handling:

    - When `require_complete_coverage=False`, entities in interaction data that
      don't have features in feat_data will use OOV index (0) during training.
    - Feature arrays reserve position 0 for OOV values, so feat_array[0] returns
      the OOV feature representation.
    - This is useful when you only want to provide features for popular entities
      (e.g., top 100k items out of 1M total items).
    - IDs in feat_data that don't exist in the mapping are dropped (via dropna).
    """
    feat_ids: pd.Series = feat_data[col_name]
    if not feat_ids.is_unique:
        raise ValueError(f"{name} feat data must have unique ids.")

    if require_complete_coverage and unique_ids is not None:
        feat_id_set = set(feat_ids.tolist())
        for i in unique_ids:
            if i not in feat_id_set:
                raise ValueError(
                    f"id `{i}` does not exist in {name} feat data. "
                    f"Set `require_complete_coverage=False` to allow partial coverage."
                )

    feat_data = feat_data.copy()
    feat_data[col_name] = feat_data[col_name].map(mapping)
    feat_data = feat_data.dropna(subset=[col_name])
    # After mapping, NA value may result in float type, converting back to int
    feat_data[col_name] = feat_data[col_name].astype(np.int64)
    return feat_data


class FeatureProcessor:
    """Feature processor for Dataset, supports sparse/dense/multi-sparse features.

    Parameters
    ----------
    id_converter : IdConverter
        Converter between raw IDs and internal indices.
    user_col : str
        Name of the user ID column.
    item_col : str
        Name of the item ID column.
    require_complete_coverage : bool
        If True, require all entity IDs from interaction data to have features.
        Set to False to allow partial feature coverage.
    """

    def __init__(
        self,
        id_converter: IdConverter,
        user_col: str,
        item_col: str,
        require_complete_coverage: bool = True,
    ):
        self.id_converter = id_converter
        self.user_col = user_col
        self.item_col = item_col
        self.require_complete_coverage = require_complete_coverage
        self.sparse_val_to_idx: dict[str, dict[Any, int]] = {}

    def process(
        self,
        user_feat_data: pd.DataFrame | None,
        item_feat_data: pd.DataFrame | None,
        config: FeatureConfig,
    ) -> FeatInfo:
        """Process features and build FeatInfo.

        Parameters
        ----------
        user_feat_data : pd.DataFrame or None
            User feature data.
        item_feat_data : pd.DataFrame or None
            Item feature data.
        config : FeatureConfig
            Feature configuration containing column names and padding values.

        Returns
        -------
        FeatInfo
            Feature information object.
        """
        # Extract unique values
        sparse_unique_vals = extract_all_sparse_values(
            user_feat_data,
            item_feat_data,
            config.user.sparse_cols,
            config.item.sparse_cols,
        )
        multi_sparse_unique_vals = extract_all_multi_sparse_values(
            user_feat_data,
            item_feat_data,
            config.user.multi_sparse_cols,
            config.item.multi_sparse_cols,
            config.user.pad_val,
            config.item.pad_val,
        )

        # Build value-to-index mappings
        all_sparse_unique_vals = sparse_unique_vals | multi_sparse_unique_vals
        self.sparse_val_to_idx = build_sequential_mapping(all_sparse_unique_vals)

        # Build feature arrays for each entity
        user_feat_unique = (
            self._build_entity_features(
                user_feat_data,
                config.user.sparse_cols,
                config.user.dense_cols,
                config.user.multi_sparse_cols,
                "user",
            )
            if user_feat_data is not None
            else {}
        )
        item_feat_unique = (
            self._build_entity_features(
                item_feat_data,
                config.item.sparse_cols,
                config.item.dense_cols,
                config.item.multi_sparse_cols,
                "item",
            )
            if item_feat_data is not None
            else {}
        )
        feat_unique = user_feat_unique | item_feat_unique

        return FeatInfo(
            config.user.sparse_cols,
            config.item.sparse_cols,
            config.user.dense_cols,
            config.item.dense_cols,
            config.user.multi_sparse_cols,
            config.item.multi_sparse_cols,
            feat_unique=feat_unique,
            sparse_val_to_idx=self.sparse_val_to_idx,
        )

    def _build_entity_features(
        self,
        feat_data: pd.DataFrame,
        sparse_cols: Sequence[str] | None,
        dense_cols: Sequence[str] | None,
        multi_sparse_cols: Sequence[Sequence[str]] | None,
        name: Literal["user", "item"],
    ) -> dict[str, np.ndarray]:
        """Build feature arrays for a single entity type (user or item).

        Parameters
        ----------
        feat_data : pd.DataFrame
            Feature data.
        sparse_cols : Sequence[str] or None
            Sparse column names.
        dense_cols : Sequence[str] or None
            Dense column names.
        multi_sparse_cols : Sequence[Sequence[str]] or None
            Multi-sparse column groups.
        name : {'user', 'item'}
            Either "user" or "item".

        Returns
        -------
        dict[str, np.ndarray]
            Dictionary mapping column names to feature arrays.
        """
        if name == "user":
            col_name = self.user_col
            unique_ids = self.id_converter.unique_users
            mapping = self.id_converter.user2id
        else:
            col_name = self.item_col
            unique_ids = self.id_converter.unique_items
            mapping = self.id_converter.item2id

        feat_data = map_and_reindex_feat_data(
            feat_data,
            col_name,
            mapping,
            name,
            unique_ids if self.require_complete_coverage else None,
            self.require_complete_coverage,
        )

        entity_ids = feat_data[col_name].to_numpy()
        n_entities = len(mapping)

        sparse_unique = build_sparse_arrays(
            feat_data, sparse_cols, self.sparse_val_to_idx, entity_ids, n_entities
        )
        dense_unique = build_dense_arrays(feat_data, dense_cols, entity_ids, n_entities)
        multi_sparse_unique = build_multi_sparse_arrays(
            feat_data, multi_sparse_cols, self.sparse_val_to_idx, entity_ids, n_entities
        )

        return {**sparse_unique, **dense_unique, **multi_sparse_unique}


class HashFeatureProcessor:
    """Feature processor for HashDataset, supports sparse features only.

    Parameters
    ----------
    hasher : Hasher
        Hasher instance for computing hash values.
    id_converter : IdConverter
        Converter between raw IDs and internal indices.
    user_col : str
        Name of the user ID column.
    item_col : str
        Name of the item ID column.
    n_bins : int
        Number of hash bins.
    retrain : bool
        Whether this is incremental training.
    existing_sparse_val_to_idx : dict or None
        Existing value-to-index mappings (for retrain).
    existing_feat_unique : dict or None
        Existing feature arrays (for retrain).
    """

    def __init__(
        self,
        hasher: Hasher,
        id_converter: IdConverter,
        user_col: str,
        item_col: str,
        n_bins: int,
        retrain: bool = False,
        existing_sparse_val_to_idx: dict[str, dict[Any, int]] | None = None,
        existing_feat_unique: dict[str, np.ndarray] | None = None,
    ):
        self.hasher = hasher
        self.id_converter = id_converter
        self.user_col = user_col
        self.item_col = item_col
        self.n_bins = n_bins
        self.retrain = retrain

        if retrain:
            self.sparse_val_to_idx = existing_sparse_val_to_idx
            self.feat_unique = existing_feat_unique
        else:
            self.sparse_val_to_idx = {}
            self.feat_unique = {}

    def process(
        self,
        user_feat_data: pd.DataFrame | None,
        item_feat_data: pd.DataFrame | None,
        config: FeatureConfig,
    ) -> FeatInfo:
        """Process sparse features and build FeatInfo.

        Only sparse features are supported in hash mode.

        Parameters
        ----------
        user_feat_data : pd.DataFrame or None
            User feature data.
        item_feat_data : pd.DataFrame or None
            Item feature data.
        config : FeatureConfig
            Feature configuration. Only sparse_cols are used;
            dense_cols and multi_sparse_cols must be None.

        Returns
        -------
        FeatInfo
            Feature information object.

        Raises
        ------
        ValueError
            If dense_cols or multi_sparse_cols are provided.
        """
        if config.user.has_dense or config.item.has_dense:
            raise ValueError(
                "HashFeatureProcessor does not support dense features. "
                "Use Dataset (non-hash mode) for dense features."
            )
        if config.user.has_multi_sparse or config.item.has_multi_sparse:
            raise ValueError(
                "HashFeatureProcessor does not support multi-sparse features. "
                "Use Dataset (non-hash mode) for multi-sparse features."
            )

        user_sparse_cols = config.user.sparse_cols
        item_sparse_cols = config.item.sparse_cols
        sparse_unique_vals = extract_all_sparse_values(
            user_feat_data, item_feat_data, user_sparse_cols, item_sparse_cols
        )

        if self.retrain:
            update_hash_mapping(self.hasher, self.sparse_val_to_idx, sparse_unique_vals)
        else:
            self.sparse_val_to_idx = build_hash_mapping(self.hasher, sparse_unique_vals)

        if user_feat_data is not None and user_sparse_cols:
            self._build_hash_features(user_feat_data, user_sparse_cols, "user")
        if item_feat_data is not None and item_sparse_cols:
            self._build_hash_features(item_feat_data, item_sparse_cols, "item")

        return FeatInfo(
            user_sparse_cols,
            item_sparse_cols,
            feat_unique=self.feat_unique,
            sparse_val_to_idx=self.sparse_val_to_idx,
        )

    def _build_hash_features(
        self,
        feat_data: pd.DataFrame,
        sparse_cols: Sequence[str],
        name: Literal["user", "item"],
    ):
        """Build hash-indexed sparse feature arrays.

        Parameters
        ----------
        feat_data : pd.DataFrame
            Feature data.
        sparse_cols : Sequence[str]
            Sparse column names.
        name : {'user', 'item'}
            Either "user" or "item".
        """
        if name == "user":
            col_name = self.user_col
            mapping = self.id_converter.user2id
        else:
            col_name = self.item_col
            mapping = self.id_converter.item2id

        feat_data = map_and_reindex_feat_data(
            feat_data, col_name, mapping, name, require_complete_coverage=False
        )

        # Always pass existing feat_unique to accumulate results
        self.feat_unique = build_hash_sparse_arrays(
            feat_data,
            sparse_cols,
            self.sparse_val_to_idx,
            col_name,
            self.n_bins,
            self.feat_unique,
        )
