import functools
import itertools
from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from endrs.data.batch import BatchData, EvalBatchData
from endrs.data.consumed import interaction_consumed, merge_consumed_data
from endrs.data.data_info import DataInfo, IdConverter
from endrs.feature.feat_info import FeatInfo
from endrs.types import ItemId, RecModel
from endrs.utils.constants import DEFAULT_HASH_BINS, ITEM_KEY, OOV_IDX, USER_KEY
from endrs.utils.hashing import Hasher
from endrs.utils.validate import check_data_cols, check_feat_cols

if TYPE_CHECKING:
    from endrs.bases.cf_base import CfBase
    from endrs.bases.torch_base import TorchBase


class Dataset:
    """Dataset class for building and managing recommendation data.

    Parameters
    ----------
    user_col_name : str
        Name of the user ID column in the data.
    item_col_name : str
        Name of the item ID column in the data.
    label_col_name : str, optional
        Name of the label column for supervised learning.
    multi_label_col_names : Sequence[str], optional
        Names of multiple label columns for multi-task learning.
    shuffle : bool, default: False
        Whether to shuffle data before processing.
    pop_num : int, default: 100
        Number of popular items to track.
    seed : int, default: 42
        Random seed for reproducibility.

    Attributes
    ----------
    train_data : BatchData
        Processed training data.
    eval_data : EvalBatchData
        Processed evaluation data.
    test_data : EvalBatchData
        Processed test data.
    data_info : DataInfo
        Metadata about the dataset.
    feat_info : FeatInfo
        Feature metadata and mappings.
    id_converter : IdConverter
        Converter between raw IDs and internal indices.
    """

    def __init__(
        self,
        user_col_name: str,
        item_col_name: str,
        label_col_name: str | None = None,
        multi_label_col_names: Sequence[str] | None = None,
        shuffle: bool = False,
        pop_num: int = 100,
        seed: int = 42,
    ):
        self.user_col_name = user_col_name
        self.item_col_name = item_col_name
        self.label_col_name = label_col_name
        self.multi_label_col_names = multi_label_col_names
        self.shuffle = shuffle
        self.pop_num = pop_num
        self.seed = seed
        self.train_data = None
        self.eval_data = None
        self.test_data = None
        self.data_info = None
        self.feat_info = None
        self.id_converter = None
        self.sparse_val_to_idx = None
        self.train_called = False

    def remove_features(self):
        """Clear feature information from the dataset."""
        self.feat_info = None

    def shuffle_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Shuffle DataFrame rows with a fixed random seed.

        Parameters
        ----------
        data : pd.DataFrame
            Data to shuffle.

        Returns
        -------
        pd.DataFrame
            Shuffled data with reset index.
        """
        data = data.sample(frac=1, random_state=self.seed)
        return data.reset_index(drop=True)

    def build_data(
        self,
        train_data: pd.DataFrame,
        eval_data: pd.DataFrame | None = None,
        test_data: pd.DataFrame | None = None,
    ):
        """Build interaction data for training and evaluation.

        Processes raw DataFrames into internal batch format, builds ID mappings,
        and computes user/item consumed history and popular items.

        Parameters
        ----------
        train_data : pd.DataFrame
            Training data containing user-item interactions.
        eval_data : pd.DataFrame, optional
            Evaluation data for validation during training.
        test_data : pd.DataFrame, optional
            Test data for final evaluation.

        Side Effects
        ------------
        Sets `train_data`, `eval_data`, `test_data`, `id_converter`, `data_info`,
        and `train_called` attributes on the instance.
        """
        check_data_cols(
            [train_data, eval_data, test_data],
            self.user_col_name,
            self.item_col_name,
            self.label_col_name,
            self.multi_label_col_names,
        )

        if self.shuffle:
            train_data = self.shuffle_data(train_data)
            if eval_data is not None:
                eval_data = self.shuffle_data(eval_data)
            if test_data is not None:
                test_data = self.shuffle_data(test_data)

        user_unique_vals = np.sort(train_data[self.user_col_name].unique())
        item_unique_vals = np.sort(train_data[self.item_col_name].unique())
        self.id_converter = self._build_id_converter(user_unique_vals, item_unique_vals)

        batch_fn = functools.partial(
            self.build_batch,
            user_col_name=self.user_col_name,
            item_col_name=self.item_col_name,
            label_col_name=self.label_col_name,
            multi_label_col_names=self.multi_label_col_names,
            id_converter=self.id_converter,
        )

        self.train_data = batch_fn(train_data, is_train=True)
        if eval_data is not None:
            self.eval_data = batch_fn(eval_data, is_train=False)
        if test_data is not None:
            self.test_data = batch_fn(test_data, is_train=False)

        pop_items = self._get_popular_items(train_data)
        user_consumed, item_consumed = interaction_consumed(*self.train_data.ui())
        self.data_info = DataInfo(
            len(train_data),
            self.user_col_name,
            self.item_col_name,
            self.label_col_name,
            self.multi_label_col_names,
            self.id_converter,
            user_consumed,
            item_consumed,
            pop_items,
        )
        self.train_called = True

    @staticmethod
    def build_batch(
        data: pd.DataFrame,
        user_col_name: str,
        item_col_name: str,
        label_col_name: str | None,
        multi_label_col_names: Sequence[str] | None,
        id_converter: IdConverter,
        is_train: bool
    ) -> BatchData | EvalBatchData:
        """Convert a DataFrame into batch data for model consumption.

        Parameters
        ----------
        data : pd.DataFrame
            Raw interaction data.
        user_col_name : str
            Name of the user column.
        item_col_name : str
            Name of the item column.
        label_col_name : str, optional
            Name of the label column.
        multi_label_col_names : Sequence[str], optional
            Names of multiple label columns.
        id_converter : IdConverter
            Converter for mapping raw IDs to internal indices.
        is_train : bool
            Whether this is training data (determines return type).

        Returns
        -------
        BatchData or EvalBatchData
            Processed batch data with mapped indices and labels.
        """
        user_indices = data[user_col_name].map(id_converter.safe_user_to_id).tolist()
        item_indices = data[item_col_name].map(id_converter.safe_item_to_id).tolist()

        labels, multi_labels = None, None
        if label_col_name:
            labels = data[label_col_name].astype(np.float32).tolist()
        if multi_label_col_names:
            multi_labels = data[multi_label_col_names].astype(np.float32)
            multi_labels = multi_labels.to_numpy().tolist()

        if is_train:
            return BatchData(user_indices, item_indices, labels, multi_labels)
        else:
            return EvalBatchData(user_indices, item_indices, labels, multi_labels)

    def _build_id_converter(
        self, user_unique_vals: np.ndarray, item_unique_vals: np.ndarray
    ) -> IdConverter:
        """Build ID converter from unique user and item values."""
        user2id, id2user = self.make_id_mapping(user_unique_vals, include_reverse=True)
        item2id, id2item = self.make_id_mapping(item_unique_vals, include_reverse=True)
        return IdConverter(user2id, item2id, id2user, id2item)

    @staticmethod
    def make_id_mapping(unique_values: np.ndarray, include_reverse: bool = False):
        """Create mapping from raw values to internal IDs starting from 1.

        Index 0 is reserved for OOV (out-of-vocabulary) values.

        Parameters
        ----------
        unique_values : np.ndarray
            Unique raw values to map.
        include_reverse : bool, default: False
            Whether to also return reverse mapping (ID to value).

        Returns
        -------
        dict or tuple of dict
            Forward mapping, or (forward, reverse) if include_reverse=True.
        """
        unique_values = unique_values.tolist()
        ids = range(1, len(unique_values) + 1)
        if include_reverse:
            return dict(zip(unique_values, ids)), dict(zip(ids, unique_values))
        else:
            return dict(zip(unique_values, ids))

    def _get_popular_items(self, train_data: pd.DataFrame) -> list[ItemId]:
        """Get the most popular items by interaction count."""
        count_items = (
            train_data.drop_duplicates(subset=[self.user_col_name, self.item_col_name])
            .groupby(self.item_col_name)[self.user_col_name]
            .count()
        )
        return count_items.nlargest(self.pop_num).index.tolist()

    def process_features(
        self,
        user_feat_data: pd.DataFrame | None = None,
        item_feat_data: pd.DataFrame | None = None,
        user_sparse_cols: Sequence[str] | None = None,
        item_sparse_cols: Sequence[str] | None = None,
        user_dense_cols: Sequence[str] | None = None,
        item_dense_cols: Sequence[str] | None = None,
        user_multi_sparse_cols: Sequence[Sequence[str]] | None = None,
        item_multi_sparse_cols: Sequence[Sequence[str]] | None = None,
        user_pad_val: int | str | Sequence = "missing",
        item_pad_val: int | str | Sequence = "missing",
    ):
        """Process user and item features for model training.

        Parameters
        ----------
        user_feat_data : pd.DataFrame, optional
            DataFrame containing user features.
        item_feat_data : pd.DataFrame, optional
            DataFrame containing item features.
        user_sparse_cols : Sequence[str], optional
            Names of sparse categorical columns for users.
        item_sparse_cols : Sequence[str], optional
            Names of sparse categorical columns for items.
        user_dense_cols : Sequence[str], optional
            Names of dense numerical columns for users.
        item_dense_cols : Sequence[str], optional
            Names of dense numerical columns for items.
        user_multi_sparse_cols : Sequence[Sequence[str]], optional
            Names of multi-value sparse columns for users.
        item_multi_sparse_cols : Sequence[Sequence[str]], optional
            Names of multi-value sparse columns for items.
        user_pad_val : int, str, or Sequence, default: "missing"
            Padding value for user multi-sparse features.
        item_pad_val : int, str, or Sequence, default: "missing"
            Padding value for item multi-sparse features.

        Processing Flow
        ---------------
        1. Extract unique values for sparse columns from user/item feature data
        2. Extract unique values for multi-sparse columns, excluding padding values
        3. Build sparse_val_to_idx mapping: {column_name: {value: index}}
        4. For each feature DataFrame (user/item):
           a. Map raw IDs to internal indices
           b. Extract sparse feature indices as 1D arrays where position 0 holds OOV_IDX,
              and subsequent positions correspond to internal entity IDs (1, 2, 3, ...).
              This allows direct indexing: feat_array[entity_id] -> feature index.
           c. Extract dense feature indices as 1D arrays where position 0 holds the
              median value (for OOV entities), with the same indexing scheme as sparse.
           d. Extract multi-sparse feature indices as 2D matrices of shape
              (n_entities + 1, n_fields), where row 0 is the OOV row filled with
              OOV_IDX, and subsequent rows correspond to internal entity IDs.

        Side Effects
        ------------
        Sets `feat_info` attribute on the instance.
        """
        if not self.train_called:
            raise RuntimeError("Trainset must be built before processing features.")

        check_feat_cols(
            self.user_col_name,
            self.item_col_name,
            user_sparse_cols,
            item_sparse_cols,
            user_dense_cols,
            item_dense_cols,
        )

        sparse_unique_vals = self._get_sparse_unique_vals(
            user_feat_data, item_feat_data, user_sparse_cols, item_sparse_cols
        )
        multi_sparse_unique_vals = self._get_multi_sparse_unique_vals(
            user_feat_data,
            item_feat_data,
            user_multi_sparse_cols,
            item_multi_sparse_cols,
            user_pad_val,
            item_pad_val,
        )

        all_sparse_unique_vals = sparse_unique_vals | multi_sparse_unique_vals
        self.sparse_val_to_idx = {
            feat: self.make_id_mapping(unique_vals)
            for feat, unique_vals in all_sparse_unique_vals.items()
        }

        user_feat_unique = (
            self.extract_unique_features(
                user_feat_data,
                user_sparse_cols,
                user_dense_cols,
                user_multi_sparse_cols,
                name="user",
            )
            if user_feat_data is not None else {}
        )
        item_feat_unique = (
            self.extract_unique_features(
                item_feat_data,
                item_sparse_cols,
                item_dense_cols,
                item_multi_sparse_cols,
                name="item",
            )
            if item_feat_data is not None else {}
        )
        feat_unique = user_feat_unique | item_feat_unique

        self.feat_info = FeatInfo(
            user_sparse_cols,
            item_sparse_cols,
            user_dense_cols,
            item_dense_cols,
            user_multi_sparse_cols,
            item_multi_sparse_cols,
            feat_unique=feat_unique,
            sparse_val_to_idx=self.sparse_val_to_idx,
        )

    @staticmethod
    def _get_sparse_unique_vals(
        user_feat_data: pd.DataFrame | None,
        item_feat_data: pd.DataFrame | None,
        user_sparse_cols: Sequence[str] | None,
        item_sparse_cols: Sequence[str] | None,
    ) -> dict[str, np.ndarray]:
        """Extract sorted unique values for each sparse column."""

        def _compute_unique(
            feat_data: pd.DataFrame | None,
            sparse_cols: Sequence[str] | None,
            name: str,
        ) -> dict[str, np.ndarray]:
            result = {}
            if feat_data is None or not sparse_cols:
                return result
            for col in sparse_cols:
                if col in feat_data:
                    result[col] = np.sort(feat_data[col].unique())
                else:
                    raise ValueError(f"`{col}` does not exist in {name} feat data.")
            return result

        user_unique_vals = _compute_unique(user_feat_data, user_sparse_cols, "user")
        item_unique_vals = _compute_unique(item_feat_data, item_sparse_cols, "item")
        return user_unique_vals | item_unique_vals

    @staticmethod
    def _get_multi_sparse_unique_vals(
        user_feat_data: pd.DataFrame | None,
        item_feat_data: pd.DataFrame | None,
        user_multi_sparse_cols: Sequence[Sequence[str]] | None,
        item_multi_sparse_cols: Sequence[Sequence[str]] | None,
        user_pad_val: int | str | Sequence,
        item_pad_val: int | str | Sequence,
    ) -> dict[str, np.ndarray]:
        """Extract sorted unique values for multi-sparse columns, excluding padding.

        The returned dict uses the first column name in each field (field[0])
        as the representative key.
        """

        def _compute_unique(
            feat_data: pd.DataFrame | None,
            multi_sparse_cols: Sequence[Sequence[str]] | None,
            pad_val: int | str | list | tuple,
            name: str,
        ) -> dict[str, np.ndarray]:
            result = {}
            if feat_data is None or not multi_sparse_cols:
                return result

            for field in multi_sparse_cols:
                if not field:
                    raise ValueError(f"{name} multi_sparse_cols has invalid field: {field}")

            if not isinstance(pad_val, list | tuple):
                pad_val = [pad_val] * len(multi_sparse_cols)

            if len(multi_sparse_cols) != len(pad_val):
                raise ValueError("Length of `multi_sparse_col` and `pad_val` doesn't match")

            for i, field in enumerate(multi_sparse_cols):
                for col in field:
                    if col not in feat_data:
                        raise ValueError(f"`{col}` does not exist in {name} feat data.")

                values = feat_data[field].T.to_numpy().tolist()
                unique_vals = set(itertools.chain.from_iterable(values))
                if pad_val[i] in unique_vals:
                    unique_vals.remove(pad_val[i])
                # use name of a field's first column as representative
                result[field[0]] = np.sort(list(unique_vals))
            return result

        user_unique_vals = _compute_unique(
            user_feat_data, user_multi_sparse_cols, user_pad_val, "user"
        )
        item_unique_vals = _compute_unique(
            item_feat_data, item_multi_sparse_cols, item_pad_val, "item"
        )
        return user_unique_vals | item_unique_vals

    def extract_unique_features(
        self,
        feat_data: pd.DataFrame,
        sparse_cols: Sequence[str] | None,
        dense_cols: Sequence[str] | None,
        multi_sparse_cols: Sequence[Sequence[str]] | None,
        name: str,
    ) -> dict[str, np.ndarray]:
        """Extract unique feature values for each user or item.

        Maps raw IDs to internal indices and extracts feature values
        for sparse, dense, and multi-sparse columns.

        The `_map_ids` step sorts feat_data by internal ID, so row order matches
        ID order (1, 2, 3, ...). This enables direct array indexing:
        feat_array[entity_id] returns the feature for that entity.

        Parameters
        ----------
        feat_data : pd.DataFrame
            Feature data containing user/item ID and feature columns.
        sparse_cols : Sequence[str], optional
            Names of sparse categorical columns.
        dense_cols : Sequence[str], optional
            Names of dense numerical columns.
        multi_sparse_cols : Sequence[Sequence[str]], optional
            Names of multi-value sparse columns.
        name : str
            Either "user" or "item", used to determine ID mapping.

        Returns
        -------
        dict[str, np.ndarray]
            Dictionary mapping column names to unique feature arrays.
        """
        feat_data = self._map_ids(feat_data, name)
        sparse_unique = self._extract_sparse_unique(feat_data, sparse_cols)
        dense_unique = self._extract_dense_unique(feat_data, dense_cols)
        multi_sparse_unique = self._extract_multi_sparse_unique(
            feat_data, multi_sparse_cols
        )
        return {**sparse_unique, **dense_unique, **multi_sparse_unique}

    def _map_ids(self, feat_data: pd.DataFrame, name: str) -> pd.DataFrame:
        """Map raw user/item IDs to internal indices and validate coverage."""
        if name.endswith("user"):
            col_name = self.user_col_name
            unique_ids = self.id_converter.unique_users
            mapping = self.id_converter.user2id
        else:
            col_name = self.item_col_name
            unique_ids = self.id_converter.unique_items
            mapping = self.id_converter.item2id

        feat_ids = feat_data[col_name]
        if not feat_ids.is_unique:
            raise ValueError(f"{name} feat data must have unique ids.")

        feat_id_set = set(feat_ids.tolist())
        for i in unique_ids:
            if i not in feat_id_set:
                raise ValueError(f"id `{i}` does not exist in {name} feat data.")

        feat_data = feat_data.copy()
        feat_data[col_name] = feat_data[col_name].map(mapping)
        feat_data = feat_data.dropna(subset=[col_name]).sort_values(col_name)
        # NA value may result in float type, converting back to int
        feat_data[col_name] = feat_data[col_name].astype(np.int64)
        return feat_data

    def _extract_sparse_unique(
        self, feat_data: pd.DataFrame, sparse_cols: Sequence[str] | None
    ) -> dict[str, np.ndarray]:
        """Extract sparse feature arrays with OOV index at position 0."""
        feat_unique = {}
        if not sparse_cols:
            return feat_unique

        for col in sparse_cols:
            features = feat_data[col].tolist()
            mapping = self.sparse_val_to_idx[col]
            data = [mapping[v] for v in features]
            feat_unique[col] = np.array([OOV_IDX] + data)
        return feat_unique

    def _extract_multi_sparse_unique(
        self,
        feat_data: pd.DataFrame,
        multi_sparse_cols: Sequence[Sequence[str]] | None,
    ) -> dict[str, np.ndarray]:
        """Extract multi-sparse feature arrays as 2D matrices with OOV row at position 0."""
        feat_unique = {}
        if not multi_sparse_cols:
            return feat_unique

        for field in multi_sparse_cols:
            col_repr = field[0]
            shape = (len(feat_data) + 1, len(field))
            data = np.full(shape, OOV_IDX)
            for i, col in enumerate(field):
                features = feat_data[col].tolist()
                mapping = self.sparse_val_to_idx[col_repr]
                # may include pad val
                data[1:, i] = [mapping.get(v, OOV_IDX) for v in features]
            feat_unique[col_repr] = data
        return feat_unique

    @staticmethod
    def _extract_dense_unique(
        feat_data: pd.DataFrame, dense_cols: Sequence[str] | None
    ) -> dict[str, np.ndarray]:
        """Extract dense feature arrays with median as OOV value at position 0."""
        feat_unique = dict()
        if not dense_cols:
            return feat_unique

        for col in dense_cols:
            oov = feat_data[col].median()
            data = feat_data[col].tolist()
            feat_unique[col] = np.array([oov, *data], dtype=np.float32)
        return feat_unique

    @classmethod
    def for_retrain(
        cls,
        user_col_name: str,
        item_col_name: str,
        label_col_name: str | None = None,
        multi_label_col_names: Sequence[str] | None = None,
        shuffle: bool = False,
        pop_num: int = 100,
        seed: int = 42,
        *,
        model: RecModel,
    ) -> "HashDataset | ExtendableDataset":
        """Create a dataset for retraining from an existing model.
        
        This factory method creates the appropriate dataset type based on the model
        architecture and data representation requirements:
        - TorchBase models with hash-based data → HashDataset
        - CfBase models without hash-based data → ExtendableDataset

        Parameters
        ----------
        user_col_name : str
            Name of the user column in the data
        item_col_name : str
            Name of the item column in the data
        label_col_name : str or None, default: None
            Name of the label column in the data
        multi_label_col_names : Sequence[str] or None, default: None
            Names of multi-label columns in the data
        shuffle : bool, default: False
            Whether to shuffle the data
        pop_num : int, default: 100
            Number of popular items to consider
        seed : int, default: 42
            Random seed for reproducibility
        model : :type:`~endrs.types.RecModel`
            Existing trained model to create retraining dataset from

        Returns
        -------
        HashDataset or ExtendableDataset
            Dataset instance appropriate for the model type
        """
        from endrs.bases.cf_base import CfBase
        from endrs.bases.torch_base import TorchBase

        if not isinstance(model, TorchBase | CfBase):
            raise TypeError(
                f"Model must be a subclass of TorchBase or CfBase, got {type(model).__name__}"
            )

        if isinstance(model, TorchBase):
            if not model.data_info.use_hash:
                raise ValueError(
                    "TorchBase model must use hash-based data representation for retraining."
                )

            return HashDataset.for_retrain(
                model=model,
                user_col_name=user_col_name,
                item_col_name=item_col_name,
                label_col_name=label_col_name,
                multi_label_col_names=multi_label_col_names,
                shuffle=shuffle,
                pop_num=pop_num,
                seed=seed,
            )

        if isinstance(model, CfBase):
            if model.data_info.use_hash:
                raise ValueError(
                    "CfBase model cannot use hash-based data representation for retraining."
                )

            return ExtendableDataset(
                model=model,
                user_col_name=user_col_name,
                item_col_name=item_col_name,
                label_col_name=label_col_name,
                multi_label_col_names=multi_label_col_names,
                shuffle=shuffle,
                pop_num=pop_num,
                seed=seed,
            )


class ExtendableDataset(Dataset):
    """Dataset that can extend entity mappings from existing models for incremental training.

    This dataset class can take an existing model and extend its entity mappings
    with new users and items found in the new training data.
    """
    
    def __init__(
        self,
        model: "CfBase",
        user_col_name: str,
        item_col_name: str,
        label_col_name: str | None = None,
        multi_label_col_names: Sequence[str] | None = None,
        shuffle: bool = False,
        pop_num: int = 100,
        seed: int = 42,
    ):
        super().__init__(
            user_col_name,
            item_col_name,
            label_col_name,
            multi_label_col_names,
            shuffle,
            pop_num,
            seed,
        )
        self.model = model

    def build_data(
        self,
        train_data: pd.DataFrame,
        eval_data: pd.DataFrame | None = None,
        test_data: pd.DataFrame | None = None,
    ):
        """Build data and merge consumed history with the existing model.

        After calling the parent's build_data, this method merges the new
        consumed history with the model's existing consumed history and
        updates the model's data_info.
        """
        super().build_data(train_data, eval_data, test_data)
        user_consumed, item_consumed = merge_consumed_data(
            self.model.data_info.user_consumed,
            self.model.data_info.item_consumed,
            self.data_info.user_consumed,
            self.data_info.item_consumed,
        )
        self.data_info.user_consumed = user_consumed
        self.data_info.item_consumed = item_consumed
        self.model.update_data_info(self.data_info)

    def _build_id_converter(
        self, new_user_vals: np.ndarray, new_item_vals: np.ndarray
    ) -> IdConverter:
        """Extend existing ID converter with new users and items.

        New users/items not in the existing model are assigned IDs starting
        from max_existing_id + 1. The existing converter is updated in-place.
        """
        existing_converter = self.model.data_info.id_converter
        existing_users = np.array(list(existing_converter.user2id.keys()))
        existing_items = np.array(list(existing_converter.item2id.keys()))
        new_users = np.setdiff1d(new_user_vals, existing_users)
        new_items = np.setdiff1d(new_item_vals, existing_items)
        max_user_id = max(existing_converter.user2id.values())
        max_item_id = max(existing_converter.item2id.values())

        new_user2id = {}
        new_id2user = {}
        for i, user in enumerate(new_users, start=max_user_id + 1):
            new_user2id[user] = i
            new_id2user[i] = user

        new_item2id = {}
        new_id2item = {}
        for i, item in enumerate(new_items, start=max_item_id + 1):
            new_item2id[item] = i
            new_id2item[i] = item

        existing_converter.update(new_user2id, new_item2id, new_id2user, new_id2item)
        return existing_converter


class HashDataset(Dataset):
    """Dataset that uses hash-based ID mapping for dynamic vocabularies.

    This dataset hashes user and item IDs to fixed-size bins, enabling:
    - Unlimited vocabulary (any new user/item can be hashed)
    - Constant memory usage regardless of entity count
    - Incremental training without vocabulary extension

    The tradeoff is potential hash collisions and inability to retrieve
    original IDs from hash values.

    Parameters
    ----------
    user_col_name : str
        Name of the user column in the data.
    item_col_name : str
        Name of the item column in the data.
    label_col_name : str or None, default: None
        Name of the label column in the data.
    multi_label_col_names : Sequence[str] or None, default: None
        Names of multi-label columns for multi-task learning.
    shuffle : bool, default: False
        Whether to shuffle the data.
    pop_num : int, default: 100
        Number of popular items to track.
    seed : int, default: 42
        Random seed for reproducibility and hashing.
    n_hash_bins : int, default: DEFAULT_HASH_BINS (200_000)
        Number of hash bins for ID mapping. Larger values reduce collisions
        but increase memory usage.

    Notes
    -----
    Only sparse features are supported in hash mode. Dense and multi-sparse
    features are not available.

    For incremental training, use the :meth:`for_retrain` class method instead
    of the constructor.
    """

    def __init__(
        self,
        user_col_name: str,
        item_col_name: str,
        label_col_name: str | None = None,
        multi_label_col_names: Sequence[str] | None = None,
        shuffle: bool = False,
        pop_num: int = 100,
        seed: int = 42,
        n_hash_bins: int = DEFAULT_HASH_BINS,
    ):
        super().__init__(
            user_col_name,
            item_col_name,
            label_col_name,
            multi_label_col_names,
            shuffle,
            pop_num,
            seed,
        )
        self.n_hash_bins = n_hash_bins
        self.hasher = Hasher(n_hash_bins, seed)
        self.retrain = False
        self.model = None
        self.id_converter = None
        self.feat_unique = None
        self.sparse_val_to_idx = None

    @classmethod
    def for_retrain(
        cls,
        model: "TorchBase",
        user_col_name: str,
        item_col_name: str,
        label_col_name: str | None = None,
        multi_label_col_names: Sequence[str] | None = None,
        shuffle: bool = False,
        pop_num: int = 100,
        seed: int = 42,
    ) -> "HashDataset":
        """Create a HashDataset configured for incremental training.

        This factory method copies n_hash_bins, id_converter, feat_unique, and
        sparse_val_to_idx from the existing model, enabling the new dataset to
        extend the model's vocabulary and features.

        Parameters
        ----------
        model : TorchBase
            Existing trained model to create retraining dataset from.
        user_col_name : str
            Name of the user column in the data.
        item_col_name : str
            Name of the item column in the data.
        label_col_name : str or None, default: None
            Name of the label column in the data.
        multi_label_col_names : Sequence[str] or None, default: None
            Names of multi-label columns for multi-task learning.
        shuffle : bool, default: False
            Whether to shuffle the data.
        pop_num : int, default: 100
            Number of popular items to track.
        seed : int, default: 42
            Random seed for reproducibility and hashing.

        Returns
        -------
        HashDataset
            Dataset instance configured for incremental training.
        """
        dataset = cls(
            user_col_name,
            item_col_name,
            label_col_name,
            multi_label_col_names,
            shuffle,
            pop_num,
            seed,
            n_hash_bins=model.data_info.n_hash_bins,
        )
        dataset.retrain = True
        dataset.model = model
        dataset.id_converter = model.data_info.id_converter
        dataset.feat_unique = model.feat_info.feat_unique if model.feat_info else None
        dataset.sparse_val_to_idx = (
            model.feat_info.sparse_val_to_idx if model.feat_info else None
        )
        return dataset

    def build_data(
        self,
        train_data: pd.DataFrame,
        eval_data: pd.DataFrame | None = None,
        test_data: pd.DataFrame | None = None,
    ):
        """Build data with hash-based ID mapping."""
        super().build_data(train_data, eval_data, test_data)
        self.data_info.use_hash = True
        self.data_info.n_hash_bins = self.n_hash_bins
        if self.retrain:
            user_consumed, item_consumed = merge_consumed_data(
                self.model.data_info.user_consumed,
                self.model.data_info.item_consumed,
                self.data_info.user_consumed,
                self.data_info.item_consumed,
            )
            self.data_info.user_consumed = user_consumed
            self.data_info.item_consumed = item_consumed
            self.model.update_data_info(self.data_info)

    def _build_id_converter(
        self, user_unique_vals: np.ndarray, item_unique_vals: np.ndarray
    ) -> IdConverter:
        """Build ID converter using hash-based mapping.

        Unlike the parent class which assigns sequential IDs, this method hashes
        raw IDs to fixed bins. In retrain mode, the existing converter is updated
        with new hash mappings rather than creating a new one.
        """
        user2id, id2user = self.hasher.to_hash_mapping(
            USER_KEY, user_unique_vals.tolist(), include_reverse=True
        )
        item2id, id2item = self.hasher.to_hash_mapping(
            ITEM_KEY, item_unique_vals.tolist(), include_reverse=True
        )
        if self.retrain:
            if self.id_converter is None:
                raise RuntimeError(
                    "id_converter must be set before calling this method in retrain mode"
                )
            return self.id_converter.update(user2id, item2id, id2user, id2item)
        else:
            return IdConverter(user2id, item2id, id2user, id2item)

    def _build_feat_unique(
        self, feat_data: pd.DataFrame, sparse_cols: Sequence[str], name: str
    ):
        col_name = self.user_col_name if name == "user" else self.item_col_name
        feat_data = self._map_ids(feat_data, name)
        ids = feat_data[col_name].to_numpy()
        for col in sparse_cols:
            mapping = self.sparse_val_to_idx[col]
            data = feat_data[col].map(mapping).to_numpy()
            if self.retrain:
                all_hash_data = self.feat_unique[col]
            else:
                all_hash_data = np.full(self.n_hash_bins + 1, OOV_IDX)

            all_hash_data[ids] = data
            self.feat_unique[col] = all_hash_data

    def process_features(
        self,
        user_feat_data: pd.DataFrame | None = None,
        item_feat_data: pd.DataFrame | None = None,
        user_sparse_cols: Sequence[str] | None = None,
        item_sparse_cols: Sequence[str] | None = None,
        *_,
    ):
        if not self.train_called:
            raise RuntimeError("Trainset must be built before processing features.")
        check_feat_cols(
            self.user_col_name, self.item_col_name, user_sparse_cols, item_sparse_cols
        )
        sparse_unique_vals = self._get_sparse_unique_vals(
            user_feat_data, item_feat_data, user_sparse_cols, item_sparse_cols
        )
        if self.retrain:
            if self.feat_unique is None or self.sparse_val_to_idx is None:
                raise ValueError(
                    "Cannot add features during retrain if original model had no features"
                )
            for col, unique_vals in sparse_unique_vals.items():
                val_to_idx = self.hasher.to_hash_mapping(col, unique_vals.tolist())
                self.sparse_val_to_idx[col].update(val_to_idx)
        else:
            self.feat_unique = dict()
            self.sparse_val_to_idx = {
                col: self.hasher.to_hash_mapping(col, unique_vals.tolist())
                for col, unique_vals in sparse_unique_vals.items()
            }

        if user_feat_data is not None and user_sparse_cols:
            self._build_feat_unique(user_feat_data, user_sparse_cols, "user")
        if item_feat_data is not None and item_sparse_cols:
            self._build_feat_unique(item_feat_data, item_sparse_cols, "item")

        self.feat_info = FeatInfo(
            user_sparse_cols,
            item_sparse_cols,
            feat_unique=self.feat_unique or None,
            sparse_val_to_idx=self.sparse_val_to_idx or None,
        )

        if self.retrain:
            self.model.feat_info = self.feat_info
