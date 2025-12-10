from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch


class FeatInfo:
    """A class for storing and updating feature information.

    This class handles different types of features for users and items, including
    sparse features, dense features, and multi-sparse features. It also manages
    mappings between feature values and indices for efficient processing.

    Parameters
    ----------
    user_sparse_feats : Sequence[str] or None, default: None
        List of user sparse feature names.
    item_sparse_feats : Sequence[str] or None, default: None
        List of item sparse feature names.
    user_dense_feats : Sequence[str] or None, default: None
        List of user dense feature names.
    item_dense_feats : Sequence[str] or None, default: None
        List of item dense feature names.
    user_multi_sparse_feats : Sequence[Sequence[str]] or None, default: None
        List of user multi-sparse feature groups, where each group is a sequence of feature names.
    item_multi_sparse_feats : Sequence[Sequence[str]] or None, default: None
        List of item multi-sparse feature groups, where each group is a sequence of feature names.
    feat_unique : Mapping[str, np.ndarray] or None, default: None
        Dictionary mapping feature names to arrays of unique feature indices.
        This is used to get feature indices for every user or item, and arrays are 1D.
    sparse_val_to_idx : Mapping[str, Mapping[Any, int]] or None, default: None
        Nested dictionary mapping feature names to value-to-index mappings
        (e.g., {"color": {"red": 0, "blue": 1}}).

    Attributes
    ----------
    multi_sparse_feat_to_main : dict[str, tuple[str, int]]
        A mapping of multi-sparse feature names to a tuple of its group's first feature and its index.
    """

    def __init__(
        self,
        user_sparse_feats: Sequence[str] | None = None,
        item_sparse_feats: Sequence[str] | None = None,
        user_dense_feats: Sequence[str] | None = None,
        item_dense_feats: Sequence[str] | None = None,
        user_multi_sparse_feats: Sequence[Sequence[str]] | None = None,
        item_multi_sparse_feats: Sequence[Sequence[str]] | None = None,
        feat_unique: Mapping[str, np.ndarray] | None = None,
        sparse_val_to_idx: Mapping[str, Mapping[Any, int]] | None = None,
    ):
        self.user_sparse_feats = user_sparse_feats or []
        self.item_sparse_feats = item_sparse_feats or []
        self.user_dense_feats = user_dense_feats or []
        self.item_dense_feats = item_dense_feats or []
        self.user_multi_sparse_all = user_multi_sparse_feats or []
        self.item_multi_sparse_all = item_multi_sparse_feats or []
        self.feat_unique = feat_unique
        self.sparse_val_to_idx = sparse_val_to_idx
        if user_multi_sparse_feats:
            self.multi_sparse_feat_to_main = self.mapping_multi_sparse_feat(
                self.user_multi_sparse_all
            )

    @property
    def has_sparse_feats(self) -> bool:
        """Check if there are any sparse features."""
        return bool(self.user_sparse_feats or self.item_sparse_feats)

    @property
    def has_multi_sparse_feats(self) -> bool:
        """Check if there are any multi-sparse features."""
        return bool(self.user_multi_sparse_all or self.item_multi_sparse_all)

    @property
    def all_feats(self) -> list[str]:
        """Get all feature names."""
        return self.user_feats + self.item_feats

    @property
    def user_feats(self) -> list[str]:
        """Get all user feature names."""
        return (
            self.user_sparse_feats +
            self.user_dense_feats +
            self.user_multi_sparse_feats
        )

    @property
    def item_feats(self) -> list[str]:
        """Get all item feature names."""
        return (
            self.item_sparse_feats +
            self.item_dense_feats +
            self.item_multi_sparse_feats
        )

    @property
    def user_multi_sparse_feats(self) -> list[str]:
        """Get all user multi-sparse feature names.

        Returns a list of representative feature names for user multi-sparse fields,
        where each name is the first column of its respective field.
        """
        return [field[0] for field in self.user_multi_sparse_all]

    @property
    def item_multi_sparse_feats(self) -> list[str]:
        """Get all item multi-sparse feature names.

        Returns a list of representative feature names for item multi-sparse fields,
        where each name is the first column of its respective field.
        """
        return [field[0] for field in self.item_multi_sparse_all]

    @staticmethod
    def mapping_multi_sparse_feat(
        multi_sparse_feats: Sequence[Sequence[str]]
    ) -> dict[str, tuple[str, int]]:
        """Map each multi-sparse feature name to a tuple of its group's first feature and its index.

        Parameters
        ----------
        multi_sparse_feats : Sequence[Sequence[str]]
            A list of multi-sparse feature groups, where each group is a sequence of feature names.

        Returns
        -------
        multi_sparse_map : dict[str, tuple[str, int]]
            A mapping of multi-sparse feature names to a tuple of its group's first feature and its index.
        """
        multi_sparse_map = dict()
        for field in multi_sparse_feats:
            for i, col in enumerate(field):
                multi_sparse_map[col] = (field[0], i)
        return multi_sparse_map

    def sparse_size(self, feat: str) -> int:
        """Get the size of a sparse feature."""
        return len(self.sparse_val_to_idx[feat])

    def set_user_features(
        self, inputs: Mapping[str, torch.Tensor], user_feats: Mapping[str, Any]
    ) -> dict[str, torch.Tensor]:
        """Set the features for a user, mainly used in dynamic recommendation.

        Parameters
        ----------
        inputs : Mapping[str, torch.Tensor]
            The input tensor.
        user_feats : Mapping[str, Any]
            A mapping of feature names to their values.

        Returns
        -------
        inputs : Mapping[str, torch.Tensor]
            The input tensor with the features set.
        """ 
        for feat, val in user_feats.items():
            if (
                feat in self.user_sparse_feats
                and val in self.sparse_val_to_idx[feat]
            ):
                # shape: [1, 1] or [B, 1]
                inputs[feat][:] = self.sparse_val_to_idx[feat][val]
            elif feat in self.user_dense_feats:
                inputs[feat][:] = val
            elif (
                self.user_multi_sparse_feats
                and feat in self.multi_sparse_feat_to_main
            ):
                main_feat, feat_idx = self.multi_sparse_feat_to_main[feat]
                if val in self.sparse_val_to_idx[main_feat]:
                    idx = self.sparse_val_to_idx[main_feat][val]
                    # shape: [1, d] or [B, d]
                    inputs[main_feat][:, feat_idx] = idx

        return inputs
