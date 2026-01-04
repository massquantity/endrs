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
from endrs.utils.constants import ITEM_KEY, OOV_IDX, USER_KEY
from endrs.utils.hashing import Hasher
from endrs.utils.validate import check_data_cols, check_feat_cols

if TYPE_CHECKING:
    from endrs.bases.cf_base import CfBase
    from endrs.bases.torch_base import TorchBase


class Dataset:
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
        self.feat_info = None

    def shuffle_data(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data.sample(frac=1, random_state=self.seed)
        return data.reset_index(drop=True)

    def build_data(
        self,
        train_data: pd.DataFrame,
        eval_data: pd.DataFrame | None = None,
        test_data: pd.DataFrame | None = None,
    ):
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
        user_indices = data[user_col_name].map(id_converter.safe_user_to_id).tolist()
        item_indices = data[item_col_name].map(id_converter.safe_item_to_id).tolist()

        labels, multi_labels = None, None
        if label_col_name:
            labels = data[label_col_name].astype("float32").tolist()
        if multi_label_col_names:
            multi_labels = data[multi_label_col_names].astype("float32")
            multi_labels = multi_labels.to_numpy().tolist()

        if is_train:
            return BatchData(user_indices, item_indices, labels, multi_labels)
        else:
            return EvalBatchData(user_indices, item_indices, labels, multi_labels)

    def _build_id_converter(
        self, user_unique_vals: np.ndarray, item_unique_vals: np.ndarray
    ) -> IdConverter:
        user2id, id2user = self.make_id_mapping(user_unique_vals, include_reverse=True)
        item2id, id2item = self.make_id_mapping(item_unique_vals, include_reverse=True)
        return IdConverter(user2id, item2id, id2user, id2item)

    @staticmethod
    def make_id_mapping(unique_values: np.ndarray, include_reverse: bool = False):
        # skip oov id 0
        unique_values = unique_values.tolist()
        ids = range(1, len(unique_values) + 1)
        if include_reverse:
            return dict(zip(unique_values, ids)), dict(zip(ids, unique_values))
        else:
            return dict(zip(unique_values, ids))

    def _get_popular_items(self, train_data: pd.DataFrame) -> list[ItemId]:
        count_items = (
            train_data.drop_duplicates(subset=[self.user_col_name, self.item_col_name])
            .groupby(self.item_col_name)[self.user_col_name]
            .count()
        )
        selected_items = count_items.sort_values(ascending=False).index.tolist()
        # if not enough items, add old populars
        # if len(selected_items) < num and self.old_info is not None:
        #    diff = num - len(selected_items)
        #    selected_items.extend(self.old_info.popular_items[:diff])
        return selected_items[:self.pop_num]

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
        all_sparse_unique_vals = {**sparse_unique_vals, **multi_sparse_unique_vals}
        self.sparse_val_to_idx = {
            feat: self.make_id_mapping(unique_vals)
            for feat, unique_vals in all_sparse_unique_vals.items()
        }

        feat_unique = dict()
        if user_feat_data is not None:
            feat_unique.update(
                self.extract_unique_features(
                    user_feat_data,
                    user_sparse_cols,
                    user_dense_cols,
                    user_multi_sparse_cols,
                    name="user",
                )
            )
        if item_feat_data is not None:
            feat_unique.update(
                self.extract_unique_features(
                    item_feat_data,
                    item_sparse_cols,
                    item_dense_cols,
                    item_multi_sparse_cols,
                    name="item",
                )
            )

        self.feat_info = FeatInfo(
            user_sparse_cols,
            item_sparse_cols,
            user_dense_cols,
            item_dense_cols,
            user_multi_sparse_cols,
            item_multi_sparse_cols,
            feat_unique=feat_unique or None,
            sparse_val_to_idx=self.sparse_val_to_idx or None,
        )

    @staticmethod
    def _get_sparse_unique_vals(
        user_feat_data: pd.DataFrame | None,
        item_feat_data: pd.DataFrame | None,
        user_sparse_cols: Sequence[str] | None,
        item_sparse_cols: Sequence[str] | None,
    ) -> dict[str, np.ndarray]:
        def _compute_unique(feat_data, sparse_cols, name):
            if feat_data is None or not sparse_cols:
                return
            for col in sparse_cols:
                if col in feat_data:
                    sparse_unique_vals[col] = np.sort(feat_data[col].unique())
                else:
                    raise ValueError(f"`{col}` does not exist in {name} feat data.")

        sparse_unique_vals = dict()
        _compute_unique(user_feat_data, user_sparse_cols, "user")
        _compute_unique(item_feat_data, item_sparse_cols, "item")
        return sparse_unique_vals

    @staticmethod
    def _get_multi_sparse_unique_vals(
        user_feat_data: pd.DataFrame | None,
        item_feat_data: pd.DataFrame | None,
        user_multi_sparse_cols: Sequence[Sequence[str]] | None,
        item_multi_sparse_cols: Sequence[Sequence[str]] | None,
        user_pad_val: int | str | Sequence,
        item_pad_val: int | str | Sequence,
    ) -> dict[str, np.ndarray]:
        def _compute_unique(
            feat_data: pd.DataFrame | None,
            multi_sparse_cols: Sequence[Sequence[str]] | None,
            pad_val: int | str | list | tuple,
            name: str,
        ):
            if feat_data is None or not multi_sparse_cols:
                return
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
                multi_sparse_unique_vals[field[0]] = np.sort(list(unique_vals))

        multi_sparse_unique_vals = dict()
        _compute_unique(user_feat_data, user_multi_sparse_cols, user_pad_val, "user")
        _compute_unique(item_feat_data, item_multi_sparse_cols, item_pad_val, "item")
        return multi_sparse_unique_vals

    def extract_unique_features(
        self,
        feat_data: pd.DataFrame,
        sparse_cols: Sequence[str] | None,
        dense_cols: Sequence[str] | None,
        multi_sparse_cols: Sequence[Sequence[str]] | None,
        name: str,
    ) -> dict[str, np.ndarray]:
        feat_data = self._map_ids(feat_data, name)
        sparse_unique = self._extract_sparse_unique(feat_data, sparse_cols)
        dense_unique = self._extract_dense_unique(feat_data, dense_cols)
        multi_sparse_unique = self._extract_multi_sparse_unique(
            feat_data, multi_sparse_cols
        )
        return {**sparse_unique, **dense_unique, **multi_sparse_unique}

    def _map_ids(self, feat_data: pd.DataFrame, name: str) -> pd.DataFrame:
        if name.endswith("user"):
            col_name = self.user_col_name
            unique_ids = self.id_converter.unique_users
            mapping = self.id_converter.user2id
        else:
            col_name = self.item_col_name
            unique_ids = self.id_converter.unique_items
            mapping = self.id_converter.item2id

        feat_ids = feat_data[col_name]
        assert feat_ids.is_unique, f"{name} feat data must have unique ids."
        feat_id_set = set(feat_ids.tolist())
        for i in unique_ids:
            if i not in feat_id_set:
                raise ValueError(f"id `{i}` does not exist in {name} feat data.")

        feat_data = feat_data.copy()
        feat_data[col_name] = feat_data[col_name].map(mapping)
        feat_data = feat_data.dropna(subset=[col_name]).sort_values(col_name)
        # NA value may result in float type
        feat_data[col_name] = feat_data[col_name].astype("int64")
        return feat_data

    def _extract_sparse_unique(
        self, feat_data: pd.DataFrame, sparse_cols: Sequence[str] | None
    ) -> dict[str, np.ndarray]:
        feat_unique = dict()
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
        feat_unique = dict()
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

        if not isinstance(model, (TorchBase, CfBase)):
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
        """Extend existing ID converter with new users and items."""
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
    n_hash_bins : int, default: 200_000
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
        n_hash_bins: int = 200_000,
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
        user2id, id2user = self.hasher.to_hash_mapping(
            USER_KEY, user_unique_vals.tolist(), include_reverse=True
        )
        item2id, id2item = self.hasher.to_hash_mapping(
            ITEM_KEY, item_unique_vals.tolist(), include_reverse=True
        )
        if self.retrain:
            assert self.id_converter is not None
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
