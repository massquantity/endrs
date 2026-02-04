import functools
from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from endrs.data.batch import BatchData, EvalBatchData
from endrs.data.consumed import interaction_consumed, merge_consumed_data
from endrs.data.data_info import DataInfo, IdConverter
from endrs.feature.config import FeatureConfig
from endrs.feature.processor import FeatureProcessor, HashFeatureProcessor
from endrs.types import ItemId, RecModel
from endrs.utils.constants import DEFAULT_HASH_BINS, ITEM_KEY, USER_KEY
from endrs.utils.hashing import Hasher
from endrs.utils.validate import check_data_cols, check_feat_cols, check_feat_data

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
        require_complete_coverage: bool = True,
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
        require_complete_coverage : bool, default: True
            If True, require all entity IDs from interaction data to have features.
            Set to False to allow partial feature coverage (e.g., only providing
            features for popular items).

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
        check_feat_data(
            user_feat_data,
            item_feat_data,
            user_sparse_cols,
            item_sparse_cols,
            user_dense_cols,
            item_dense_cols,
            user_multi_sparse_cols,
            item_multi_sparse_cols,
        )

        config = FeatureConfig.from_flat_params(
            user_sparse_cols=user_sparse_cols,
            item_sparse_cols=item_sparse_cols,
            user_dense_cols=user_dense_cols,
            item_dense_cols=item_dense_cols,
            user_multi_sparse_cols=user_multi_sparse_cols,
            item_multi_sparse_cols=item_multi_sparse_cols,
            user_pad_val=user_pad_val,
            item_pad_val=item_pad_val,
        )

        processor = FeatureProcessor(
            self.id_converter,
            self.user_col_name,
            self.item_col_name,
            require_complete_coverage,
        )
        self.feat_info = processor.process(user_feat_data, item_feat_data, config)

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

    def process_features(
        self,
        user_feat_data: pd.DataFrame | None = None,
        item_feat_data: pd.DataFrame | None = None,
        user_sparse_cols: Sequence[str] | None = None,
        item_sparse_cols: Sequence[str] | None = None,
        *_,
    ):
        """Process sparse features using hash-based value mapping.

        Only sparse features are supported in hash mode. Dense and multi-sparse
        features are ignored (captured by *_).

        Processing flow:
        1. Extract unique values for each sparse column
        2. Build/update sparse_val_to_idx with hash mappings:
           - New training: create fresh hash mappings for all values
           - Retrain: merge new values' hash mappings into existing mappings
        3. Initialize feat_unique (new training only; retrain uses model's copy)
        4. Build feat_unique arrays via _build_feat_unique
        5. Create FeatInfo (in retrain mode, also update model's feat_info)

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
        *_ : ignored
            Dense and multi-sparse parameters are ignored in hash mode.
        """
        self._validate_feature_preconditions(
            user_feat_data, item_feat_data, user_sparse_cols, item_sparse_cols
        )

        config = FeatureConfig.from_flat_params(
            user_sparse_cols=user_sparse_cols,
            item_sparse_cols=item_sparse_cols,
        )

        processor = HashFeatureProcessor(
            self.hasher,
            self.id_converter,
            self.user_col_name,
            self.item_col_name,
            self.n_hash_bins,
            retrain=self.retrain,
            existing_sparse_val_to_idx=self.sparse_val_to_idx,
            existing_feat_unique=self.feat_unique,
        )
        self.feat_info = processor.process(user_feat_data, item_feat_data, config)
        self.sparse_val_to_idx = processor.sparse_val_to_idx
        self.feat_unique = processor.feat_unique

        if self.retrain:
            self.model.feat_info = self.feat_info

    def _validate_feature_preconditions(
        self,
        user_feat_data: pd.DataFrame | None,
        item_feat_data: pd.DataFrame | None,
        user_sparse_cols: Sequence[str] | None,
        item_sparse_cols: Sequence[str] | None,
    ):
        """Validate preconditions for feature processing.

        Checks:
        1. Training data must be built first
        2. Feature columns don't conflict with user/item ID columns
        3. Feature columns require corresponding feat_data
        4. In retrain mode, original model must have features
        5. In retrain mode, feature columns must match original model
        """
        if not self.train_called:
            raise RuntimeError("Trainset must be built before processing features.")

        check_feat_cols(
            self.user_col_name, self.item_col_name, user_sparse_cols, item_sparse_cols
        )
        check_feat_data(
            user_feat_data, item_feat_data, user_sparse_cols, item_sparse_cols
        )

        # Remaining checks only apply to retrain mode
        if not self.retrain:
            return
        if self.feat_unique is None or self.sparse_val_to_idx is None:
            raise ValueError(
                "Cannot add features during retrain if original model had no features"
            )

        original_cols = set(self.sparse_val_to_idx.keys())
        new_cols = set(user_sparse_cols or []) | set(item_sparse_cols or [])
        if new_cols != original_cols:
            missing = original_cols - new_cols
            extra = new_cols - original_cols
            raise ValueError(
                f"Feature columns must match original model in retrain mode. "
                f"Missing: {missing or 'none'}, Extra: {extra or 'none'}"
            )
