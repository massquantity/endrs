"""Base class for collaborative filtering models (UserCF, ItemCF, Swing)."""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Self

import joblib
import numpy as np
import pandas as pd

from endrs.data.batch import BatchData, EvalBatchData
from endrs.data.data_info import DataInfo
from endrs.evaluation.evaluator import Evaluator
from endrs.inference.cold_start import popular_recommendations
from endrs.inference.postprocess import construct_rec
from endrs.inference.preprocess import convert_ids, get_unknown, sep_unknown_users
from endrs.types import ItemId, RustModel, UserId
from endrs.utils.logger import normal_log
from endrs.utils.misc import show_start_time, time_block
from endrs.utils.sparse import construct_sparse
from endrs.utils.validate import check_labels
from endrs_ext import (
    load_item_cf,
    load_swing,
    load_user_cf,
    save_item_cf,
    save_swing,
    save_user_cf,
)


@dataclass(frozen=True)
class _ModelConfig:
    """Configuration for a specific CF model type."""

    display_name: str
    save_fn: Callable[[Any, str, str], None]
    load_fn: Callable[[str, str], RustModel]


_MODEL_CONFIGS: dict[str, _ModelConfig] = {
    "user_cf": _ModelConfig(
        display_name="UserCF",
        save_fn=save_user_cf,
        load_fn=load_user_cf,
    ),
    "item_cf": _ModelConfig(
        display_name="ItemCF",
        save_fn=save_item_cf,
        load_fn=load_item_cf,
    ),
    "swing": _ModelConfig(
        display_name="Swing",
        save_fn=save_swing,
        load_fn=load_swing,
    ),
}


class CfBase:
    """Internal base class for collaborative filtering models.

    This class should not be instantiated directly. Use UserCF, ItemCF, or Swing instead.
    """

    model_type: str

    def __init__(
        self,
        task: str,
        data_info: DataInfo,
        num_threads: int = 1,
        seed: int = 42,
    ):
        if not hasattr(self, "model_type") or self.model_type not in _MODEL_CONFIGS:
            raise TypeError(
                f"{self.__class__.__name__} cannot be instantiated directly. "
                "Use UserCF, ItemCF, or Swing instead."
            )

        self._cfg = _MODEL_CONFIGS[self.model_type]
        self.task = task
        self.data_info = data_info
        self.n_users = data_info.n_users
        self.n_items = data_info.n_items
        self.user_consumed = data_info.user_consumed
        self.id_converter = data_info.id_converter
        self.num_threads = num_threads
        self.seed = seed
        self.np_rng = np.random.default_rng(seed)
        self.rs_model: RustModel | None = None

    def _create_rust_model(
        self,
        user_interacts: list[list[tuple[int, float]]],
        item_interacts: list[list[tuple[int, float]]],
    ) -> RustModel:
        """Create the Rust model. Subclasses must override this method."""
        raise NotImplementedError("Subclasses must implement _create_rust_model")

    def fit(
        self,
        train_data: BatchData,
        neg_sampling: bool,
        verbose: int = 2,
        eval_data: EvalBatchData | None = None,
        metrics: Sequence[str] | None = None,
        k: int = 10,
        eval_batch_size: int = 8192,
        eval_user_num: int | None = None,
    ):
        """Train the model on the provided training data.

        Parameters
        ----------
        train_data : :class:`~endrs.data.BatchData`
            Training data.
        neg_sampling : bool
            Whether to use negative sampling.

            .. NOTE::
               Negative sampling is needed if your data is implicit(i.e., `task` is ranking)
               and ONLY contains positive labels. Otherwise, it should be False.

        verbose : int, default: 2
            Verbosity level (0=quiet, 1=epoch results, 2=progress bar).

            - ``verbose <= 0`` : Print nothing.
            - ``verbose >= 1`` : Print evaluation metrics if ``eval_data`` is provided.
            - ``verbose >= 2`` : Enable progress bar.

        eval_data : :class:`~endrs.data.EvalBatchData` or None, default: None
            Evaluation data for validation during training.
        metrics : Sequence[str] or None, default: None
            Evaluation metrics to calculate during validation.
        k : int, default: 10
            Number of items to recommend for evaluation metrics.
        eval_batch_size : int, default: 8192
            Batch size for evaluation.
        eval_user_num : int or None, default: None
            Number of users to sample for evaluation.
            Setting it to a positive number will sample users randomly from eval data.
        """
        check_labels(self.task, train_data.labels, neg_sampling)
        show_start_time()
        user_interacts, item_interacts = construct_sparse(train_data)

        if self.rs_model is None:
            self.rs_model = self._create_rust_model(user_interacts, item_interacts)
            with time_block("computing similarities", verbose=1):
                self.rs_model.compute_similarities(self.num_threads)
        else:
            with time_block("updating similarities", verbose=1):
                self.rs_model.update_similarities(
                    user_interacts, item_interacts, self.num_threads
                )

        self._log_similarity_stats()

        if eval_data is not None and verbose >= 1:
            evaluator = Evaluator(
                during_training=False,
                data=eval_data,
                model=self,
                is_multi_task=False,
                neg_sampling=neg_sampling,
                metrics=metrics,
                k=k,
                eval_batch_size=eval_batch_size,
                sample_user_num=eval_user_num,
                seed=self.seed,
                verbose=verbose,
            )
            evaluator.log_metrics(epoch=0)

    def _log_similarity_stats(self) -> None:
        """Log similarity matrix statistics."""
        num = self.rs_model.num_sim_elements()
        base = self.n_users if self._cfg.display_name == "UserCF" else self.n_items
        density_ratio = 100 * num / base / base
        normal_log(
            f"{self._cfg.display_name} num_elements: {num}, density: {density_ratio:5.4f} %"
        )

    def evaluate(
        self,
        eval_data: EvalBatchData | pd.DataFrame,
        neg_sampling: bool,
        metrics: Sequence[str] | None = None,
        k: int = 10,
        eval_batch_size: int = 8192,
        eval_user_num: int | None = None,
        verbose: int = 2,
    ) -> dict[str, float]:
        """Evaluate the trained model on evaluation data.

        Parameters
        ----------
        eval_data : :class:`~endrs.data.EvalBatchData` or :class:`pandas.DataFrame`
            The evaluation dataset containing user-item interactions.
        neg_sampling : bool
            Whether to use negative sampling.
        metrics : str or Sequence[str] or None, default: None
            Evaluation metrics to calculate.
        k : int, default: 10
            Number of items to recommend for ranking metrics (e.g., Precision@k).
        eval_batch_size : int, default: 8192
            Batch size for evaluation.
        eval_user_num : int or None, default: None
            Number of users to sample for evaluation. If None, all users are used.
        verbose : int, default: 2
            Verbosity level for evaluation progress.

        Returns
        -------
        dict[str, float]
            Dictionary containing computed evaluation metrics.

        Raises
        ------
        RuntimeError
            If the model hasn't been trained yet.
        """
        if self.rs_model is None:
            raise RuntimeError(
                "Model must be trained before evaluation. Call fit() first."
            )

        evaluator = Evaluator(
            during_training=False,
            data=eval_data,
            model=self,
            is_multi_task=False,
            neg_sampling=neg_sampling,
            metrics=metrics,
            k=k,
            eval_batch_size=eval_batch_size,
            sample_user_num=eval_user_num,
            seed=self.seed,
            verbose=verbose,
        )
        return evaluator.compute_eval_results()

    def predict(
        self,
        user: UserId | list[UserId] | np.ndarray,
        item: ItemId | list[ItemId] | np.ndarray,
        cold_start: str = "popular",
        inner_id: bool = False,
    ) -> list[float]:
        """Predict ratings or scores for user-item pairs.

        Parameters
        ----------
        user : :type:`~endrs.types.UserId` or list[UserId] or np.ndarray
            User id or batch of user ids.
        item : :type:`~endrs.types.ItemId` or list[ItemId] or np.ndarray
            Item id or batch of item ids.
        cold_start : {'popular', 'default'}, default: 'popular'
            Strategy for handling cold-start users/items.
        inner_id : bool, default: False
            Whether the provided IDs are internal IDs.

        Returns
        -------
        list[float]
            Predicted scores for each user-item pair.
        """
        user, item = convert_ids(user, item, self.id_converter, inner_id)
        unknown_num, _ = get_unknown(user, item)
        if unknown_num > 0 and cold_start != "popular":
            raise ValueError(
                f"{self._cfg.display_name} only supports popular strategy"
            )

        return self.rs_model.predict(user.tolist(), item.tolist())

    def recommend_user(
        self,
        user: UserId | list[UserId] | np.ndarray,
        n_rec: int,
        cold_start: str = "popular",
        inner_id: bool = False,
        filter_consumed: bool = True,
        random_rec: bool = False,
    ) -> dict[UserId, list[ItemId]]:
        """Recommend a list of items for given user(s).

        Parameters
        ----------
        user : :type:`~endrs.types.UserId` or list[UserId] or np.ndarray
            User id or a batch of user ids to recommend.
        n_rec : int
            Number of recommendations to generate.
        cold_start : str, default: 'popular'
            Strategy for handling cold-start users.
        inner_id : bool, default: False
            Whether to use inner_id defined in `endrs`.
        filter_consumed : bool, default: True
            Whether to filter out items the user has already consumed.
        random_rec : bool, default: False
            Whether to add randomness to recommendations.

        Returns
        -------
        dict[UserId, list[ItemId]]
            Dictionary mapping user IDs to list of recommended item IDs.
        """
        result_recs = dict()
        user_ids, unknown_users = sep_unknown_users(self.id_converter, user, inner_id)

        if unknown_users:
            if cold_start != "popular":
                raise ValueError(
                    f"{self._cfg.display_name} only supports `popular` cold start strategy"
                )
            for u in unknown_users:
                result_recs[u] = popular_recommendations(
                    self.data_info, inner_id, n_rec, self.np_rng
                )

        if user_ids:
            computed_recs, additional_rec_counts = self.rs_model.recommend(
                user_ids,
                n_rec,
                filter_consumed,
                random_rec,
            )
            for i, arc in enumerate(additional_rec_counts):
                if arc > 0:
                    additional_recs = popular_recommendations(
                        self.data_info, inner_id=True, n_rec=arc, np_rng=self.np_rng
                    )
                    computed_recs[i].extend(additional_recs)

            user_recs = construct_rec(
                self.id_converter, user_ids, computed_recs, inner_id
            )
            result_recs.update(user_recs)

        return result_recs

    def save(self, path: str | Path, model_name: str):
        """Save the trained model to disk.

        Parameters
        ----------
        path : str or Path
            Directory path where the model will be saved.
        model_name : str
            Name of the model file (without extension).
        """
        if self.rs_model is None:
            raise ValueError("Model must be trained before saving")

        path = Path(path) / model_name
        path.parent.mkdir(parents=True, exist_ok=True)

        model_state = {
            k: v
            for k, v in self.__dict__.items()
            if k not in ("rs_model", "_cfg", "np_rng")
        }
        joblib.dump(model_state, f"{path}.joblib")

        self._cfg.save_fn(self.rs_model, str(path.parent), model_name)

    @classmethod
    def load(cls, path: str | Path, model_name: str) -> Self:
        """Load a trained model from disk.

        Parameters
        ----------
        path : str or Path
            Directory path where the model is saved.
        model_name : str
            Name of the model file (without extension).

        Returns
        -------
        Self
            Loaded model instance.
        """
        path = Path(path) / model_name
        model_state = joblib.load(f"{path}.joblib")

        model = cls.__new__(cls)
        model.__dict__.update(model_state)

        if model.model_type not in _MODEL_CONFIGS:
            raise ValueError(f"Unknown model_type: {model.model_type}")
        model._cfg = _MODEL_CONFIGS[model.model_type]

        model.np_rng = np.random.default_rng(model.seed)
        model.rs_model = model._cfg.load_fn(str(path.parent), model_name)
        return model

    def update_data_info(self, data_info: DataInfo):
        """Update the model's data information for incremental training.

        Parameters
        ----------
        data_info : DataInfo
            New data information containing updated user/item mappings and consumed data.
        """
        cur_n_users = data_info.n_users
        cur_n_items = data_info.n_items
        old_n_users = self.n_users
        old_n_items = self.n_items
        if cur_n_users > old_n_users or cur_n_items > old_n_items:
            normal_log(
                f"Expanding vocabulary: ({old_n_users}, {old_n_items}) -> ({cur_n_users}, {cur_n_items})"
            )

        self.data_info = data_info
        self.n_users = cur_n_users
        self.n_items = cur_n_items
        self.user_consumed = data_info.user_consumed
        self.id_converter = data_info.id_converter
        # update Rust model parameters
        self.rs_model.n_users = cur_n_users
        self.rs_model.n_items = cur_n_items
        self.rs_model.user_consumed = data_info.user_consumed
