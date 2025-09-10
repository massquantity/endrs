from collections.abc import Sequence
from pathlib import Path
from typing import Self

import joblib
import numpy as np
import pandas as pd

from endrs.data.batch import BatchData, EvalBatchData
from endrs.data.data_info import DataInfo
from endrs.evaluation.evaluator import Evaluator
from endrs.inference.cold_start import popular_recommendations
from endrs.inference.postprocess import construct_rec
from endrs.inference.preprocess import convert_ids, get_unknown, sep_unknown_users
from endrs.types import ItemId, UserId
from endrs.utils.logger import normal_log
from endrs.utils.misc import show_start_time, time_block
from endrs.utils.sparse import construct_sparse
from endrs.utils.validate import check_labels
from endrs_ext import Swing as RsSwing, save_swing, load_swing


class Swing:
    def __init__(
        self,
        task: str,
        data_info: DataInfo,
        top_k: int = 20,
        alpha: float = 0.1,
        max_cache_num: int = 100_000_000,
        num_threads: int = 1,
        seed: int = 42,
    ):
        # super().__init__()

        assert task == "ranking", "Swing only supports ranking task."
        self.task = task
        self.data_info = data_info
        self.n_users = data_info.n_users
        self.n_items = data_info.n_items
        self.user_consumed = data_info.user_consumed
        self.id_converter = data_info.id_converter
        self.top_k = top_k
        self.alpha = alpha
        self.max_cache_num = max_cache_num
        self.num_threads = num_threads
        self.seed = seed
        self.np_rng = np.random.default_rng(seed)
        self.rs_model = None

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
        """Train the Swing model on the provided training data.

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
            self.rs_model = RsSwing(
                self.top_k,
                self.alpha,
                self.max_cache_num,
                self.n_users,
                self.n_items,
                user_interacts,
                item_interacts,
                self.user_consumed,
            )
            with time_block("swing computing", verbose=1):
                self.rs_model.compute_swing(self.num_threads)
        else:
            with time_block("update swing", verbose=1):
                self.rs_model.update_swing(
                    self.num_threads, user_interacts, item_interacts
                )

        num = self.rs_model.num_swing_elements()
        density_ratio = 100 * num / (self.n_items * self.n_items)
        normal_log(f"swing num_elements: {num}, density: {density_ratio:5.4f} %")

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
        """Evaluate the trained Swing model on evaluation data.

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
            raise RuntimeError("Model must be trained before evaluation. Call fit() first.")

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
        user, item = convert_ids(user, item, self.id_converter, inner_id)
        unknown_num, _ = get_unknown(user, item)
        if unknown_num > 0 and cold_start != "popular":
            raise ValueError("Swing only supports popular strategy")

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
        result_recs = dict()
        user_ids, unknown_users = sep_unknown_users(self.id_converter, user, inner_id)
        if unknown_users:
            if cold_start != "popular":
                raise ValueError("Swing only supports `popular` cold start strategy")
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
            for rec, arc in zip(computed_recs, additional_rec_counts):
                if arc > 0:
                    additional_recs = popular_recommendations(
                        self.data_info, inner_id=True, n_rec=arc, np_rng=self.np_rng
                    )
                    rec.extend(additional_recs)

            user_recs = construct_rec(
                self.id_converter, user_ids, computed_recs, inner_id
            )
            result_recs.update(user_recs)

        return result_recs

    def save(self, path: str | Path, model_name: str):
        if self.rs_model is None:
            raise ValueError("Model must be trained before saving")
        
        path = Path(path) / model_name
        path.parent.mkdir(parents=True, exist_ok=True)

        model_state = {k: v for k, v in self.__dict__.items() if k != "rs_model"}
        joblib.dump(model_state, f"{path}.joblib")

        save_swing(self.rs_model, str(path.parent), model_name)

    @classmethod
    def load(cls, path: str | Path, model_name: str) -> Self:
        path = Path(path) / model_name
        model_state = joblib.load(f"{path}.joblib")

        model = cls.__new__(cls)
        model.__dict__.update(model_state)

        model.np_rng = np.random.default_rng(model.seed)
        model.rs_model = load_swing(str(path.parent), model_name)
        return model

    def update_data_info(self, data_info: DataInfo):
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
