import math
import numbers
from collections.abc import Sequence

import numpy as np
import pandas as pd
from lightning.pytorch.utilities import CombinedLoader
from sklearn.metrics import log_loss, mean_absolute_error, r2_score, roc_auc_score
from torch.utils.data import BatchSampler, DataLoader, SequentialSampler
from tqdm import tqdm

from endrs.data.batch import EvalBatchData
from endrs.data.dataset import Dataset
from endrs.evaluation.metrics import (
    map_at_k,
    listwise_scores,
    ndcg_at_k,
    pr_auc_score,
    precision_at_k,
    rec_coverage,
    recall_at_k,
    rmse,
    roc_gauc_score,
)
from endrs.types import RecModel
from endrs.utils.constants import (
    LISTWISE_METRICS,
    POINTWISE_METRICS,
    RANKING_METRICS,
    RATING_METRICS,
)
from endrs.utils.logger import get_logger
from endrs.utils.validate import check_data_cols


class Evaluator:
    """Evaluation class for recommendation models.

    Parameters
    ----------
    during_training : bool
        Whether the evaluation is occurring during model training.
    data : EvalBatchData or pd.DataFrame
        The evaluation dataset, either as a batch data object or a pandas DataFrame.
    model : :type:`~endrs.types.RecModel`
        The recommendation model to evaluate.
    is_multi_task : bool
        Whether it is a multi-task model.
    neg_sampling : bool
        Whether negative sampling is used for training/evaluation.
    metrics : str or Sequence[str] or None, default: None
        The evaluation metrics to compute. If None, defaults to ['loss'].
    k : int, default: 10
        The number of items to recommend for ranking metrics (e.g., Precision@k).
    eval_batch_size : int, default: 8192
        The batch size for evaluation.
    sample_user_num : int or None, default: None
        Number of users to sample for listwise metrics. If None, all users are used.
    seed : int, default: 42
        Random seed for reproducibility.
    verbose : int, default: 2
        Verbosity level (0: silent, 1: minimal, 2: progress bars, 3: full output).
    num_workers : int, default: 0
        Number of workers for data loading.

    Attributes
    ----------
    data_info : :class:`~endrs.data.DataInfo`
        Information about the dataset used by the model.
    task : str
        The recommendation task type ('rating' or 'ranking').
    n_items : int
        Number of items in the dataset.
    preds : list
        Stored model predictions.
    recos : dict
        Stored model recommendations.
    pred_labels : np.ndarray
        Ground truth labels for predictions.
    reco_labels : dict
        Ground truth consumed items for recommendations.
    """

    def __init__(
        self,
        during_training: bool,
        data: EvalBatchData | pd.DataFrame,
        model: RecModel,
        is_multi_task: bool,
        neg_sampling: bool,
        metrics: str | Sequence[str] | None = None,
        k: int = 10,
        eval_batch_size: int = 8192,
        sample_user_num: int | None = None,
        seed: int = 42,
        verbose: int = 2,
        num_workers: int = 0,
    ):
        self.model = model
        self.data_info = model.data_info
        self.task = model.task
        self.is_multi_task = is_multi_task
        self.n_items = model.n_items
        self.neg_sampling = neg_sampling
        self.metrics = self._check_metrics(metrics)
        self.k = k
        self.eval_batch_size = eval_batch_size
        self.sample_user_num = sample_user_num
        self.seed = seed
        self.verbose = verbose
        self.num_workers = num_workers
        self.during_training = during_training
        self.data = self.build_eval_data(data)
        self.preds = list()
        self.recos = dict()
        self.pred_labels = self.data.get_labels(is_multi_task)
        self.reco_labels = self.data.get_positive_consumed(is_multi_task)

    def build_eval_data(self, data: EvalBatchData | pd.DataFrame) -> EvalBatchData:
        """Convert input data to evaluation batch data format.
        
        Parameters
        ----------
        data : EvalBatchData or pd.DataFrame
            The evaluation data, either already in batch format or as a DataFrame.
            
        Returns
        -------
        EvalBatchData
            Processed evaluation data ready for metrics computation.
            
        Raises
        ------
        ValueError
            If data is neither a pandas DataFrame nor an EvalBatchData object.
        """
        if not isinstance(data, (pd.DataFrame, EvalBatchData)):
            raise ValueError("`data` must be `pandas.DataFrame` or `EvalBatchData`")
        if isinstance(data, pd.DataFrame):
            user_col = self.data_info.user_col_name
            item_col = self.data_info.item_col_name
            label_col = self.data_info.label_col_name
            multi_label_col = self.data_info.multi_label_col_name
            check_data_cols([data], [user_col, item_col, label_col, multi_label_col])
            data = Dataset.build_batch(
                data,
                user_col,
                item_col,
                label_col,
                multi_label_col,
                self.data_info.id_converter,
                is_train=False,
            )
            data.check_labels(self.is_multi_task, self.task, self.neg_sampling)

        if self.neg_sampling and not data.has_sampled:
            data.build_negatives(
                self.n_items,
                self.model.num_neg,
                self.model.cand_items,
                self.seed,
            )
        return data

    def build_data_loader(self) -> CombinedLoader:
        """Create data loaders for model evaluation.

        Creates loaders for both pointwise prediction (if needed by metrics) and
        listwise recommendation evaluation (if needed by metrics). Uses PyTorch Lightning's
        CombinedLoader to handle multiple loader types efficiently.

        Returns
        -------
        CombinedLoader
            A PyTorch Lightning combined loader that sequentially yields batches
            from both prediction and recommendation loaders.
        """
        pred_loader, reco_loader = [], []
        if self.need_preds:
            sampler = SequentialSampler(self.data)
            batch_sampler = BatchSampler(sampler, self.eval_batch_size, drop_last=False)
            pred_loader = DataLoader(
                self.data,
                batch_size=None,
                sampler=batch_sampler,
                collate_fn=lambda batch: (batch.users, batch.items),
                num_workers=self.num_workers,
            )
        if self.need_recos:
            users = self.sample_users()
            num_batch_users = max(1, math.floor(self.eval_batch_size / self.n_items))
            reco_loader = DataLoader(
                users,
                batch_size=num_batch_users,
                drop_last=False,
                collate_fn=lambda x: x,
                num_workers=self.num_workers,
            )
        return CombinedLoader([pred_loader, reco_loader], mode="sequential")

    def update_preds(self, batch: tuple[np.ndarray, np.ndarray]):
        users, items = batch
        batch_preds = self.model.predict(users, items, inner_id=True)
        self.preds.extend(batch_preds)

    def update_recos(self, batch_users: np.ndarray):
        batch_recos = self.model.recommend_user(
            user=batch_users,
            n_rec=self.k,
            inner_id=True,
            filter_consumed=True,
            random_rec=False,
        )
        self.recos.update(batch_recos)

    def compute_eval_results(self) -> dict[str, float]:
        """Compute evaluation metrics on the current data.
        
        If not during training, first generates predictions and recommendations
        for all evaluation data. Then computes all specified metrics based on the
        task type (rating or ranking).
        
        Returns
        -------
        dict[str, float]
            Dictionary mapping metric names to their computed values.
        """
        # during training, states will be updated in `validation_step()`
        if not self.during_training:
            combined_loader = iter(self.build_data_loader())
            disable = True if self.verbose < 2 else False
            leave = True if self.verbose >= 3 else False
            for batch, _, dataloader_idx in tqdm(
                combined_loader,
                desc="Validation DataLoader",
                disable=disable,
                leave=leave,
            ):
                if dataloader_idx == 0:
                    self.update_preds(batch)
                else:
                    self.update_recos(batch)

        eval_result = dict()
        if self.task == "rating":
            for m in self.metrics:
                if m in ("rmse", "loss"):
                    eval_result[m] = rmse(self.pred_labels, self.preds)
                elif m == "mae":
                    eval_result[m] = mean_absolute_error(self.pred_labels, self.preds)
                elif m == "r2":
                    eval_result[m] = r2_score(self.pred_labels, self.preds)
        else:
            if self.need_preds:
                for m in self.metrics:
                    if m in ("log_loss", "loss"):
                        eval_result[m] = log_loss(self.pred_labels, self.preds)
                    elif m == "roc_auc":
                        eval_result[m] = roc_auc_score(self.pred_labels, self.preds)
                    elif m == "roc_gauc":
                        eval_result[m] = roc_gauc_score(
                            self.pred_labels, self.preds, self.data.users
                        )
                    elif m == "pr_auc":
                        eval_result[m] = pr_auc_score(self.pred_labels, self.preds)

            if self.need_recos:
                for m in self.metrics:
                    if m not in LISTWISE_METRICS:
                        continue

                    users = self.sample_users()
                    if m == "coverage":
                        eval_result[m] = rec_coverage(self.recos, users, self.n_items)
                    elif m == "precision":
                        eval_result[m] = listwise_scores(
                            precision_at_k, self.reco_labels, self.recos, users, self.k
                        )
                    elif m == "recall":
                        eval_result[m] = listwise_scores(
                            recall_at_k, self.reco_labels, self.recos, users, self.k
                        )
                    elif m == "map":
                        eval_result[m] = listwise_scores(
                            map_at_k, self.reco_labels, self.recos, users, self.k
                        )
                    elif m == "ndcg":
                        eval_result[m] = listwise_scores(
                            ndcg_at_k, self.reco_labels, self.recos, users, self.k
                        )

        return eval_result

    def log_metrics(self, epoch: int):
        message = ""
        eval_metrics = self.compute_eval_results()
        for m, val in eval_metrics.items():
            if m in LISTWISE_METRICS:
                metric = f"{m}@{self.k}"
            else:
                metric = m
            str_val = f"{round(val, 2)}%" if m == "coverage" else f"{val:.4f}"
            message += f"    eval {metric}: {str_val}\n"

        logger = get_logger()
        if logger is not None:
            logger.bind(task="metrics").info(message)
            message = f"Epoch {epoch + 1}:\n" + message
            logger.bind(task="metrics_file").info(message)
        else:
            tqdm.write(message)

    # def print_metrics(self):
    #     eval_metrics = self.compute_eval_results()
    #     for m, val in eval_metrics.items():
    #         if m in LISTWISE_METRICS:
    #             metric = f"{m}@{self.k}"
    #         else:
    #             metric = m
    #         str_val = f"{round(val, 2)}%" if m == "coverage" else f"{val:.4f}"
    #         tqdm.write(f"\t eval {metric}: {str_val}")

    def clear_state(self):
        """Clear all stored prediction and recommendation data."""
        self.preds.clear()
        self.recos.clear()

    @property
    def need_preds(self) -> bool:
        if self.task == "rating":
            return True
        for m in self.metrics:
            if m in POINTWISE_METRICS:
                return True
        return False

    @property
    def need_recos(self) -> bool:
        return bool(set(LISTWISE_METRICS).intersection(self.metrics))

    def _check_metrics(self, metrics: str | Sequence[str] | None) -> list[str]:
        if not metrics:
            metrics = ["loss"]
        if not isinstance(metrics, list | tuple):
            metrics = [metrics]

        if self.task == "rating":
            for m in metrics:
                if m not in RATING_METRICS:
                    raise ValueError(f"Metrics `{m}` is not suitable for rating task...")
        elif self.task == "ranking":
            for m in metrics:
                if m not in RANKING_METRICS:
                    raise ValueError(f"Metrics `{m}` is not suitable for ranking task...")

        return metrics

    def sample_users(self) -> list[int]:
        unique_users = list(self.data.get_positive_consumed(self.is_multi_task))
        if isinstance(self.sample_user_num, numbers.Integral) and (
            0 < self.sample_user_num < len(unique_users)
        ):
            np_rng = np.random.default_rng(self.seed)
            users = np_rng.choice(unique_users, self.sample_user_num, replace=False)
            users = users.tolist()
        else:
            users = unique_users
        return users
