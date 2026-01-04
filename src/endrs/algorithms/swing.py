from typing import ClassVar, Literal

from endrs.bases.cf_base import CfBase
from endrs.data.data_info import DataInfo
from endrs.utils.sparse import SparseMatrix
from endrs_ext import Swing as RsSwing


class Swing(CfBase):
    """Swing model for item similarity computation.

    Swing is an item-based collaborative filtering algorithm that computes
    item similarities based on user co-occurrence patterns with a penalty
    for popular user pairs.

    Parameters
    ----------
    task : {'ranking'}
        The recommendation task type. Swing only supports ranking task.
    data_info : :class:`~endrs.data.DataInfo`
        Object that contains useful information about the dataset.
    top_k : int, default: 20
        Number of top similar items to keep for each item.
    alpha : float, default: 0.1
        Smoothing parameter for the swing weight computation.
    max_cache_num : int, default: 100_000_000
        Maximum number of user pairs to cache during computation.
    num_threads : int, default: 1
        Number of threads to use for computation.
    seed : int, default: 42
        Random seed for reproducibility.
    """

    model_type: ClassVar[str] = "swing"

    def __init__(
        self,
        task: Literal["ranking"],
        data_info: DataInfo,
        top_k: int = 20,
        alpha: float = 0.1,
        max_cache_num: int = 100_000_000,
        num_threads: int = 1,
        seed: int = 42,
    ):
        if task != "ranking":
            raise ValueError(f"Swing only supports ranking task, got {task!r}")

        super().__init__(task, data_info, num_threads, seed)
        self.top_k = top_k
        self.alpha = alpha
        self.max_cache_num = max_cache_num

    def _create_rust_model(
        self,
        user_interacts: SparseMatrix,
        item_interacts: SparseMatrix,
    ) -> RsSwing:
        return RsSwing(
            self.top_k,
            self.alpha,
            self.max_cache_num,
            self.n_users,
            self.n_items,
            user_interacts,
            item_interacts,
            self.user_consumed,
        )
