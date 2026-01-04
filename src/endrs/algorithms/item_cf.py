from typing import ClassVar, Literal

from endrs.bases.cf_base import CfBase
from endrs.data.data_info import DataInfo
from endrs.utils.sparse import SparseMatrix
from endrs_ext import ItemCF as RsItemCF


class ItemCF(CfBase):
    """Item-based Collaborative Filtering model.

    Parameters
    ----------
    task : {'rating', 'ranking'}
        The recommendation task type.
    data_info : :class:`~endrs.data.DataInfo`
        Object that contains useful information about the dataset.
    k_sim : int, default: 20
        Number of similar items to consider for recommendation.
    min_common : int, default: 1
        Minimum number of common users required for similarity computation.
    num_threads : int, default: 1
        Number of threads to use for similarity computation.
    seed : int, default: 42
        Random seed for reproducibility.
    """

    model_type: ClassVar[str] = "item_cf"

    def __init__(
        self,
        task: Literal["rating", "ranking"],
        data_info: DataInfo,
        k_sim: int = 20,
        min_common: int = 1,
        num_threads: int = 1,
        seed: int = 42,
    ):
        super().__init__(task, data_info, num_threads, seed)
        self.k_sim = k_sim
        self.min_common = min_common

    def _create_rust_model(
        self,
        user_interacts: SparseMatrix,
        item_interacts: SparseMatrix,
    ) -> RsItemCF:
        return RsItemCF(
            self.task,
            self.k_sim,
            self.n_users,
            self.n_items,
            self.min_common,
            user_interacts,
            item_interacts,
            self.user_consumed,
        )
