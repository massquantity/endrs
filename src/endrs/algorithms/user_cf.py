from endrs.bases.cf_base import CfBase
from endrs.data.data_info import DataInfo
from endrs_ext import UserCF as RsUserCF


class UserCF(CfBase):
    """User-based Collaborative Filtering model.

    Parameters
    ----------
    task : {'rating', 'ranking'}
        The recommendation task type.
    data_info : :class:`~endrs.data.DataInfo`
        Object that contains useful information about the dataset.
    k_sim : int, default: 20
        Number of similar users to consider for recommendation.
    min_common : int, default: 1
        Minimum number of common items required for similarity computation.
    num_threads : int, default: 1
        Number of threads to use for similarity computation.
    seed : int, default: 42
        Random seed for reproducibility.
    """

    model_type = "user_cf"

    def __init__(
        self,
        task: str,
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
        user_interacts: list[list[tuple[int, float]]],
        item_interacts: list[list[tuple[int, float]]],
    ) -> RsUserCF:
        return RsUserCF(
            self.task,
            self.k_sim,
            self.n_users,
            self.n_items,
            self.min_common,
            user_interacts,
            item_interacts,
            self.user_consumed,
        )
