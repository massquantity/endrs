import dataclasses

import numpy as np
import torch

from endrs.data.batch import Batch, label_sampling, multi_label_sampling
from endrs.data.data_info import DataInfo
from endrs.data.sequence import SeqParams, get_interacted_seqs
from endrs.feature.feat_info import FeatInfo
from endrs.inference.preprocess import get_item_inputs, get_seq_inputs, get_user_inputs
from endrs.sampling.negatives import (
    neg_probs_from_frequency,
    negatives_from_popular,
    negatives_from_random,
    negatives_from_unconsumed,
)
from endrs.utils.constants import LABEL_KEY


class BaseCollator:
    """Base class for all data collators."""

    def __init__(
        self,
        data_info: DataInfo,
        feat_info: FeatInfo | None,
        neg_sampling: bool,
        sampler: str = "random",
        num_neg: int = 1,
        seq_params: SeqParams | None = None,
        is_multi_task: bool = False,
        seed: int = 42,
        temperature: float = 0.75,
    ):
        self.use_hash = data_info.use_hash
        self.n_users = data_info.n_users
        self.n_items = data_info.n_items
        self.user_consumed = data_info.user_consumed
        self.item_consumed = data_info.item_consumed
        self.cand_users = data_info.candidate_users
        self.cand_items = data_info.candidate_items
        self.feat_info = feat_info
        self.neg_sampling = neg_sampling
        self.sampler = sampler
        self.num_neg = num_neg
        self.seq_params = seq_params
        self.is_multi_task = is_multi_task
        self.seed = seed
        self.temperature = temperature
        self.user_consumed_set = None
        self.neg_probs = None
        self.np_rng = None

    def __call__(self, batch: Batch) -> dict[str, torch.Tensor]:
        users, items, labels, multi_labels = dataclasses.astuple(batch)
        lb = multi_labels if self.is_multi_task else labels
        if self.neg_sampling:
            # use original users and items to sample
            items_neg = self.sample_neg_items(users, items)
            users = np.repeat(users, self.num_neg + 1)
            items = np.repeat(items, self.num_neg + 1)
            for i in range(self.num_neg):
                items[(i + 1) :: (self.num_neg + 1)] = items_neg[i :: self.num_neg]
            if self.is_multi_task:
                lb = multi_label_sampling(multi_labels, self.num_neg)
            else:
                lb = label_sampling(labels, self.num_neg)

        user_batch = get_user_inputs(users, self.feat_info)
        item_batch = get_item_inputs(items, self.feat_info)
        batch_data = {**user_batch, **item_batch, LABEL_KEY: torch.from_numpy(lb)}
        if self.seq_params:
            batch_data = self.get_seq_features(batch_data, users, items)
        return batch_data

    def get_seq_features(
        self, batch: dict[str, torch.Tensor], users: np.ndarray, items: np.ndarray
    ) -> dict[str, torch.Tensor]:
        self._set_user_consumed()
        # TODO: if isinstance(self.seq_params, DualSeqParams):
        seqs = get_interacted_seqs(
            users,
            items,
            self.user_consumed,
            self.seq_params.max_seq_len,
            self.user_consumed_set,
        )
        return get_seq_inputs(batch, seqs, self.feat_info)

    def sample_neg_items(self, users: np.ndarray, items: np.ndarray):
        if self.sampler == "unconsumed":
            self._set_user_consumed()
            items_neg = negatives_from_unconsumed(
                self.user_consumed_set,
                users,
                items,
                self.n_items,
                self.num_neg,
                self.cand_items,
            )
        elif self.sampler == "popular":
            self._set_random_seeds()
            self._set_neg_probs()
            items_neg = negatives_from_popular(
                self.np_rng,
                items,
                self.num_neg,
                self.cand_items,
                probs=self.neg_probs,
            )
        elif self.sampler == "random":
            self._set_random_seeds()
            items_neg = negatives_from_random(
                self.np_rng,
                items,
                self.num_neg,
                self.cand_items,
            )
        else:
            raise ValueError(
                f"sampler must be one of `random`, `popular`, `unconsumed`, got `{self.sampler}`"
            )
        return items_neg

    def _set_user_consumed(self):
        if self.user_consumed_set is None:
            self.user_consumed_set = {
                u: set(self.user_consumed[u]) for u in self.cand_users
            }

    def _set_neg_probs(self):
        if self.neg_probs is None:
            self.neg_probs = neg_probs_from_frequency(
                self.cand_items, self.item_consumed, self.temperature
            )

    def _set_random_seeds(self):
        if self.np_rng is None:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is None:
                worker_id = 0
            else:
                worker_id = worker_info.id

            seed = self.seed + worker_id
            # TODO: set in `seed_everything()`
            # random.seed(seed)
            # torch.manual_seed(seed)
            self.np_rng = np.random.default_rng(seed)


class PairwiseCollator(BaseCollator):
    def __init__(
        self,
        data_info: DataInfo,
        feat_info: FeatInfo | None,
        sampler: str = "random",
        num_neg: int = 1,
        seq_params: SeqParams | None = None,
        is_multi_task: bool = False,
        seed: int = 42,
    ):
        super().__init__(
            data_info=data_info,
            feat_info=feat_info,
            neg_sampling=True,
            sampler=sampler,
            num_neg=num_neg,
            seq_params=seq_params,
            is_multi_task=is_multi_task,
            seed=seed,
        )

    def __call__(
        self, batch: Batch
    ) -> tuple[
        dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, torch.Tensor]
    ]:
        users, items = batch.users, batch.items
        items_neg = self.sample_neg_items(users, items)
        user_batch = get_user_inputs(users, self.feat_info)
        item_batch = get_item_inputs(items, self.feat_info)
        item_neg_batch = get_item_inputs(items_neg, self.feat_info)
        if self.seq_params:
            user_batch = self.get_seq_features(user_batch, users, items)
        return user_batch, item_batch, item_neg_batch
