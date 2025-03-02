import math
import random
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset as TorchDataset

from endrs.data.consumed import interaction_consumed
from endrs.sampling.negatives import negatives_from_unconsumed
from endrs.utils.validate import check_labels, check_multi_labels


@dataclass
class Batch:
    users: np.ndarray
    items: np.ndarray
    labels: np.ndarray | None
    multi_labels: np.ndarray | None


class BatchData(TorchDataset):
    def __init__(
        self,
        user_indices: Sequence[int],
        item_indices: Sequence[int],
        labels: Sequence[float] | None,
        multi_labels: Sequence[Sequence[float]] | None,
        factor: int = 1,
    ):
        self.users = np.array(user_indices)
        self.items = np.array(item_indices)
        self.labels = np.array(labels) if labels else None
        self.multi_labels = np.array(multi_labels) if multi_labels else None
        self.factor = factor

    def __getitem__(self, idx):
        return Batch(
            self.users[idx],
            self.items[idx],
            self.labels[idx] if self.labels is not None else None,
            self.multi_labels[idx] if self.multi_labels is not None else None,
        )

    def __len__(self):
        assert self.factor > 0
        return math.ceil(len(self.users) / self.factor)

    def ui(self) -> tuple[np.ndarray, np.ndarray]:
        return self.users, self.items

    def check_labels(self, is_multi_task: str, task: str, neg_sampling: bool):
        if is_multi_task:
            check_multi_labels(task, self.multi_labels, neg_sampling)
        else:
            check_labels(task, self.labels, neg_sampling)


class EvalBatchData(BatchData):
    def __init__(
        self,
        user_indices: Sequence[int],
        item_indices: Sequence[int],
        labels: Sequence[float] | None,
        multi_labels: Sequence[Sequence[float]] | None,
    ):
        super().__init__(user_indices, item_indices, labels, multi_labels)
        self.has_sampled = False
        self.positive_consumed = None

    def get_labels(self, is_multi_task: bool) -> list[float]:
        if is_multi_task:
            assert self.multi_labels is not None
            labels = get_single_labels(self.multi_labels)
        else:
            assert self.labels is not None
            labels = self.labels
        return labels.tolist()

    def get_positive_consumed(self, is_multi_task: bool) -> dict[int, list[int]]:
        if self.positive_consumed is not None:
            return self.positive_consumed
        labels = np.array(self.get_labels(is_multi_task))
        # data without label column has dummy labels 0
        label_all_positive = np.all(labels == 0.0)
        user_consumed = defaultdict(list)
        for u, i, lb in zip(self.users, self.items, labels):
            if label_all_positive or lb != 0.0:
                user_consumed[u].append(i)
        self.positive_consumed = {
            u: list(set(items)) for u, items in user_consumed.items()
        }
        return self.positive_consumed

    def build_negatives(
        self,
        n_items: int,
        num_neg: int,
        candidate_items: np.ndarray,
        seed: int,
    ):
        random.seed(seed)
        self.has_sampled = True
        # use original users and items to sample
        items_neg = self._sample_neg_items(
            self.users, self.items, n_items, num_neg, candidate_items
        )
        self.users = np.repeat(self.users, num_neg + 1)
        self.items = np.repeat(self.items, num_neg + 1)
        for i in range(num_neg):
            self.items[(i + 1) :: (num_neg + 1)] = items_neg[i::num_neg]
        self.labels = label_sampling(self.labels, num_neg)
        self.multi_labels = multi_label_sampling(self.multi_labels, num_neg)

    @staticmethod
    def _sample_neg_items(
        users: np.ndarray,
        items: np.ndarray,
        n_items: int,
        num_neg: int,
        candidate_items: np.ndarray,
    ) -> np.ndarray:
        user_consumed, _ = interaction_consumed(users, items)
        user_consumed_set = {u: set(uis) for u, uis in user_consumed.items()}
        return negatives_from_unconsumed(
            user_consumed_set, users, items, n_items, num_neg, candidate_items
        )


def get_single_labels(
    multi_labels: np.ndarray | torch.Tensor
) -> np.ndarray | torch.Tensor:
    """label is 1 if one of the tasks is 1"""
    if isinstance(multi_labels, np.ndarray):
        labels = np.zeros(len(multi_labels), dtype=np.float32)
        labels[np.any(multi_labels != 0.0, axis=1)] = 1.0
        return labels
    else:
        return torch.any(multi_labels != 0.0, dim=1).float()


def label_sampling(labels: np.ndarray | None, num_neg: int):
    if labels is None:
        return
    size = len(labels) * (num_neg + 1)
    labels = np.zeros(size, dtype=np.float32)
    labels[:: (num_neg + 1)] = 1.0
    return labels


def multi_label_sampling(multi_labels: np.ndarray | None, num_neg: int):
    if multi_labels is None:
        return
    size = len(multi_labels) * (num_neg + 1)
    labels = np.zeros((size, multi_labels.shape[1]), dtype=np.float32)
    # use original labels
    labels[:: (num_neg + 1)] = multi_labels
    return labels
