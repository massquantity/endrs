from collections.abc import Callable, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    auc,
    mean_squared_error,
    precision_recall_curve,
    roc_auc_score,
)


def rmse(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    return np.sqrt(mean_squared_error(y_true, y_pred))


def roc_gauc_score(
    y_true: Sequence[float], y_prob: Sequence[float], user_indices: Sequence[int]
) -> float:
    # gauc = 0
    # users = np.unique(user_indices)
    # y_true, y_prob = np.array(y_true), np.array(y_prob)
    # for u in users:
    #    index = np.where(user_indices == u)[0]
    #    user_auc = roc_auc_score(y_true[index], y_prob[index])
    #    gauc += len(index) * user_auc
    # return gauc / len(user_indices)

    def _safe_roc_auc(y_true, y_score):
        try:
            auc = roc_auc_score(y_true, y_score)
        except ValueError:  # only has one label
            auc = 0.0
        return auc

    roc_data = pd.DataFrame({"label": y_true, "prob": y_prob, "user": user_indices})
    gauc = (
        roc_data.groupby("user")
        .apply(lambda x: _safe_roc_auc(x["label"], x["prob"]) * len(x))
        .tolist()
    )
    return sum(gauc) / len(user_indices)


def pr_auc_score(y_true: Sequence[float], y_prob: Sequence[float]) -> float:
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    return auc(recall, precision)


def listwise_scores(
    fn: Callable,
    y_true_lists: Mapping[int, Sequence[int]],
    y_reco_lists: Mapping[int, Sequence[int]],
    users: Sequence[int],
    k: int,
) -> float:
    scores = list()
    for u in users:
        y_true = y_true_lists[u]
        y_reco = y_reco_lists[u]
        scores.append(fn(y_true, y_reco, k))
    return np.mean(scores)


def precision_at_k(y_true: Sequence[int], y_reco: Sequence[int], k: int) -> float:
    common_items = set(y_reco).intersection(y_true)
    return len(common_items) / k


def recall_at_k(y_true: Sequence[int], y_reco: Sequence[int], _k: int) -> float:
    common_items = set(y_reco).intersection(y_true)
    return len(common_items) / len(y_true)


def map_at_k(y_true: Sequence[int], y_reco: Sequence[int], k: int) -> float:
    common_items, _, indices_in_reco = np.intersect1d(
        y_true, y_reco, assume_unique=True, return_indices=True
    )
    if len(common_items) == 0:
        return 0
    rank_list = np.zeros(k, np.float32)
    rank_list[indices_in_reco] = 1
    ap = [np.mean(rank_list[: i + 1]) for i in range(k) if rank_list[i]]
    assert len(ap) == len(common_items), "common size doesn't match..."
    return np.mean(ap)


def ndcg_at_k(y_true: Sequence[int], y_reco: Sequence[int], k: int) -> float:
    common_items, _, indices_in_reco = np.intersect1d(
        y_true, y_reco, assume_unique=True, return_indices=True
    )
    if len(common_items) == 0:
        return 0
    rank_list = np.zeros(k, np.float32)
    rank_list[indices_in_reco] = 1
    ideal_list = np.sort(rank_list)[::-1]
    dcg = np.sum(rank_list / np.log2(np.arange(2, k + 2)))
    idcg = np.sum(ideal_list / np.log2(np.arange(2, k + 2)))
    return dcg / idcg


def rec_coverage(
    y_reco_lists: Mapping[int, Sequence[int]], users: Sequence[int], n_items: int
) -> float:
    item_recs = set()
    for u in users:
        y_reco = y_reco_lists[u]
        item_recs.update(y_reco)
    return len(item_recs) / n_items * 100
