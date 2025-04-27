from collections.abc import Sequence

import numpy as np
from scipy.special import expit

from endrs.data.data_info import IdConverter
from endrs.utils.constants import DEFAULT_PRED
from endrs.types import ItemId, UserId


def normalize_prediction(
    preds: np.ndarray,
    task: str,
    cold_start: str,
    unknown_num: int,
    unknown_index: list[int],
) -> list[float]:
    if task == "ranking":
        preds = expit(preds)
    if unknown_num > 0 and cold_start == "popular":
        preds[unknown_index] = DEFAULT_PRED
    return preds.tolist()


def construct_rec(
    id_converter: IdConverter,
    user_ids: Sequence[int],
    computed_recs: list[list[int]],
    inner_id: bool,
) -> dict[UserId, list[ItemId]]:
    """Convert internal recommendation IDs to the original format.

    Parameters
    ----------
    id_converter : :class:`~endrs.data.data_info.IdConverter`
        Converter between internal and original IDs.
    user_ids : Sequence[int]
        User IDs for which recommendations are made.
    computed_recs : list[list[int]]
        Computed recommendations as lists of item IDs.
    inner_id : bool
        Whether to return recommendations in internal IDs.

    Returns
    -------
    dict[UserId, list[ItemId]]
        Dictionary mapping user IDs to lists of recommended item IDs.
    """
    result_recs = dict()
    for i, u in enumerate(user_ids):
        if inner_id:
            result_recs[u] = computed_recs[i]
        else:
            u = id_converter.id2user[u]
            result_recs[u] = [id_converter.id2item[ri] for ri in computed_recs[i]]
    return result_recs
