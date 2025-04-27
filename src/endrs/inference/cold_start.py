from collections.abc import Sequence

import numpy as np

from endrs.data.data_info import DataInfo
from endrs.types import ItemId, UserId


def popular_recommendations(
    data_info: DataInfo, inner_id: bool, n_rec: int, np_rng: np.random.Generator
) -> list[ItemId]:
    """Generate recommendations based on item popularity.

    Parameters
    ----------
    data_info : :class:`~endrs.data.DataInfo`
        Object that contains useful information about the dataset.
    inner_id : bool
        Whether to return items in inner IDs.
    n_rec : int
        Number of items to recommend.
    np_rng : np.random.Generator
        Random number generator for sampling.

    Returns
    -------
    list[ItemId]
        List of recommended item IDs.
    """
    popular_recs = np_rng.choice(data_info.pop_items, n_rec)
    if inner_id:
        return [data_info.id_converter.item2id[i] for i in popular_recs]
    else:
        return popular_recs.tolist()


def average_recommendations(
    data_info: DataInfo,
    default_recs: Sequence[int],
    inner_id: bool,
    n_rec: int,
    np_rng: np.random.Generator,
) -> list[ItemId]:
    """Generate recommendations based on default/average items.

    Average items are calculated bsed on mean embeddings.

    Parameters
    ----------
    data_info : :class:`~endrs.data.DataInfo`
        Object that contains useful information about the dataset.
    default_recs : Sequence[int]
        Default recommendations to choose from.
    inner_id : bool
        Whether to return items in inner IDs.
    n_rec : int
        Number of items to recommend.
    np_rng : np.random.Generator
        Random number generator for sampling.

    Returns
    -------
    list[ItemId]
        List of recommended item IDs.
    """
    average_recs = np_rng.choice(default_recs, n_rec)
    if inner_id:
        return average_recs.tolist()
    else:
        return [data_info.id_converter.id2item[i] for i in average_recs]


def cold_start_rec(
    data_info: DataInfo,
    default_recs: Sequence[int],
    cold_start: str,
    users: Sequence[UserId],
    n_rec: int,
    inner_id: bool,
    np_rng: np.random.Generator,
) -> dict[UserId, list[ItemId]]:
    """Generate recommendations for cold-start users.

    Parameters
    ----------
    data_info : :class:`~endrs.data.DataInfo`
        Object that contains useful information about the dataset.
    default_recs : Sequence[int]
        Default recommendations to choose from.
    cold_start : str
        Strategy for cold-start recommendations. Must be either 'average' or 'popular'.
    users : Sequence[UserId]
        List of user IDs to generate recommendations for.
    n_rec : int
        Number of items to recommend.
    inner_id : bool
        Whether to return items in inner IDs.
    np_rng : np.random.Generator
        Random number generator for sampling.

    Returns
    -------
    dict[UserId, list[ItemId]]
        Dictionary mapping user IDs to lists of recommended item IDs.

    Raises
    ------
    ValueError
        If cold_start is not 'average' or 'popular'.
    """
    if cold_start not in ("average", "popular"):
        raise ValueError(f"Unknown cold start strategy: {cold_start}")
    result_recs = dict()
    for u in users:
        if cold_start == "average":
            result_recs[u] = average_recommendations(
                data_info, default_recs, inner_id, n_rec, np_rng
            )
        else:
            result_recs[u] = popular_recommendations(data_info, inner_id, n_rec, np_rng)
    return result_recs
