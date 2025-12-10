import sys
from collections import OrderedDict, defaultdict
from collections.abc import Mapping, Sequence

import numpy as np


def interaction_consumed(
    user_indices: np.ndarray | Sequence[int],
    item_indices: np.ndarray | Sequence[int],
) -> tuple[dict[int, list[int]], dict[int, list[int]]]:
    """Build dictionaries mapping users to their consumed items and items to users who consumed them.

    This function processes user-item interaction data and creates two dictionaries:
    one mapping users to their consumed items, and another mapping items to users who consumed them.
    The underlying rust function will remove consecutive repeated elements.

    Parameters
    ----------
    user_indices : np.ndarray or Sequence[int]
        Array or sequence of user indices.
    item_indices : np.ndarray or Sequence[int]
        Array or sequence of item indices.

    Returns
    -------
    tuple[dict[int, list[int]], dict[int, list[int]]]
        A tuple containing two dictionaries:
        - First dictionary maps each user to a list of their consumed items.
        - Second dictionary maps each item to a list of users who consumed it.
        Both dictionaries have duplicates removed.
    """
    if isinstance(user_indices, np.ndarray):
        user_indices = user_indices.tolist()
    if isinstance(item_indices, np.ndarray):
        item_indices = item_indices.tolist()

    try:
        from endrs_ext import build_consumed_unique

        return build_consumed_unique(user_indices, item_indices)
    except ImportError:  # pragma: no cover
        return _interaction_consumed(user_indices, item_indices)


def _interaction_consumed(
    user_indices: Sequence[int], item_indices: Sequence[int]
) -> tuple[dict[int, list[int]], dict[int, list[int]]]:  # pragma: no cover
    """Create dictionaries mapping users to consumed items and items to users.

    This is a fallback Python implementation used when the Rust extension is not available.

    Parameters
    ----------
    user_indices : Sequence[int]
        Sequence of user indices.
    item_indices : Sequence[int]
        Sequence of item indices.

    Returns
    -------
    tuple[dict[int, list[int]], dict[int, list[int]]]
        A tuple containing:
        - Dictionary mapping users to their consumed items
        - Dictionary mapping items to users who consumed them
    """
    user_consumed = defaultdict(list)
    item_consumed = defaultdict(list)
    for u, i in zip(user_indices, item_indices):
        user_consumed[u].append(i)
        item_consumed[i].append(u)
    return _remove_duplicates(user_consumed, item_consumed)


def _remove_duplicates(
    user_consumed: Mapping[int, Sequence[int]],
    item_consumed: Mapping[int, Sequence[int]]
) -> tuple[dict[int, list[int]], dict[int, list[int]]]:  # pragma: no cover
    """Remove duplicate entries in consumed dictionaries.

    Parameters
    ----------
    user_consumed : Mapping[int, Sequence[int]]
        Dictionary mapping users to their consumed items.
    item_consumed : Mapping[int, Sequence[int]]
        Dictionary mapping items to users who consumed them.

    Returns
    -------
    tuple[dict[int, list[int]], dict[int, list[int]]]
        A tuple containing:
        - Dictionary mapping users to their consumed items (without duplicates)
        - Dictionary mapping items to users who consumed them (without duplicates)

    Notes
    -----
    This function uses different approaches for Python 3.7+ versus older versions since
    dictionary key order is guaranteed to be insertion order in Python 3.7+.
    """
    # keys will preserve order in dict since Python3.7
    if sys.version_info[:2] >= (3, 7):
        dict_func = dict.fromkeys
    else:  # pragma: no cover
        dict_func = OrderedDict.fromkeys
    user_dedup = {u: list(dict_func(items)) for u, items in user_consumed.items()}
    item_dedup = {i: list(dict_func(users)) for i, users in item_consumed.items()}
    return user_dedup, item_dedup


def merge_consumed_data(
    existing_user_consumed: dict[int, list[int]],
    existing_item_consumed: dict[int, list[int]],
    new_user_consumed: dict[int, list[int]],
    new_item_consumed: dict[int, list[int]],
) -> tuple[dict[int, list[int]], dict[int, list[int]]]:
    """Merge new consumed data with existing consumed data.

    Parameters
    ----------
    existing_user_consumed : dict[int, list[int]]
        Existing mapping of user IDs to consumed item IDs.
    existing_item_consumed : dict[int, list[int]]
        Existing mapping of item IDs to consuming user IDs.
    new_user_consumed : dict[int, list[int]]
        New mapping of user IDs to consumed item IDs.
    new_item_consumed : dict[int, list[int]]
        New mapping of item IDs to consuming user IDs.

    Returns
    -------
    tuple[dict[int, list[int]], dict[int, list[int]]]
        Merged user_consumed and item_consumed dictionaries.
    """
    merged_user_consumed = dict(existing_user_consumed)
    for user_id, items in new_user_consumed.items():
        if user_id in merged_user_consumed:
            merged_user_consumed[user_id].extend(items)
        else:
            merged_user_consumed[user_id] = list(items)

    merged_item_consumed = dict(existing_item_consumed)
    for item_id, users in new_item_consumed.items():
        if item_id in merged_item_consumed:
            merged_item_consumed[item_id].extend(users)
        else:
            merged_item_consumed[item_id] = list(users)

    return merged_user_consumed, merged_item_consumed
