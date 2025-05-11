import random
from collections.abc import Mapping, Sequence, Set
from dataclasses import dataclass

import numpy as np

from endrs.utils.constants import OOV_IDX


@dataclass
class SeqParams:
    max_seq_len: int
    cached_seqs: np.ndarray | Mapping[int, np.ndarray]


@dataclass
class DualSeqParams:
    long_max_len: int
    short_max_len: int
    cached_long_seqs: np.ndarray
    cached_short_seqs: np.ndarray


def get_interacted_seqs(
    user_indices: np.ndarray,
    item_indices: np.ndarray,
    user_consumed: Mapping[int, Sequence[int]],
    max_seq_len: int,
    user_consumed_set: Mapping[int, Set[int]],
) -> np.ndarray:
    """Get users' historical interaction sequences up to a certain position.
    
    For positive samples (items the user has interacted with), the function finds the exact
    position of the item in the user's history. For negative samples (items not in the 
    user's history), it randomly samples a position from the user's history.

    Parameters
    ----------
    user_indices : np.ndarray
        Array of user indices in the current batch.
    item_indices : np.ndarray
        Array of item indices in the current batch, corresponding to user_indices.
    user_consumed : Mapping[int, Sequence[int]]
        Dictionary mapping users to their chronologically ordered consumed items.
    max_seq_len : int
        Maximum length of sequence to return. If a user's history is longer, only
        the most recent max_seq_len items before the target position are included.
    user_consumed_set : Mapping[int, Set[int]]
        Dictionary mapping users to sets of their consumed items for faster lookup.
        This is used to quickly check if an item exists in a user's history.

    Returns
    -------
    np.ndarray
        Array of shape (batch_size, max_seq_len) containing users' historical interaction
        sequences up to the target items. Sequences shorter than max_seq_len are padded
        with OOV_IDX at the end.
    """
    batch_size = len(user_indices)
    seqs = np.full((batch_size, max_seq_len), OOV_IDX, dtype=np.int32)
    for j, (u, i) in enumerate(zip(user_indices, item_indices)):
        consumed_items = user_consumed[u]
        consumed_len = len(consumed_items)
        consumed_set = user_consumed_set[u]
        # If `i` is a negative item, sample sequence from user's past interaction
        position = (
            consumed_items.index(i)
            if i in consumed_set
            else random.randrange(0, consumed_len)
        )
        if position == 0:
            # first item has no historical interaction, fill in with pad_index
            continue
        elif position < max_seq_len:
            seqs[j, :position] = consumed_items[:position]
        else:
            start_index = position - max_seq_len
            seqs[j] = consumed_items[start_index:position]

    return seqs


def get_recent_seqs(
    n_users: int, user_consumed: Mapping[int, Sequence[int]], max_seq_len: int
) -> np.ndarray:
    """Get most recent interaction sequences for all users.
    
    The function assumes that items in user_consumed are already sorted chronologically,
    with the most recent items at the end of the list.
    
    Note that the resulting array includes an entry at index 0 (filled with OOV_IDX)
    to accommodate 1-indexed user IDs.

    Parameters
    ----------
    n_users : int
        Number of users in the dataset.
    user_consumed : Mapping[int, Sequence[int]]
        Dictionary mapping users to their consumed items, assumed to be sorted 
        chronologically with most recent items at the end.
    max_seq_len : int
        Maximum length of sequence to extract for each user. If a user has consumed
        fewer items than this, their sequence will be padded with OOV_IDX at the end.

    Returns
    -------
    np.ndarray
        Array of shape (n_users + 1, max_seq_len) containing users' most recent interaction
        sequences.
    """
    recent_seqs = np.full((n_users + 1, max_seq_len), OOV_IDX, dtype=np.int32)
    for u in range(1, n_users + 1):
        u_consumed_items = user_consumed[u]
        u_items_len = len(u_consumed_items)
        if u_items_len < max_seq_len:
            recent_seqs[u, :u_items_len] = u_consumed_items
        else:
            recent_seqs[u] = u_consumed_items[-max_seq_len:]
    return recent_seqs


def get_recent_seq_dict(
    user_consumed: Mapping[int, Sequence[int]], max_seq_len: int
) -> dict[int, np.ndarray]:
    """Get most recent interaction sequences for all users as a dictionary.

    Parameters
    ----------
    user_consumed : Mapping[int, Sequence[int]]
        Dictionary mapping users to their consumed items, assumed to be sorted 
        chronologically with most recent items at the end.
    max_seq_len : int
        Maximum length of sequence to extract for each user. If a user has consumed
        fewer items than this, their sequence will be padded with OOV_IDX.

    Returns
    -------
    dict[int, np.ndarray]
        Dictionary mapping user indices to arrays of their most recent interaction
        sequences.
    """
    recent_seqs = {OOV_IDX: np.full(max_seq_len, OOV_IDX, dtype=np.int32)}
    for u, u_consumed_items in user_consumed.items():
        u_items_len = len(u_consumed_items)
        if u_items_len < max_seq_len:
            seq = np.full(max_seq_len, OOV_IDX, dtype=np.int32)
            seq[:u_items_len] = u_consumed_items
        else:
            seq = np.array(u_consumed_items[-max_seq_len:])
        recent_seqs[u] = seq
    return recent_seqs
