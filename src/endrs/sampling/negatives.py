import math
import random
from collections.abc import Mapping, Sequence, Set

import numpy as np


def _check_invalid_negatives(
    negatives: np.ndarray,
    items_pos: np.ndarray,
    items: np.ndarray | None,
    num_neg: int,
) -> list[int]:
    """Check for invalid negative samples.

    Parameters
    ----------
    negatives : np.ndarray
        Array of sampled negative items.
    items_pos : np.ndarray
        Array of positive items that should not appear in negatives.
    items : np.ndarray or None
        Additional array of items that should not appear in negatives.
    num_neg : int
        Number of negative samples per positive sample.

    Returns
    -------
    list[int]
        Indices of invalid negative samples.
    """
    items_pos = np.repeat(items_pos, num_neg)
    if items is not None:
        items = np.repeat(items, num_neg)

    if items is not None and len(items) > 0:
        invalid_indices = np.union1d(
            np.where(negatives == items_pos)[0], np.where(negatives == items)[0]
        )
    else:
        invalid_indices = np.where(negatives == items_pos)[0]
    return invalid_indices.tolist()


def negatives_from_random(
    np_rng: np.random.Generator,
    items_pos: np.ndarray,
    num_neg: int,
    candidates: np.ndarray,
    items: np.ndarray | None = None,
    tolerance: int = 10,
) -> np.ndarray:
    """Sample negative items using random sampling.

    Parameters
    ----------
    np_rng : np.random.Generator
        NumPy random number generator.
    items_pos : np.ndarray
        Array of positive items for which negatives need to be sampled.
    num_neg : int
        Number of negative samples per positive sample.
    candidates : np.ndarray
        Array of candidate items to sample from.
    items : np.ndarray or None, default: None
        Additional array of items that should not be sampled as negatives.
    tolerance : int, default: 10
        Maximum number of attempts to resample invalid negatives.

    Returns
    -------
    np.ndarray
        Array of sampled negative items.
    """
    sample_num = len(items_pos) * num_neg
    replace = False if len(items_pos) < len(candidates) else True
    negatives = np_rng.choice(candidates, size=sample_num, replace=replace)
    for _ in range(tolerance):
        invalid_indices = _check_invalid_negatives(negatives, items_pos, items, num_neg)
        if not invalid_indices:
            break
        negatives[invalid_indices] = np_rng.choice(
            candidates, size=len(invalid_indices), replace=True
        )
    return negatives


def negatives_from_popular(
    np_rng: np.random.Generator,
    items_pos: np.ndarray,
    num_neg: int,
    candidates: np.ndarray,
    items: np.ndarray | None = None,
    probs: Sequence[float] | None = None,
) -> np.ndarray:
    """Sample negative items based on item popularity.

    Parameters
    ----------
    np_rng : np.random.Generator
        NumPy random number generator.
    items_pos : np.ndarray
        Array of positive items for which negatives need to be sampled.
    num_neg : int
        Number of negative samples per positive sample.
    candidates : np.ndarray
        Array of candidate items to sample from.
    items : np.ndarray or None, default: None
        Additional array of items that should not be sampled as negatives.
    probs : Sequence[float] or None, default: None
        Probability distribution for sampling. Higher values indicate more popular items.

    Returns
    -------
    np.ndarray
        Array of sampled negative items.
    """
    sample_num = len(items_pos) * num_neg
    negatives = np_rng.choice(candidates, size=sample_num, replace=True, p=probs)
    invalid_indices = _check_invalid_negatives(negatives, items_pos, items, num_neg)
    if invalid_indices:
        negatives[invalid_indices] = np_rng.choice(
            candidates, size=len(invalid_indices), replace=True, p=probs
        )
    return negatives


# def negatives_from_out_batch(
#     np_rng: np.random.Generator,
#     n_items: int,
#     items_pos: np.ndarray,
#     items: np.ndarray,
#     num_neg: int,
# ) -> np.ndarray:
#     sample_num = len(items_pos) * num_neg
#     candidate_items = list(set(range(1, n_items + 1)) - set(items_pos) - set(items))
#     if not candidate_items:
#         candidates = np.arange(1, n_items + 1)
#         return np_rng.choice(candidates, size=sample_num, replace=True)
#     replace = False if sample_num < len(candidate_items) else True
#     return np_rng.choice(candidate_items, size=sample_num, replace=replace)


def negatives_from_unconsumed(
    user_consumed_set: Mapping[int, Set[int]],
    users: np.ndarray,
    items: np.ndarray,
    n_items: int,
    num_neg: int,
    candidates: np.ndarray | None,
    tolerance: int = 10,
) -> np.ndarray:
    """Sample negative items that haven't been consumed by the user.

    Parameters
    ----------
    user_consumed_set : Mapping[int, Set[int]]
        Mapping of user IDs to sets of consumed item IDs.
    users : np.ndarray
        Array of user IDs.
    items : np.ndarray
        Array of positive item IDs for which negatives need to be sampled.
    n_items : int
        Total number of items in the dataset.
    num_neg : int
        Number of negative samples per positive sample.
    candidates : np.ndarray or None
        Array of candidate items to sample from. If None, samples from all items.
    tolerance : int, default: 10
        Maximum number of attempts to find a valid negative item.

    Returns
    -------
    np.ndarray
        Array of sampled negative items.
    """
    def sample_one():
        if candidates is not None:
            return random.choice(candidates)
        else:
            return math.floor(n_items * random.random()) + 1

    negatives = []
    for u, i in zip(users, items):
        u_negs = []
        for _ in range(num_neg):
            success = False
            n = sample_one()
            for _ in range(tolerance):
                if n != i and n not in u_negs and n not in user_consumed_set[u]:
                    success = True
                    break
                n = sample_one()
            if not success:
                for _ in range(tolerance):
                    if n != i and n not in u_negs:
                        break
                    n = sample_one()
            u_negs.append(n)
        negatives.extend(u_negs)
    return np.array(negatives)


def neg_probs_from_frequency(
    candidate_items: np.ndarray,
    item_consumed: Mapping[int, Sequence[int]],
    temperature: float,
) -> np.ndarray:
    """Calculate sampling probabilities based on item frequency.

    Parameters
    ----------
    candidate_items : np.ndarray
        Array of candidate item IDs.
    item_consumed : Mapping[int, Sequence[int]]
        Mapping of item IDs to sequences of users who consumed them.
    temperature : float
        Temperature parameter to control probability distribution.
        Higher values make popular items more likely to be sampled.

    Returns
    -------
    np.ndarray
        Normalized probabilities for each candidate item.
    """
    freqs = []
    for i in candidate_items:
        freq = len(set(item_consumed[i]))
        if temperature != 1.0:
            freq = pow(freq, temperature)
        freqs.append(freq)
    freqs = np.array(freqs)
    return freqs / np.sum(freqs)


def pos_probs_from_frequency(
    candidate_items: np.ndarray,
    item_consumed: Mapping[int, Sequence[int]],
    n_users: int,
    alpha: float,
) -> list[float]:
    """Calculate positive sampling probabilities based on frequency.

    Parameters
    ----------
    candidate_items : np.ndarray
        Array of candidate item IDs.
    item_consumed : Mapping[int, Sequence[int]]
        Mapping of item IDs to sequences of users who consumed them.
    n_users : int
        Total number of users in the dataset.
    alpha : float
        Smoothing parameter to control the probability distribution.

    Returns
    -------
    list[float]
        List of sampling probabilities for positive items.
    """
    probs = []
    for i in candidate_items:
        prob = len(set(item_consumed[i])) / n_users
        prob = (math.sqrt(prob / alpha) + 1) * (alpha / prob)
        probs.append(prob)
    return probs
