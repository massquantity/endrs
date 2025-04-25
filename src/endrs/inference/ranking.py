from collections.abc import Mapping, Sequence

import numpy as np
from scipy.special import expit, softmax


class Ranking:
    """Class for ranking and selecting items for recommendation.

    Parameters
    ----------
    task : {'rating', 'ranking'}
        The recommendation task type.
    user_consumed : Mapping[int, Sequence[int]]
        Dictionary mapping users to their consumed items.
    np_rng : np.random.Generator
        NumPy random number generator for reproducibility.
    candidate_ids : np.ndarray
        Array of candidate item IDs for recommendation.
    """

    def __init__(
        self,
        task: str,
        user_consumed: Mapping[int, Sequence[int]],
        np_rng: np.random.Generator,
        candidate_ids: np.ndarray,
    ):
        self.task = task
        self.user_consumed = user_consumed
        self.np_rng = np_rng
        self.candidate_ids = candidate_ids
        self.n_items = len(candidate_ids)

    def get_top_items(
        self,
        user_ids: Sequence[int],
        model_preds: np.ndarray,
        n_rec: int,
        filter_consumed: bool,
        random_rec: bool = False,
        return_scores: bool = False,
    ) -> list[list[int]] | tuple[list[list[int]], list[list[float]]]:
        """Get top recommended items for each user based on model predictions.

        This method processes model predictions to select the top items for each user.
        The process involves:
        1. Reshaping predictions if necessary
        2. For each user:
           - Filtering out consumed items if requested
           - Selecting items either randomly (weighted by scores) or deterministically
        3. Sorting the selected items by their prediction scores
        4. Formatting the output based on the return_scores parameter

        Parameters
        ----------
        user_ids : Sequence[int]
            IDs of users to generate recommendations for.
        model_preds : np.ndarray
            Model predictions for user-item pairs. Can be 1D (flattened predictions) or 
            2D (user-item matrix), and will be handled accordingly.
        n_rec : int
            Number of items to recommend for each user. Should be a positive integer
            less than or equal to the number of available items after filtering.
        filter_consumed : bool
            Whether to filter out items the user has already consumed. Has no effect
            if the user has no consumed items or if filtering would leave too few items.
        random_rec : bool, default: False
            Whether to add randomness to recommendations. When True, items are selected
            probabilistically based on their scores. When False, top-n items are selected.
        return_scores : bool, default: False
            Whether to return prediction scores along with item IDs.

        Returns
        -------
        list[list[int]] or tuple[list[list[int]], list[list[float]]]
            If return_scores is False, returns a list of lists, where each inner list
            contains the recommended item IDs for a user.
            If return_scores is True, returns a tuple of (item_ids, scores), where both
            elements are lists of lists.
        """
        batch_size, all_preds = self._prepare_preds(model_preds)
        all_candidate_ids = np.tile(self.candidate_ids, (batch_size, 1))
        batch_ids, batch_preds = [], []
        for i in range(batch_size):
            user = user_ids[i]
            ids = all_candidate_ids[i]
            preds = all_preds[i]
            consumed = self.user_consumed[user] if user in self.user_consumed else []
            if filter_consumed and consumed and n_rec + len(consumed) <= self.n_items:
                ids, preds = filter_items(ids, preds, consumed)

            if random_rec:
                ids, preds = random_select(ids, preds, n_rec, self.np_rng)
            else:
                ids, preds = partition_select(ids, preds, n_rec)

            batch_ids.append(ids)
            batch_preds.append(preds)

        ids, preds = np.array(batch_ids), np.array(batch_preds)
        indices = np.argsort(preds, axis=1)[:, ::-1]
        ids = np.take_along_axis(ids, indices, axis=1)
        return self._items_or_scores(ids, preds, indices, return_scores)

    def _prepare_preds(self, preds: np.ndarray) -> tuple[int, np.ndarray]:
        """Prepare prediction array for processing by reshaping if necessary.

        This method handles different input prediction formats:
        - 1D arrays from torch models (flattened predictions for multiple users)
        - 2D arrays from embedding-based models (user-item matrices)

        Parameters
        ----------
        preds : np.ndarray
            Raw model predictions. Can be either:
            - 1D array of length (batch_size * n_items) for torch models
            - 2D array of shape (batch_size, n_items) for embedding models

        Returns
        -------
        tuple[int, np.ndarray]
            Tuple containing:
            - batch_size: Number of users in the batch
            - reshaped_predictions: 2D array of shape (batch_size, n_items) containing
              the prediction scores for each user-item pair
        """
        # 1d from torch models
        if preds.ndim == 1:
            assert len(preds) % self.n_items == 0
            batch_size = int(len(preds) / self.n_items)
            all_preds = preds.reshape(batch_size, self.n_items)
        # 2d from embed models
        else:
            batch_size = len(preds)
            all_preds = preds
        return batch_size, all_preds

    def _items_or_scores(
        self,
        ids: np.ndarray,
        preds: np.ndarray,
        indices: np.ndarray,
        return_scores: bool,
    ) -> list[list[int]] | tuple[list[list[int]], list[list[float]]]:
        """Prepare final output based on the return_scores flag.

        Parameters
        ----------
        ids : np.ndarray
            2D array of shape (batch_size, n_rec) containing sorted item IDs.
        preds : np.ndarray
            2D array of shape (batch_size, n_items) containing raw prediction scores.
        indices : np.ndarray
            2D array of shape (batch_size, n_rec) containing sorted indices of top items.
        return_scores : bool
            Whether to return prediction scores along with item IDs.

        Returns
        -------
        list[list[int]] or tuple[list[list[int]], list[list[float]]]
            If return_scores is False, returns a list of lists of item IDs.
            If return_scores is True, returns a tuple of (item_ids, scores), where:
            - item_ids is a list of lists of recommended item IDs for each user
            - scores is a list of lists of prediction scores corresponding to those items,
              with values normalized to [0, 1] for ranking tasks
        """
        if return_scores:
            scores = np.take_along_axis(preds, indices, axis=1)
            if self.task == "ranking":
                scores = expit(scores)
            return ids.tolist(), scores.tolist()
        else:
            return ids.tolist()


def filter_items(ids: np.ndarray, preds: np.ndarray, items: Sequence[int]):
    mask = np.isin(ids, items, assume_unique=True, invert=True)
    return ids[mask], preds[mask]


def get_reco_probs(preds: np.ndarray):
    p = np.power(softmax(preds), 0.75) + 1e-8  # avoid zero probs
    return p / p.sum()


def random_select(
    ids: np.ndarray, preds: np.ndarray, n_rec: int, np_rng: np.random.Generator
):
    p = get_reco_probs(preds)
    mask = np_rng.choice(len(preds), n_rec, p=p, replace=False, shuffle=False)
    return ids[mask], preds[mask]


def partition_select(ids: np.ndarray, preds: np.ndarray, n_rec: int):
    mask = np.argpartition(preds, -n_rec)[-n_rec:]
    return ids[mask], preds[mask]


def ddp_rerank(kernel_matrix: np.ndarray, n_rec: int, item_ids: list[int]) -> list[int]:
    """Rerank items using the Determinantal Point Process (DDP) algorithm.

    DDP promotes diversity in the recommendation list by considering 
    both relevance and item similarity. The algorithm selects a subset of items
    that are both high-quality and diverse from each other.

    Parameters
    ----------
    kernel_matrix : np.ndarray
        Similarity matrix between items. This L-ensemble kernel matrix encodes both 
        the quality of items (in diagonal elements) and similarity between items 
        (in off-diagonal elements).
    n_rec : int
        Number of items to recommend.
    item_ids : list[int]
        List of candidate item IDs.

    Returns
    -------
    list[int]
        Reranked list of item IDs that balances relevance and diversity.
        
    References
    ----------
    *Chen, L., Zhang, G., & Zhou, H.* `Fast greedy MAP inference for
    Determinantal Point Process to improve recommendation diversity
    <https://arxiv.org/abs/1709.05135>`_.
    """
    candidate_indices = list(range(len(item_ids)))
    rec_ids = []
    while True:
        chosen_indices = fast_greedy_map(kernel_matrix, n_rec, candidate_indices)
        for i in chosen_indices:
            rec_ids.append(item_ids[i])
        if n_rec == len(chosen_indices):
            break
        n_rec -= len(chosen_indices)
        candidate_indices = list(set(candidate_indices) - set(chosen_indices))
    return rec_ids


def fast_greedy_map(
    kernel_matrix: np.ndarray, n_rec: int, cand_indices: list[int]
) -> list[int]:
    """Fast greedy Maximum A-Posteriori (MAP) inference for DPP.

    Parameters
    ----------
    kernel_matrix : np.ndarray
        Similarity matrix between items.
    n_rec : int
        Number of items to recommend.
    cand_indices : list[int]
        List of candidate indices to choose from.

    Returns
    -------
    list[int]
        Indices of selected items optimized for relevance and diversity.
    """
    # get specific rows and cols from matrix
    inner_kernel_matrix = kernel_matrix[cand_indices][:, cand_indices]
    n_cand = len(cand_indices)
    c = np.zeros((n_cand, n_rec))
    d2 = np.diag(inner_kernel_matrix).copy()
    j = np.argmax(np.log(d2))
    inner_cand_set = set(range(n_cand))
    inner_cand_set.remove(j)
    chosen_idx = [j]
    i = 0
    while len(chosen_idx) < n_rec and d2[j] >= 1e-10:
        inner_cand_idx = list(inner_cand_set)
        L_j = inner_kernel_matrix[j, inner_cand_idx]
        d_j = np.sqrt(d2[j])
        if i == 0:
            e = L_j / d_j
        else:
            c_j = c[j, :i]
            c_i = c[inner_cand_idx, :i]
            e = (L_j - c_j @ c_i.T) / d_j
        c[inner_cand_idx, i] = e
        d2[inner_cand_idx] -= np.square(e)
        j = inner_cand_idx[np.argmax(np.log(d2[inner_cand_idx]))]
        chosen_idx.append(j)
        inner_cand_set.remove(j)
        i += 1
    return [cand_indices[i] for i in chosen_idx]
