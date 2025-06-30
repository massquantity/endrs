from collections.abc import Sequence
from typing import Any

import numpy as np
import torch

from endrs.algorithms.two_tower import TwoTower
from endrs.data.batch import BatchData, EvalBatchData
from endrs.data.data_info import DataInfo
from endrs.feature.feat_info import FeatInfo
from endrs.inference.ranking import ddp_rerank
from endrs.types import ItemId, UserId


class DPP:
    """Determinantal Point Process (DPP) for diverse recommendation.

    To generate diverse recommendations, the model first retrieves candidate
    items from a two-tower model. It then constructs a kernel matrix from the
    item embeddings of these candidates to capture both quality and similarity.
    Finally, a Determinantal Point Process (DPP) is used to sample a diverse
    subset of items from this matrix, balancing relevance and diversity in the
    final recommendation list.

    Parameters
    ----------
    task : {'rating', 'ranking'}
        The recommendation task type. See :ref:`Task`.
    data_info : :class:`~endrs.data.DataInfo`
        Object that contains useful information about the dataset.
    feat_info : :class:`~endrs.feature.FeatInfo` or None, default: None
        Object that contains information about features used in the model.
    embed_size : int, default: 16
        Size of the embedding vectors.
    n_epochs : int, default: 20
        Number of training epochs.
    lr : float, default: 0.001
        Learning rate for the optimizer.
    weight_decay : float, default: 0.0
        Weight decay (L2 regularization) for the optimizer.
    batch_size : int, default: 256
        Number of samples per batch.
    sampler : {'random', 'unconsumed', 'popular'}, default: 'random'
        Strategy for sampling negative examples.
    num_neg : int, default: 1
        Number of negative samples per positive sample, only used in `ranking` task.
    use_bn : bool, default: True
        Whether to use batch normalization in the deep neural network layers.
    dropout_rate : float, default: 0.0
        Dropout rate applied to DNN layers to prevent overfitting.
    hidden_units : Sequence[int], default: (128, 64, 32)
        Sequence of hidden layer sizes for both user and item towers.
    multi_sparse_combiner : str, default: 'sqrtn'
        Method to combine multiple sparse features.
    use_correction : bool, default: True
        Whether to apply popularity-based correction in softmax loss.
        When set to True, the model will adjust the logits based on item popularity
        to counteract the popularity bias in recommendations. This correction factor
        is applied during training with softmax-based losses to give less popular items
        a boost in the probability distribution, potentially improving the diversity of
        recommendations.
    temperature : float, default: 1.0
        Temperature parameter for scaling logits in listwise losses.
    remove_accidental_hits : bool, default: False
        Whether to mask out matching items in the batch that might appear as false negatives.
        When enabled, the model will identify and mask items in the batch that the user has
        actually interacted with (based on the input data) but appear as negative samples due
        to random sampling. This prevents the model from penalizing items that are actually
        positive examples.
    seed : int, default: 42
        Random seed for reproducibility.
    accelerator : str, default: 'auto'
        Hardware accelerator type for training (e.g., 'cpu', 'gpu', 'auto', etc.).
    devices : Sequence[int] or str or int, default: 'auto'
        Devices to use for training.

    References
    ----------
    *Laming Chen et al.* `Fast Greedy MAP Inference for Determinantal Point Process to Improve Recommendation Diversity
    <https://arxiv.org/pdf/1709.05135>`_.
    """

    def __init__(
        self,
        task: str,
        data_info: DataInfo,
        feat_info: FeatInfo | None = None,
        # loss: str = "softmax",
        embed_size: int = 16,
        # norm_embed: bool = False,
        n_epochs: int = 20,
        lr: float = 0.001,
        weight_decay: float = 0.0,
        batch_size: int = 256,
        sampler: str = "random",
        num_neg: int = 1,
        use_bn: bool = True,
        dropout_rate: float = 0.0,
        hidden_units: Sequence[int] = (128, 64, 32),
        multi_sparse_combiner: str = "sqrtn",
        use_correction: bool = True,
        temperature: float = 1.0,
        remove_accidental_hits: bool = False,
        seed: int = 42,
        accelerator: str = "auto",
        devices: Sequence[int] | str | int = "auto",
    ):
        self.n_users = data_info.n_users
        self.n_items = data_info.n_items
        self.id_converter = data_info.id_converter
        self.use_hash = data_info.use_hash
        self.two_tower = TwoTower(
            task=task,
            data_info=data_info,
            feat_info=feat_info,
            loss="softmax",
            embed_size=embed_size,
            norm_embed=True,
            n_epochs=n_epochs,
            lr=lr,
            weight_decay=weight_decay,
            batch_size=batch_size,
            sampler=sampler,
            num_neg=num_neg,
            use_bn=use_bn,
            dropout_rate=dropout_rate,
            hidden_units=hidden_units,
            multi_sparse_combiner=multi_sparse_combiner,
            use_correction=use_correction,
            temperature=temperature,
            remove_accidental_hits=remove_accidental_hits,
            seed=seed,
            accelerator=accelerator,
            devices=devices,
        )

    def fit(
        self,
        train_data: BatchData,
        neg_sampling: bool,
        verbose: int = 1,
        shuffle: bool = True,
        eval_data: EvalBatchData | None = None,
        metrics: Sequence[str] | None = None,
        k: int = 10,
        eval_batch_size: int = 8192,
        eval_user_num: int | None = None,
        num_workers: int = 0,
    ):
        self.two_tower.fit(
            train_data,
            neg_sampling,
            verbose,
            shuffle,
            eval_data,
            metrics,
            k,
            eval_batch_size,
            eval_user_num,
            num_workers,
        )

    def predict(self, *_: Any, **__: Any):
        raise NotImplementedError("DPP does not support predicting.")

    def recommend_user(
        self,
        user: UserId,
        n_rec: int,
        n_candidates: int | None = None,
        filter_consumed: bool = True,
    ) -> dict[UserId, list[ItemId]]:
        """Recommend diverse items for a single user using DPP reranking.

        This method first generates candidate items using the two-tower model,
        then applies DPP reranking to promote diversity in the final recommendation list.

        Parameters
        ----------
        user : :type:`~endrs.types.UserId`
            User ID for whom to generate recommendations.
        n_rec : int
            Number of items to recommend.
        n_candidates : int or None, default: None
            Number of candidate items to generate before DPP reranking.
            If None, defaults to ``n_rec * 20``.
        filter_consumed : bool, default: True
            Whether to filter out items the user has already consumed.

        Returns
        -------
        dict[:type:`~endrs.types.UserId`, list[:type:`~endrs.types.ItemId`]]
            Dictionary mapping the user to their recommended items.

        Raises
        ------
        ValueError
            If ``n_candidates`` is not bigger than ``n_rec`` or exceeds
            the total number of items.
        """
        if n_candidates is None:
            assert isinstance(n_rec, int)
            n_candidates = n_rec * 20
        if n_candidates <= n_rec:
            raise ValueError(
                f"`n_candidates` must be bigger than `n_rec`, got {n_candidates}"
            )
        if n_candidates > self.n_items:
            raise ValueError(
                f"`n_candidates` {n_candidates} exceeds num of items {self.n_items}"
            )
        user_id = self.convert_user(user)
        item_ids, scores = self.compute_recs(user_id, n_candidates, filter_consumed)
        item_embeds = self.get_ddp_embeddings(item_ids)
        kernel_matrix = self.build_kernel_matrix(scores, item_embeds)
        computed_recs = ddp_rerank(kernel_matrix, n_rec, item_ids)
        rec_items = [self.id_converter.id2item[i] for i in computed_recs]
        return {user: rec_items}

    def convert_user(self, user: UserId) -> int:
        assert np.isscalar(user), "DDP only supports single user recommendation."
        if user not in self.id_converter.user2id:
            raise ValueError(
                f"DDP does not support cold start recommendation, got unknown user: {user}"
            )
        return self.id_converter.user2id[user]

    @torch.inference_mode()
    def compute_recs(
        self,
        user_id: int,
        n_candidates: int,
        filter_consumed: bool,
    ) -> tuple[list[int], list[float]]:
        """Compute initial recommendations using the two-tower model.

        Parameters
        ----------
        user_id : int
            Internal user ID.
        n_candidates : int
            Number of candidate items to retrieve.
        filter_consumed : bool
            Whether to filter out consumed items.

        Returns
        -------
        tuple[list[int], list[float]]
            Tuple containing list of candidate item IDs and their prediction scores.
        """
        self.two_tower.eval()
        user_ids = [user_id]
        inputs = self.two_tower.get_rec_inputs(user_ids)
        preds = self.two_tower(inputs).cpu().numpy()
        item_ids, scores = self.two_tower.ranking_model.get_top_items(
            user_ids, preds, n_candidates, filter_consumed, return_scores=True
        )
        return item_ids[0], scores[0]

    @torch.inference_mode()
    def get_ddp_embeddings(self, item_ids: list[int]) -> np.ndarray:
        """Get item embeddings for DPP kernel matrix computation.

        Parameters
        ----------
        item_ids : list[int]
            List of item IDs to get embeddings for.

        Returns
        -------
        np.ndarray
            Item embeddings as a numpy array of shape (n_items, embed_size).
        """
        item_ids = torch.tensor(item_ids, device=self.two_tower.device)
        if self.use_hash:
            item_embeds = self.two_tower.hash_embeds_tensor[item_ids]
        else:
            item_embeds = self.two_tower.item_embeds_tensor[item_ids]
        return item_embeds.cpu().numpy()

    @staticmethod
    def build_kernel_matrix(scores: list[float], item_embeds: np.ndarray) -> np.ndarray:
        """Build the DPP kernel matrix from scores and item embeddings.

        The kernel matrix L is decomposed as L = diag(q) × S × diag(q), where q represents
        item quality scores and S represents item similarity based on embedding dot products.

        Parameters
        ----------
        scores : list[float]
            Prediction scores for each item, representing item quality.
        item_embeds : np.ndarray
            Item embeddings of shape (n_items, embed_size) used to compute similarity.

        Returns
        -------
        np.ndarray
            DPP kernel matrix of shape (n_items, n_items) that encodes both
            item quality and similarity.
        """
        score_diag = np.diag(scores)
        sim_matrix = item_embeds @ item_embeds.T
        # avoid negative values to preserve semi-definite sim matrix
        sim_matrix = (sim_matrix + 1) / 2
        return score_diag.dot(sim_matrix).dot(score_diag)
