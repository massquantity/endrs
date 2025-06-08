import abc
from collections.abc import Mapping, MutableMapping, Sequence
from typing import Any, Literal, Union, override

import numpy as np
import torch

from endrs.bases.torch_base import TorchBase
from endrs.data.batch import BatchData, EvalBatchData
from endrs.data.data_info import DataInfo
from endrs.feature.feat_info import FeatInfo
from endrs.inference.preprocess import get_item_inputs, get_user_inputs
from endrs.torchops.loss import (
    bpr_loss,
    max_margin_loss,
    softmax_cross_entropy_loss,
)
from endrs.types import ItemId
from endrs.utils.constants import (
    ALL_LOSSES,
    ITEM_KEY,
    LISTWISE_LOSS,
    OOV_IDX,
    POINTWISE_LOSS,
)


class TorchEmbedBase(TorchBase):
    """Base class for embedding-based recommender systems using PyTorch Lightning.

    Parameters
    ----------
    task : {'rating', 'ranking'}
        The recommendation task type. See :ref:`Task`.
    data_info : :class:`~endrs.data.DataInfo`
        Object that contains useful information about the dataset.
    feat_info : :class:`~endrs.feature.FeatInfo`
        Object that contains information about features used in the model.
    loss : str
        The loss to use for training (e.g., 'bce', 'focal', 'softmax', etc.).
    embed_size : int
        Size of the embedding vectors.
    norm_embed : bool
        Whether to normalize embeddings.
    n_epochs : int
        Number of training epochs.
    lr : float
        Learning rate for the optimizer.
    weight_decay : float
        Weight decay (L2 regularization) for the optimizer.
    batch_size : int
        Number of samples per batch.
    sampler : {'random', 'unconsumed', 'popular'}, default: 'random'
        Strategy for sampling negative examples.
    num_neg : int
        Number of negative samples per positive sample, only used in `ranking` task.
    use_correction : bool
        Whether to apply popularity-based correction in softmax loss.
        When set to True, the model will adjust the logits based on item popularity 
        to counteract the popularity bias in recommendations. This correction factor 
        is applied during training with softmax-based losses to give less popular items 
        a boost in the probability distribution, potentially improving the diversity of 
        recommendations.
    temperature : float
        Temperature parameter for scaling logits in listwise losses.
    remove_accidental_hits : bool
        Whether to mask out matching items in the batch that might appear as false negatives.
        When enabled, the model will identify and mask items in the batch that the user has 
        actually interacted with (based on the input data) but appear as negative samples due 
        to random sampling. This prevents the model from penalizing items that are actually 
        positive examples.
    seed : int
        Random seed for reproducibility.
    accelerator : str
        Hardware accelerator type for training (e.g., 'cpu', 'gpu', 'auto', etc.).
    devices : Sequence[int] or str or int
        Devices to use for training.
    paradigm : {'u2i', 'i2i'}, default: 'u2i'
        Recommendation paradigm: user-to-item (u2i) or item-to-item (i2i).

    Attributes
    ----------
    user_embeds_tensor : torch.Tensor
        Tensor storing user embeddings (when not using hash mode).
    item_embeds_tensor : torch.Tensor
        Tensor storing item embeddings (when not using hash mode).
    hash_embeds_tensor : torch.Tensor
        Tensor storing hash-based embeddings (when using hash mode).
    item_corrections : torch.Tensor
        Popularity-based correction factors for items when using softmax loss.
    """

    def __init__(
        self,
        task: str,
        data_info: DataInfo,
        feat_info: FeatInfo,
        loss: str,
        embed_size: int,
        norm_embed: bool,
        n_epochs: int,
        lr: float,
        weight_decay: float,
        batch_size: int,
        sampler: str,
        num_neg: int,
        use_correction: bool,
        temperature: float,
        remove_accidental_hits: bool,
        seed: int,
        accelerator: str,
        devices: Sequence[int] | str | int,
        paradigm: Literal["u2i", "i2i"] = "u2i",
    ):
        super().__init__(
            task,
            data_info,
            feat_info,
            loss,
            embed_size,
            n_epochs,
            lr,
            weight_decay,
            batch_size,
            sampler,
            num_neg,
            seed,
            accelerator,
            devices,
        )
        assert temperature > 0.0
        self.norm_embed = norm_embed
        self.use_correction = use_correction
        self.temperature = temperature
        self.remove_accidental_hits = remove_accidental_hits
        self.paradigm = paradigm
        if self.use_hash:
            hash_embeds_tensor = torch.zeros(
                self.data_info.n_hash_bins + 1, self.embed_size, device=self.device
            )
            self.register_buffer("hash_embeds_tensor", hash_embeds_tensor)
        else:
            user_embeds_tensor = torch.zeros(
                self.n_users + 1, self.embed_size, device=self.device
            )
            item_embeds_tensor = torch.zeros(
                self.n_items + 1, self.embed_size, device=self.device
            )
            self.register_buffer("user_embeds_tensor", user_embeds_tensor)
            self.register_buffer("item_embeds_tensor", item_embeds_tensor)

    @override
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
        enable_early_stopping: bool = False,
        patience: int = 3,
        checkpoint_path: str | None = None,
    ):
        if self.loss == "softmax" and self.use_correction:
            num = self.data_info.n_hash_bins + 1 if self.use_hash else self.n_items + 1
            self.item_corrections = torch.ones(
                num, dtype=torch.float32, device=self.device
            )
            # item indices start from 1, skip oov 0
            indices, item_counts = np.unique(train_data.items, return_counts=True)
            item_counts = torch.as_tensor(
                item_counts, dtype=torch.float32, device=self.device
            )
            self.item_corrections[indices] = item_counts / len(train_data)

        super().fit(
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
            enable_early_stopping,
            patience,
            checkpoint_path,
        )

    @override
    def on_validation_epoch_start(self):
        super().on_validation_epoch_start()
        self.set_embeddings()

    @override
    def on_train_end(self):
        self.set_embeddings()
        super().on_train_end()

    @override
    @torch.inference_mode()
    def _pred_inner(self, users: np.ndarray, items: np.ndarray) -> np.ndarray:
        self.eval()
        if self.use_hash:
            user_embeds = self.hash_embeds_tensor[users]
            item_embeds = self.hash_embeds_tensor[items]
        else:
            user_embeds = self.user_embeds_tensor[users]
            item_embeds = self.item_embeds_tensor[items]
        preds = torch.sum(user_embeds * item_embeds, dim=1)
        return preds.cpu().numpy()

    @override
    @torch.inference_mode()
    def _rec_inner(
        self,
        user_ids: list[int],
        n_rec: int,
        filter_consumed: bool,
        user_feats: Mapping[str, Any] | None = None,
        seq: list[ItemId] | None = None,
        inner_seq_ids: bool = False,
        random_rec: bool = False,
    ) -> list[list[int]]:
        if user_feats is not None:
            raise ValueError(f"{self.name} doesn't support arbitrary features.")
        if seq is not None:
            raise ValueError(f"{self.name} doesn't support arbitrary item sequence.")

        self.eval()
        if self.use_hash:
            user_embed = self.hash_embeds_tensor[user_ids]
            item_embeds = self.hash_embeds_tensor[self.cand_items]
        else:
            user_embed = self.user_embeds_tensor[user_ids]  # 2D, B * E
            item_embeds = self.item_embeds_tensor[self.cand_items]
        preds = (user_embed @ item_embeds.t()).cpu().numpy()
        return self.ranking_model.get_top_items(
            user_ids, preds, n_rec, filter_consumed, random_rec
        )

    @abc.abstractmethod
    def get_user_embeddings(self, inputs: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """Compute user embeddings from inputs."""

    @abc.abstractmethod
    def get_item_embeddings(self, inputs: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """Compute item embeddings from inputs."""

    def check_params(self):
        if self.loss not in ALL_LOSSES:
            raise ValueError(f"`loss` must be one of {ALL_LOSSES}, got {self.loss}")

    @override
    def compute_loss(
        self,
        batch: Union[
            MutableMapping[str, torch.Tensor],
            tuple[
                Mapping[str, torch.Tensor],
                Mapping[str, torch.Tensor],
                Mapping[str, torch.Tensor],
            ],
        ],
    ) -> torch.Tensor:
        if self.task == "rating" or self.loss in POINTWISE_LOSS:
            loss = super().compute_loss(batch)
        elif self.loss in LISTWISE_LOSS:
            logits = self.compute_list_inputs(batch)
            targets = torch.arange(logits.shape[0], device=self.device)
            loss = softmax_cross_entropy_loss(logits, targets)
        else:
            assert isinstance(batch, tuple) and len(batch) == 3
            pos_scores, neg_scores = self.compute_pair_inputs(*batch)
            if self.loss == "bpr":
                loss = bpr_loss(pos_scores, neg_scores)
            else:
                loss = max_margin_loss(pos_scores, neg_scores, margin=1.0)
        return loss

    def compute_pair_inputs(
        self,
        target_inputs: Mapping[str, torch.Tensor],
        item_pos_inputs: Mapping[str, torch.Tensor],
        item_neg_inputs: Mapping[str, torch.Tensor],
        repeat_positives: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.paradigm == "i2i":
            targets = self.get_item_embeddings(target_inputs)
        else:
            targets = self.get_user_embeddings(target_inputs)
        items_pos = self.get_item_embeddings(item_pos_inputs)
        items_neg = self.get_item_embeddings(item_neg_inputs)

        if len(targets) == len(items_pos) == len(items_neg):
            pos_scores = (targets * items_pos).sum(1)
            neg_scores = (targets * items_neg).sum(1)
            return pos_scores, neg_scores

        pos_len, neg_len = len(items_pos), len(items_neg)
        if neg_len % pos_len != 0:
            raise ValueError(
                f"negatives length is not a multiple of positives length, "
                f"got {neg_len} and {pos_len}"
            )
        factor = int(neg_len / pos_len)
        pos_scores = torch.einsum("ij,ij->i", targets, items_pos)
        if repeat_positives:
            pos_scores = pos_scores.repeat_interleave(factor)
        items_neg = items_neg.view(pos_len, factor, -1)
        # neg_scores = torch.einsum("i...k,ijk->ij", targets, items_neg).ravel()
        neg_scores = torch.matmul(targets, items_neg.transpose(1, 2)).ravel()
        return pos_scores, neg_scores

    def compute_list_inputs(self, inputs: Mapping[str, torch.Tensor]) -> torch.Tensor:
        # if GCNModels.contains(model.name) or model.name in ("NGCF", "LightGCN"): TODO
        #    user_embeds, item_embeds = model(inputs)
        user_embeds = self.get_user_embeddings(inputs)
        item_embeds = self.get_item_embeddings(inputs)
        logits = torch.matmul(user_embeds, item_embeds.transpose(0, 1))
        logits = self.adjust_logits(logits, inputs[ITEM_KEY])
        return logits

    def adjust_logits(
        self, logits: torch.Tensor, items: torch.Tensor, all_adjust: bool = True
    ) -> torch.Tensor:
        """Adjust logits for training and inference.
        
        This method applies several adjustments to the raw logits:
        1. Scales logits by temperature parameter
        2. Applies popularity-based correction (if enabled)
        3. Masks accidental hits to prevent false negatives (if enabled)
        
        Parameters
        ----------
        logits : torch.Tensor
            Raw similarity scores between users and items.
        items : torch.Tensor
            Item IDs in the current batch.
        all_adjust : bool, default: True
            Whether to apply all adjustments (correction and hit removal).
            
        Returns
        -------
        torch.Tensor
            Adjusted logits ready for loss calculation or ranking.
        """
        logits = torch.div(logits, self.temperature)
        if self.use_correction and all_adjust:
            correction = self.item_corrections[items]
            correction = torch.clamp(correction, 1e-8, 1.0)
            logQ = torch.log(correction).view(1, -1)
            logits -= logQ

        if self.remove_accidental_hits and all_adjust:
            row_items = items.view(1, -1)
            col_items = items.view(-1, 1)
            equal_items = (row_items == col_items).float()
            label_diag = torch.eye(logits.shape[0]).float()
            mask = (equal_items - label_diag).bool()
            min_val = torch.finfo(torch.float32).min
            paddings = torch.full(logits.shape, min_val, device=self.device)
            logits = torch.where(mask, paddings, logits)

        return logits

    @torch.inference_mode()
    def set_embeddings(self):
        """Precompute and store embeddings for all users and items.
        
        For hash-based embeddings, a single tensor is used to store both user and item embeddings.
        For standard embeddings, separate tensors are used for users and items.
        """
        self.eval()
        users, items = self.cand_users, self.cand_items
        user_inputs = get_user_inputs(users, self.feat_info, self.device)
        if self.seq_params:
            user_inputs = self.build_seq(user_inputs, users)
        item_inputs = get_item_inputs(items, self.feat_info, self.device)

        if self.use_hash:
            self.hash_embeds_tensor[users] = self.get_user_embeddings(user_inputs)
            self.hash_embeds_tensor[items] = self.get_item_embeddings(item_inputs)
        else:
            self.user_embeds_tensor[users] = self.get_user_embeddings(user_inputs)
            self.item_embeds_tensor[items] = self.get_item_embeddings(item_inputs)

    @override
    @torch.inference_mode()
    def assign_embed_oovs(self):
        """Assign embeddings for out-of-vocabulary (OOV) IDs."""
        if self.use_hash:
            all_ids = [*self.cand_users, *self.cand_items]
            self.hash_embeds_tensor[OOV_IDX] = torch.mean(
                self.hash_embeds_tensor[all_ids], dim=0
            )
        else:
            self.user_embeds_tensor[OOV_IDX] = torch.mean(
                self.user_embeds_tensor[1:], dim=0
            )
            self.item_embeds_tensor[OOV_IDX] = torch.mean(
                self.item_embeds_tensor[1:], dim=0
            )
