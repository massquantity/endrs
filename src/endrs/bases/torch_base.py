from collections.abc import Mapping, MutableMapping, Sequence
from pathlib import Path
from typing import Any, override

import lightning as L
import numpy as np
import pandas as pd
import torch
from lightning import seed_everything
from torch import optim
from torch.utils.data import BatchSampler, DataLoader, RandomSampler, SequentialSampler

from endrs.data.batch import BatchData, EvalBatchData
from endrs.data.collators import BaseCollator as Collator, PairwiseCollator
from endrs.data.data_info import DataInfo
from endrs.data.sequence import (
    DualSeqParams,
    SeqParams,
    get_recent_seq_dict,
    get_recent_seqs,
)
from endrs.evaluation.evaluator import Evaluator
from endrs.feature.feat_info import FeatInfo
from endrs.inference.cold_start import cold_start_rec
from endrs.inference.postprocess import construct_rec, normalize_prediction
from endrs.inference.preprocess import (
    convert_ids,
    get_item_inputs,
    get_seq_inputs,
    get_unknown,
    get_user_inputs,
    sep_unknown_users,
)
from endrs.inference.ranking import Ranking
from endrs.torchops.embedding import UnionEmbedding, Embedding
from endrs.torchops.loss import (
    binary_cross_entropy_loss,
    focal_loss,
    mse_loss,
)
from endrs.types import ItemId, UserId
from endrs.utils.constants import LABEL_KEY, LISTWISE_LOSS, OOV_IDX, PAIRWISE_LOSS
from endrs.utils.misc import LightningProgressBar
from endrs.utils.validate import check_labels, check_multi_labels


class TorchBase(L.LightningModule):
    """Base class for recommender system models using PyTorch Lightning.

    Parameters
    ----------
    task : {'rating', 'ranking'}
        The recommendation task type. See :ref:`Task`.
    data_info : :class:`~endrs.data.DataInfo`
        Object that contains useful information about the dataset.
    feat_info : :class:`~endrs.feature.FeatInfo`
        Object that contains information about features used in the model.
    loss : str
        The loss to use for training (e.g., 'bce', 'focal', etc.).
    embed_size : int
        Size of the embedding vectors.
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

        - ``'random'`` means random sampling.
        - ``'unconsumed'`` samples items that the target user did not consume before.
        - ``'popular'`` has a higher probability to sample popular items as negative samples.

    num_neg : int
        Number of negative samples per positive sample, only used in `ranking` task.
    seed : int
        Random seed for reproducibility.
    accelerator : str
        Hardware accelerator type for training (e.g., 'cpu', 'gpu', 'auto', etc.).
    devices : Sequence[int] or str or int
        Devices to use for training.

    Attributes
    ----------
    name : str
        Name of the model class.
    n_users : int
        Number of users in the dataset.
    n_items : int
        Number of items in the dataset.
    user_consumed : Mapping[int, Sequence[int]]
        Dictionary mapping users to their consumed items.
    id_converter : :class:`~endrs.data_info.IdConverter`
        Converter between internal and original IDs.
    cand_users : np.ndarray
        Candidate users for recommendation, which are represented as internal IDs.
    cand_items : np.ndarray
        Candidate items for recommendation, which are represented as internal IDs.
    ranking_model : :class:`~endrs.inference.Ranking`
        Model for ranking items during recommendation.
    is_multi_task : bool
        Whether the model supports multi-task learning.
    evaluator : :class:`~endrs.evaluation.Evaluator` or None
        Evaluator object for model evaluation.
    seq_params : :class:`~endrs.data.sequence.SeqParams` or None
        Parameters for sequential recommendation.
    """

    def __init__(
        self,
        task: str,
        data_info: DataInfo,
        feat_info: FeatInfo,
        loss: str,
        embed_size: int,
        n_epochs: int,
        lr: float,
        weight_decay: float,
        batch_size: int,
        sampler: str,
        num_neg: int,
        seed: int,
        accelerator: str,
        devices: Sequence[int] | str | int,
    ):
        super().__init__()
        assert num_neg > 0
        self.name = self.__class__.__name__
        self.task = task
        self.data_info = data_info
        self.feat_info = feat_info
        self.loss = loss
        self.embed_size = embed_size
        self.n_epochs = n_epochs
        self.lr = lr
        self.wd = weight_decay
        self.batch_size = batch_size
        self.sampler = sampler
        self.num_neg = num_neg
        self.seed = seed
        self.accelerator = accelerator
        self.devices = devices
        self.n_users = data_info.n_users
        self.n_items = data_info.n_items
        self.user_consumed = data_info.user_consumed
        self.id_converter = data_info.id_converter
        self.use_hash = data_info.use_hash
        self.cand_users = data_info.candidate_users
        self.cand_items = data_info.candidate_items
        self.np_rng = np.random.default_rng(seed)
        self.ranking_model = Ranking(
            task, self.user_consumed, self.np_rng, self.cand_items
        )
        self.is_multi_task = False
        self.evaluator = None
        self.seq_params = None
        self.verbose = 0
        seed_everything(seed, workers=True)

    def fit(
        self,
        train_data: BatchData,
        neg_sampling: bool,
        verbose: int = 2,
        shuffle: bool = True,
        eval_data: EvalBatchData | None = None,
        metrics: Sequence[str] | None = None,
        k: int = 10,
        eval_batch_size: int = 8192,
        eval_user_num: int | None = None,
        num_workers: int = 0,
    ):
        """Train the model on the provided data.

        Parameters
        ----------
        train_data : :class:`~endrs.data.BatchData`
            Training data.
        neg_sampling : bool
            Whether to use negative sampling.

            .. NOTE::
               Negative sampling is needed if your data is implicit(i.e., `task` is ranking)
               and ONLY contains positive labels. Otherwise, it should be False.

        verbose : int, default: 2
            Verbosity level (0=quiet, 1=epoch results, 2=progress bar, 3=detailed).

            - ``verbose <= 0``: Print nothing.
            - ``verbose >= 1`` : Print evaluation metrics if ``eval_data`` is provided.
            - ``verbose >= 2`` : Enable progress bar.
            - ``verbose >= 2`` : Print model summary.

        shuffle : bool, default: True
            Whether to shuffle the training data.
        eval_data : :class:`~endrs.data.EvalBatchData` or None, default: None
            Evaluation data for validation during training.
        metrics : Sequence[str] or None, default: None
            Evaluation metrics to calculate during validation.
        k : int, default: 10
            Number of items to recommend for evaluation metrics.
        eval_batch_size : int, default: 8192
            Batch size for evaluation.
        eval_user_num : int or None, default: None
            Number of users to sample for evaluation.
            Setting it to a positive number will sample users randomly from eval data.
        num_workers : int, default: 0
            Number of workers for data loading.
            0 means that the data will be loaded in the main process,
            which is slower than multiprocessing.

            .. CAUTION::
               Using multiprocessing(``num_workers`` > 0) may consume more memory than
               single processing. See `Multi-process data loading <https://pytorch.org/docs/stable/data.html#multi-process-data-loading>`_.
        """
        self.check_data_labels(train_data, neg_sampling)
        if eval_data:
            self.check_data_labels(eval_data, neg_sampling)
            self.evaluator = Evaluator(
                during_training=True,
                data=eval_data,
                model=self,
                is_multi_task=self.is_multi_task,
                neg_sampling=neg_sampling,
                metrics=metrics,
                k=k,
                eval_batch_size=eval_batch_size,
                sample_user_num=eval_user_num,
                seed=self.seed,
                verbose=verbose,
                num_workers=num_workers,
            )

        train_dataloader = self.get_batch_loader(
            train_data, neg_sampling, self.batch_size, shuffle, num_workers
        )
        eval_dataloader = (
            self.evaluator.build_data_loader() if self.evaluator else None
        )
        # TODO: enable_checkpointing, enable_model_summary(verbose >= 2), early stopping
        enable_pbar = True if verbose >= 2 else False
        leave = True if verbose >= 3 else False
        enable_model_summary = True if verbose >= 3 else False
        callbacks = [LightningProgressBar(self.name, leave)] if enable_pbar else None
        trainer = L.Trainer(
            accelerator=self.accelerator,
            devices=self.devices,
            max_epochs=self.n_epochs,
            check_val_every_n_epoch=1,
            logger=False,
            log_every_n_steps=0,
            enable_checkpointing=False,
            enable_progress_bar=enable_pbar,
            enable_model_summary=enable_model_summary,
            accumulate_grad_batches=1,
            gradient_clip_val=None,
            num_sanity_val_steps=0,
            default_root_dir=Path.cwd(),
            callbacks=callbacks,
        )
        trainer.fit(
            model=self,
            train_dataloaders=train_dataloader,
            val_dataloaders=eval_dataloader,
        )

    def evaluate(
        self,
        data: EvalBatchData | pd.DataFrame,
        neg_sampling: bool,
        metrics: str | Sequence[str] | None = None,
        k: int = 10,
        eval_batch_size: int = 8192,
        eval_user_num: int | None = None,
        verbose: int = 2,
        num_workers: int = 0,
    ) -> dict[str, float]:
        """Evaluate the model on the provided data.

        Parameters
        ----------
        data : :class:`~endrs.data.EvalBatchData` or :class:`pandas.DataFrame`
            Data to evaluate on.
        neg_sampling : bool
            Whether to use negative sampling.
        metrics : str or Sequence[str] or None, default: None
            Evaluation metrics to calculate.
        k : int, default: 10
            Number of items to recommend for evaluation metrics.
        eval_batch_size : int, default: 8192
            Batch size for evaluation.
        eval_user_num : int or None, default: None
            Number of users to sample for evaluation.
            By default, it will use all the users in eval_data.
            Setting it to a positive number will sample users randomly from eval data.
        verbose : int, default: 2
            Verbosity level.
        num_workers : int, default: 0
            Number of workers for data loading.

        Returns
        -------
        dict[str, float]
            Dictionary of metric names to values.
        """
        evaluator = Evaluator(
            during_training=False,
            data=data,
            model=self,
            is_multi_task=self.is_multi_task,
            neg_sampling=neg_sampling,
            metrics=metrics,
            k=k,
            eval_batch_size=eval_batch_size,
            sample_user_num=eval_user_num,
            seed=self.seed,
            verbose=verbose,
            num_workers=num_workers,
        )
        return evaluator.compute_eval_results()

    @override
    def training_step(self, batch: MutableMapping[str, torch.Tensor]) -> torch.Tensor:
        train_loss = self.compute_loss(batch)
        self.log("train_loss", train_loss, prog_bar=True)
        return train_loss

    @override
    def on_train_end(self):
        self.evaluator = None
        self.assign_embed_oovs()
        n_rec = min(2000, self.n_items)
        self.default_recs = self._rec_inner(
            user_ids=[OOV_IDX], n_rec=n_rec, filter_consumed=False
        )[0]

    @override
    def on_validation_epoch_start(self):
        if self.evaluator:
            self.evaluator.clear_state()

    @override
    def validation_step(
        self,
        batch: tuple[np.ndarray, np.ndarray] | np.ndarray,
        batch_idx: int,
        dataloader_idx: int
    ):
        if dataloader_idx == 0:
            self.evaluator.update_preds(batch)
        else:
            self.evaluator.update_recos(batch)

    # use `on_train_epoch_end` to display eval metrics after progress bar completion
    @override
    def on_train_epoch_end(self):
        if self.evaluator and self.evaluator.verbose >= 1:
            self.evaluator.print_metrics()

    @override
    def configure_optimizers(self) -> optim.Optimizer:
        return optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.wd)

    def compute_loss(self, batch: MutableMapping[str, torch.Tensor]) -> torch.Tensor:
        """Compute the loss for a batch of data."""
        outputs = self(batch)
        labels = batch.pop(LABEL_KEY)
        if self.task == "rating":
            loss = mse_loss(outputs, labels)
        elif self.loss == "focal":
            loss = focal_loss(outputs, labels)
        else:
            loss = binary_cross_entropy_loss(outputs, labels)
        return loss

    def get_batch_loader(
        self,
        data: BatchData,
        neg_sampling: bool,
        batch_size: int,
        shuffle: bool,
        num_workers: int,
    ) -> DataLoader:
        # TODO: consider `batch_size` when used in Sage models
        # if SageModels.contains(model.model_name) and model.paradigm == "i2i":
        #    batch_data.factor = model.num_walks * model.sample_walk_len
        sampler = RandomSampler(data) if shuffle else SequentialSampler(data)
        batch_size = self.adjust_batch_size(batch_size, neg_sampling)
        batch_sampler = BatchSampler(sampler, batch_size, drop_last=False)
        collate_fn = self.get_collate_fn(neg_sampling)
        return DataLoader(
            data,
            batch_size=None,  # `batch_size=None` disables automatic batching
            sampler=batch_sampler,
            collate_fn=collate_fn,
            num_workers=num_workers,
        )

    # consider negative sampling and random walks in batch_size
    def adjust_batch_size(self, original_batch_size: int, neg_sampling: bool) -> int:
        """Adjust batch size based on negative sampling strategy."""
        if not neg_sampling or self.loss in LISTWISE_LOSS:
            return original_batch_size
        elif self.loss in PAIRWISE_LOSS:
            return max(1, int(original_batch_size / self.num_neg))
        else:
            return max(1, int(original_batch_size / (self.num_neg + 1)))

    # TODO: refactor get_collate_fn
    def get_collate_fn(self, neg_sampling) -> Collator:
        # pairwise loss must use negative sampling
        if self.loss in PAIRWISE_LOSS:
            return PairwiseCollator(
                self.data_info,
                self.feat_info,
                self.sampler,
                self.num_neg,
                self.seq_params,
                self.is_multi_task,
                self.seed,
            )
        else:
            if self.loss in LISTWISE_LOSS:
                neg_sampling = False
            return Collator(
                self.data_info,
                self.feat_info,
                neg_sampling,
                self.sampler,
                self.num_neg,
                self.seq_params,
                self.is_multi_task,
                self.seed,
            )

    def get_seq_params(self, max_seq_len: int) -> SeqParams | DualSeqParams:
        if self.use_hash:
            seqs = get_recent_seq_dict(self.user_consumed, max_seq_len)
        else:
            seqs = get_recent_seqs(self.n_users, self.user_consumed, max_seq_len)
        return SeqParams(max_seq_len, seqs)

    def predict(
        self,
        user: UserId | list[UserId] | np.ndarray,
        item: ItemId | list[ItemId] | np.ndarray,
        cold_start: str = "average",
        inner_id: bool = False,
    ) -> list[float]:
        """Predict ratings or scores for user-item pairs.

        Parameters
        ----------
        user : :type:`~endrs.types.UserId` or list[UserId] or np.ndarray
            User id or batch of user ids.
        item : :type:`~endrs.types.ItemId` or list[ItemId] or np.ndarray
            Item id or batch of item ids.
        cold_start : {'popular', 'average'}, default: 'average'
            Strategy for handling cold-start users/items.
        inner_id : bool, default: False
            Whether the provided IDs are internal IDs.
            For library users inner_id may never be used.

        Returns
        -------
        list[float]
            Predicted scores for each user-item pair.
        """
        user, item = convert_ids(user, item, self.id_converter, inner_id)
        unknown_num, unknown_index = get_unknown(user, item)
        preds = self._pred_inner(user, item)
        return normalize_prediction(
            preds, self.task, cold_start, unknown_num, unknown_index
        )

    @torch.inference_mode()
    def _pred_inner(self, users: np.ndarray, items: np.ndarray) -> np.ndarray:
        self.eval()
        inputs = self.get_pred_inputs(users, items)
        if self.seq_params:
            inputs = self.build_seq(inputs, users)
        return self(inputs).cpu().numpy()

    def get_pred_inputs(
        self, users: np.ndarray, items: np.ndarray
    ) -> dict[str, torch.Tensor]:
        user_inputs = get_user_inputs(users, self.feat_info, self.device)
        item_inputs = get_item_inputs(items, self.feat_info, self.device)
        return {**user_inputs, **item_inputs}

    def recommend_user(
        self,
        user: UserId | list[UserId] | np.ndarray,
        n_rec: int,
        user_feats: Mapping[str, Any] | None = None,
        seq: list[ItemId] | None = None,
        cold_start: str = "average",
        inner_id: bool = False,
        filter_consumed: bool = True,
        random_rec: bool = False,
    ) -> dict[UserId, list[ItemId]]:
        """Recommend a list of items for given user(s).

        If both ``user_feats`` and ``seq`` are ``None``, the model will use the stored features
        for recommendation, and the ``cold_start`` strategy will be used for unknown users.

        If either ``user_feats`` or ``seq`` is provided, the model will use them for recommendation.
        In this case, if the ``user`` is unknown, it will be set to padding id, which means
        the ``cold_start`` strategy will not be applied.
        This situation is common when one wants to recommend for an unknown user based on
        user features or behavior sequence.

        Parameters
        ----------
        user : :type:`~endrs.types.UserId` or list[UserId] or np.ndarray
            User id or a batch of user ids to recommend.
        n_rec : int
            Number of recommendations to generate.
        user_feats : Mapping[str, Any] or None, default: None
            Additional user features for recommendation.
        seq : list[ItemId] or None, default: None
            Extra item sequence for recommendation. If the sequence length is larger than
            `recent_num` hyperparameter specified in the model, it will be truncated.
            If smaller, it will be padded.

        cold_start : {'popular', 'average'}, default: 'average'
            Strategy for handling cold-start users.

            - 'popular' will sample from popular items.
            - 'average' will use the average of all the user/item embeddings as the
              representation of the cold-start user/item.

        inner_id : bool, default: False
            Whether to use inner_id defined in `endrs`.
            For library users inner_id may never be used.
        filter_consumed : bool, default: True
            Whether to filter out items the user has already consumed.
        random_rec : bool, default: False
            Whether to add randomness to recommendations.

        Returns
        -------
        dict[UserId, list[ItemId]]
            Dictionary mapping user IDs to list of recommended item IDs.
        """
        self.check_dynamic_rec_feats(user, user_feats, seq)
        if n_rec > self.n_items:
            raise ValueError(f"`n_rec` {n_rec} exceeds num of items {self.n_items}")

        result_recs = dict()
        user_ids, unknown_users = sep_unknown_users(self.id_converter, user, inner_id)
        if unknown_users:
            cold_recs = cold_start_rec(
                self.data_info,
                self.default_recs,
                cold_start,
                unknown_users,
                n_rec,
                inner_id,
                self.np_rng,
            )
            result_recs.update(cold_recs)
        if user_ids:
            computed_recs = self._rec_inner(
                user_ids,
                n_rec,
                filter_consumed,
                user_feats,
                seq,
                inner_id,
                random_rec,
            )
            user_recs = construct_rec(
                self.id_converter, user_ids, computed_recs, inner_id
            )
            result_recs.update(user_recs)

        return result_recs

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
        self.eval()
        inputs = self.get_rec_inputs(
            user_ids,
            user_feats,
            seq,
            inner_seq_ids,
        )
        preds = self(inputs).cpu().numpy()
        return self.ranking_model.get_top_items(
            user_ids, preds, n_rec, filter_consumed, random_rec
        )

    def get_rec_inputs(
        self,
        users: list[int],
        user_feats: Mapping[str, Any] | None = None,
        seq: list[ItemId] | None = None,
        inner_seq_ids: bool = False,
    ) -> dict[str, torch.Tensor]:
        n_users = len(users)
        users = np.repeat(users, len(self.cand_items))
        items = np.tile(self.cand_items, n_users)
        inputs = self.get_pred_inputs(users, items)
        # if user_feats or seq is not None, only 1 user is allowed
        if self.feat_info and user_feats:
            inputs = self.feat_info.set_user_features(inputs, user_feats)
        if self.seq_params:
            inputs = self.build_seq(inputs, users, seq, inner_seq_ids)
        return inputs

    def build_seq(
        self,
        inputs: dict[str, torch.Tensor],
        users: np.ndarray,
        seq: list[ItemId] | None = None,
        inner_seq_ids: bool = False,
    ) -> dict[str, torch.Tensor]:
        if seq:
            if not inner_seq_ids:
                seq = [self.id_converter.safe_item_to_id(i) for i in seq]
            max_seq_len = self.seq_params.max_seq_len
            seq_len = min(max_seq_len, len(seq))
            input_seqs = np.full((len(users), max_seq_len), OOV_IDX)
            # shape: [1, max_seq_len] or [B, max_seq_len]
            input_seqs[:, :seq_len] = seq[-seq_len:]
        elif self.use_hash:
            input_seqs = np.array([self.seq_params.cached_seqs[u] for u in users])
        else:
            input_seqs = self.seq_params.cached_seqs[users]
        return get_seq_inputs(inputs, input_seqs, self.feat_info, self.device)

    @torch.inference_mode()
    def assign_embed_oovs(self):
        for module in self.modules():
            if isinstance(module, Embedding | UnionEmbedding):
                module.assign_oovs()

    def embed_output_dim(self, mode: str, add_embed: bool = True) -> int:
        """Calculate the output dimension of embeddings."""
        dim = 2 if mode == "all" else 1
        if self.feat_info:
            if mode == "user":
                dim += len(self.feat_info.user_feats)
            elif mode == "item":
                dim += len(self.feat_info.item_feats)
            else:
                dim += len(self.feat_info.all_feats)
        if add_embed:
            dim *= self.embed_size
        return dim

    def check_data_labels(self, data: BatchData, neg_sampling: bool):
        if self.is_multi_task:
            check_multi_labels(self.task, data.multi_labels, neg_sampling)
        else:
            check_labels(self.task, data.labels, neg_sampling)

    def check_dynamic_rec_feats(
        self,
        user: UserId | list[UserId],
        user_feats: Mapping[str, Any] | None,
        seq: list[ItemId] | None,
    ):
        if self.seq_params is None and seq is not None:
            raise ValueError(f"`{self.name}` doesn't support arbitrary seq inference.")
        if not np.isscalar(user):
            if user_feats is not None:
                raise ValueError(
                    f"Batch inference doesn't support arbitrary features: {user}"
                )
            if seq is not None:
                raise ValueError(
                    f"Batch inference doesn't support arbitrary item sequence: {user}"
                )
        if seq is not None and not isinstance(seq, list):
            raise ValueError("`seq` must be list.")
        if user_feats is not None and not isinstance(user_feats, dict):
            raise ValueError("`user_feats` must be `dict`.")
