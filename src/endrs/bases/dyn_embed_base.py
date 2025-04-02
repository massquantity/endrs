import abc
from collections.abc import Mapping, Sequence
from typing import Any, Literal, override

import numpy as np
import torch

from endrs.bases.torch_embed_base import TorchEmbedBase
from endrs.data.data_info import DataInfo
from endrs.feature.feat_info import FeatInfo
from endrs.inference.preprocess import get_user_inputs
from endrs.torchops.embedding import Embedding
from endrs.types import ItemId, UserId
from endrs.utils.constants import USER_KEY


class DynEmbedBase(TorchEmbedBase):
    """Base class for recommender models supporting dynamic embeddings and inference.

    This class extends TorchEmbedBase to provide functionality for generating embeddings
    and recommendations in dynamic scenarios, where user features or behavior sequences
    may be provided at inference time rather than relying solely on pre-computed embeddings.

    Parameters
    ----------
    task : {'rating', 'ranking'}
        The recommendation task type.
    data_info : :class:`~endrs.data.DataInfo`
        Object that contains useful information about the dataset.
    feat_info : :class:`~endrs.feature.FeatInfo` or None
        Object that contains information about features used in the model.
    loss : str
        The loss to use for training (e.g., 'bce', 'focal', etc.).
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
        The recommendation paradigm, either user-to-item or item-to-item.

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

    See Also
    --------
    TorchEmbedBase : Base class for embedding-based recommender models.
    """

    def __init__(
        self,
        task: str,
        data_info: DataInfo,
        feat_info: FeatInfo | None,
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
            norm_embed,
            n_epochs,
            lr,
            weight_decay,
            batch_size,
            sampler,
            num_neg,
            use_correction,
            temperature,
            remove_accidental_hits,
            seed,
            accelerator,
            devices,
            paradigm,
        )

    def convert_array_id(self, user: UserId, inner_id: bool) -> np.ndarray:
        """Convert a single user to inner user id.

        If the user doesn't exist, it will be converted to padding id.
        The return type should be `array_like` for further shape compatibility.
        """
        assert np.isscalar(user), f"User to convert must be scalar, got: {user}"
        if inner_id:
            if not isinstance(user, int | np.integer):
                raise ValueError(f"`inner id` user must be int, got {user}")
            return np.array([user])
        else:
            return np.array([self.id_converter.safe_user_to_id(user)])

    @torch.inference_mode()
    def dyn_user_embedding(
        self,
        user: UserId,
        user_feats: Mapping[str, Any] | None = None,
        seq: list[ItemId] | None = None,
        inner_id: bool = False,
        to_numpy: bool = True,
    ) -> np.ndarray | torch.Tensor:
        """Generate a dynamic user embedding based on user ID and optional features or sequence.

        This method allows generating user embeddings on-the-fly, which is useful for
        recommending to users with new features or behavior patterns that weren't
        available during training.

        Parameters
        ----------
        user : :type:`~endrs.types.UserId`
            The user ID to generate embeddings for.
        user_feats : Mapping[str, Any] or None, default: None
            Additional user features to incorporate into the embedding.
        seq : list[ItemId] or None, default: None
            Item sequence for sequential models. If provided, it will be used
            instead of the pre-computed sequences from training data.
        inner_id : bool, default: False
            Whether the provided user ID is an internal ID.
            For library users inner_id may never be used.
        to_numpy : bool, default: True
            Whether to convert the output tensor to a numpy array.

        Returns
        -------
        np.ndarray or torch.Tensor
            The generated user embedding, either as a numpy array or PyTorch tensor.
        """
        self.eval()
        self.check_dynamic_rec_feats(user, user_feats, seq)
        user_id = self.convert_array_id(user, inner_id)
        inputs = get_user_inputs(user_id, self.feat_info, self.device)
        if self.feat_info and user_feats:
            inputs = self.feat_info.set_user_features(inputs, user_feats)
        if self.seq_params:
            inputs = self.build_seq(inputs, user_id, seq, inner_id)

        embedding = self.get_user_embeddings(inputs)
        if to_numpy:
            embedding = embedding.cpu().numpy().squeeze(axis=0)
        return embedding

    @override
    @torch.inference_mode()
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
        """Recommend items for a user, optionally using dynamic features or sequences.

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
            User ID or batch of user IDs to recommend for.
        n_rec : int
            Number of recommendations to generate.
        user_feats : Mapping[str, Any] or None, default: None
            Additional user features for recommendation.
        seq : list[ItemId] or None, default: None
            Extra item sequence for recommendation in sequential models.

        cold_start : {'popular', 'average'}, default: 'average'
            Strategy for handling cold-start users.

            - 'popular' will sample from popular items.
            - 'average' will use the average of all the user/item embeddings as the
              representation of the cold-start user/item.

        inner_id : bool, default: False
            Whether to use internal IDs for users and items.
        filter_consumed : bool, default: True
            Whether to filter out items the user has already consumed.
        random_rec : bool, default: False
            Whether to add randomness to recommendations.

        Returns
        -------
        dict[UserId, list[ItemId]]
            Dictionary mapping user IDs to lists of recommended item IDs.
        """
        if user_feats is None and seq is None:
            return super().recommend_user(
                user,
                n_rec,
                user_feats,
                seq,
                cold_start,
                inner_id,
                filter_consumed,
                random_rec,
            )

        user_embed = self.dyn_user_embedding(
            user, user_feats, seq, inner_id, to_numpy=False
        )
        if self.use_hash:
            item_embeds = self.hash_embeds_tensor[self.cand_items]
        else:
            item_embeds = self.item_embeds_tensor[self.cand_items]
        preds = (user_embed @ item_embeds.t()).cpu().numpy()
        inner_user_id = self.convert_array_id(user, inner_id).tolist()
        computed_recs = self.ranking_model.get_top_items(
            inner_user_id, preds, n_rec, filter_consumed, random_rec
        )
        rec_items = (
            computed_recs[0]
            if inner_id
            else [self.id_converter.id2item[i] for i in computed_recs[0]]
        )
        # only one user is allowed in dynamic situation
        return {user: rec_items}

    @override
    @torch.inference_mode()
    def assign_embed_oovs(self):
        """Assign out-of-vocabulary embeddings for all embedding layers.
        
        This method extends the parent implementation to specifically handle user
        embeddings, ensuring that OOV embeddings are properly assigned for dynamic
        user representations.
        """
        super().assign_embed_oovs()
        for module in self.modules():
            if isinstance(module, Embedding) and module.id_key == USER_KEY:
                module.assign_oovs()

    @abc.abstractmethod
    def get_user_embeddings(self, inputs: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """Compute user embeddings from inputs."""

    @abc.abstractmethod
    def get_item_embeddings(self, inputs: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """Compute item embeddings from inputs."""
