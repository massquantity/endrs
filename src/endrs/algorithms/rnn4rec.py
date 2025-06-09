from collections.abc import Mapping, Sequence
from typing import Literal

import torch
from torch import nn
from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence, pad_packed_sequence

from endrs.bases.dyn_embed_base import DynEmbedBase
from endrs.data.data_info import DataInfo
from endrs.feature.feat_info import FeatInfo
from endrs.torchops.embedding import Embedding, HashEmbedding, normalize_embeds
from endrs.utils.constants import OOV_IDX, SEQ_KEY


class RNN4Rec(DynEmbedBase):
    """Recurrent Neural Network for Recommendation (RNN4Rec).

    .. NOTE::
        The original paper used GRU, but in this implementation we can also use LSTM.

    Parameters
    ----------
    task : {'rating', 'ranking'}
        The recommendation task type. See :ref:`Task`.
    data_info : :class:`~endrs.data.DataInfo`
        Object that contains useful information about the dataset.
    feat_info : :class:`~endrs.feature.FeatInfo` or None, default: None
        Object that contains information about features used in the model.
    loss : str, default: 'cross_entropy'
        The loss to use for training (e.g., 'bce', 'focal', 'softmax', etc.).
    rnn_type : {'lstm', 'gru'}, default: 'gru'
        Type of recurrent neural network cell to use for sequence modeling.
    embed_size : int, default: 16
        Size of the embedding vectors.
    norm_embed : bool, default: False
        Whether to normalize embeddings.
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
    dropout_rate : float, default: 0.0
        Dropout rate applied to RNN layers to prevent overfitting.
    max_seq_len : int, default: 10
        Maximum sequence length for user interaction history.
    num_layers : int, default: 1
        Number of stacked RNN layers.
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
    *Balázs Hidasi et al.* `Session-based Recommendations with Recurrent Neural Networks
    <https://arxiv.org/pdf/1511.06939.pdf>`_.
    """

    def __init__(
        self,
        task: str,
        data_info: DataInfo,
        feat_info: FeatInfo | None = None,
        loss: str = "cross_entropy",
        rnn_type: Literal["lstm", "gru"] = "gru",
        embed_size: int = 16,
        norm_embed: bool = False,
        n_epochs: int = 20,
        lr: float = 0.001,
        weight_decay: float = 0.0,
        batch_size: int = 256,
        sampler: str = "random",
        num_neg: int = 1,
        dropout_rate: float = 0.0,
        max_seq_len: int = 10,
        num_layers: int = 1,
        multi_sparse_combiner: str = "sqrtn",
        use_correction: bool = True,
        temperature: float = 1.0,
        remove_accidental_hits: bool = False,
        seed: int = 42,
        accelerator: str = "auto",
        devices: Sequence[int] | str | int = "auto",
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
        )
        self.check_params()
        if rnn_type not in ("lstm", "gru"):
            raise ValueError("`rnn_type` must either be `lstm` or `gru`")
        self.item_embeds_dim = self.embed_output_dim("item")
        self.seq_params = self.get_seq_params(max_seq_len)
        if self.use_hash:
            n_bins = data_info.n_hash_bins
            self.hash_embed_layer = HashEmbedding(n_bins, self.embed_size, feat_info)
        else:
            self.seq_embed_layer = Embedding(
                "item",
                self.n_items,
                self.embed_size,
                multi_sparse_combiner,
                self.feat_info,
            )
            self.item_embed_layer = Embedding(
                "item",
                self.n_items,
                self.embed_size,
                multi_sparse_combiner,
                self.feat_info,
            )

        rnn_cls = nn.LSTM if rnn_type == "lstm" else nn.GRU
        self.rnn_layer = rnn_cls(
            self.item_embeds_dim,
            self.embed_size,
            num_layers,
            batch_first=True,
            dropout=dropout_rate,
        )
        self.user_output_layer = nn.Linear(self.embed_size, self.embed_size)
        self.item_output_layer = nn.Linear(self.item_embeds_dim, self.embed_size)
        self.init_weights()

    def init_weights(self):
        for name, param in self.rnn_layer.named_parameters():
            if "weight" in name:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
        nn.init.xavier_uniform_(self.user_output_layer.weight)
        nn.init.zeros_(self.user_output_layer.bias)
        nn.init.xavier_uniform_(self.item_output_layer.weight)
        nn.init.zeros_(self.item_output_layer.bias)

    def get_packed_seq(self, inputs: Mapping[str, torch.Tensor]) -> PackedSequence:
        if self.use_hash:
            seq_embeds = self.hash_embed_layer(inputs, is_seq=True)
        else:
            seq_embeds = self.seq_embed_layer(inputs, is_seq=True)
        # noinspection PyUnresolvedReferences
        lengths = (inputs[SEQ_KEY] != OOV_IDX).int().sum(dim=1).detach().cpu()
        lengths = torch.where(lengths != 0, lengths, 1)
        return pack_padded_sequence(
            seq_embeds, lengths, batch_first=True, enforce_sorted=False
        )

    def get_user_embeddings(self, inputs: Mapping[str, torch.Tensor]) -> torch.Tensor:
        packed_seq_embeds = self.get_packed_seq(inputs)
        packed_rnn_output, _ = self.rnn_layer(packed_seq_embeds)
        rnn_output, _ = pad_packed_sequence(packed_rnn_output, batch_first=True)
        user_embeds = self.user_output_layer(rnn_output[:, -1, :])
        if self.norm_embed:
            user_embeds = normalize_embeds(user_embeds)
        return user_embeds

    def get_item_embeddings(self, inputs: Mapping[str, torch.Tensor]) -> torch.Tensor:
        if self.use_hash:
            item_embeds = self.hash_embed_layer(inputs, is_item=True)
        else:
            item_embeds = self.item_embed_layer(inputs)
        item_embeds = self.item_output_layer(item_embeds)
        if self.norm_embed:
            item_embeds = normalize_embeds(item_embeds)
        return item_embeds

    def forward(self, inputs: Mapping[str, torch.Tensor]) -> torch.Tensor:
        user_embeds = self.get_user_embeddings(inputs)
        item_embeds = self.get_item_embeddings(inputs)
        return torch.sum(user_embeds * item_embeds, dim=1)
