from collections.abc import Mapping, Sequence

import torch
from torch import nn

from endrs.bases.dyn_embed_base import DynEmbedBase
from endrs.data.data_info import DataInfo
from endrs.feature.feat_info import FeatInfo
from endrs.torchops.embedding import Embedding, HashEmbedding, normalize_embeds
from endrs.torchops.layers import ConvLayer


class Caser(DynEmbedBase):
    """Convolutional Sequence Embedding Recommendation Model (Caser).

    A convolutional neural network based recommendation model that captures 
    sequential patterns in user-item interactions through horizontal and vertical 
    convolutional operations on the embedding matrix of a user's sequence history.

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
    use_bn : bool, default: False
        Whether to use batch normalization in convolutional layers.
    nh_channels : int, default: 2
        Number of channels (filters) for horizontal convolutional filters.
    nv_channels : int, default: 4
        Number of channels (filters) for vertical convolutional filters.
    max_seq_len : int, default: 10
        Maximum sequence length for user interaction history.
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
    *Jiaxi Tang & Ke Wang.* `Personalized Top-N Sequential Recommendation via Convolutional Sequence Embedding
    <https://arxiv.org/pdf/1809.07426.pdf>`_.
    """

    def __init__(
        self,
        task: str,
        data_info: DataInfo,
        feat_info: FeatInfo | None = None,
        loss: str = "cross_entropy",
        embed_size: int = 16,
        norm_embed: bool = False,
        n_epochs: int = 20,
        lr: float = 0.001,
        weight_decay: float = 0.0,
        batch_size: int = 256,
        sampler: str = "random",
        num_neg: int = 1,
        use_bn: bool = False,
        nh_channels: int = 2,
        nv_channels: int = 4,
        max_seq_len: int = 10,
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
        self.nh_channels = nh_channels
        self.nv_channels = nv_channels
        self.max_seq_len = max_seq_len
        self.user_embeds_dim = self.embed_output_dim("user")
        self.item_embeds_dim = self.embed_output_dim("item")
        self.seq_params = self.get_seq_params(max_seq_len)
        if self.use_hash:
            n_bins = data_info.n_hash_bins
            self.hash_embed_layer = HashEmbedding(n_bins, self.embed_size, feat_info)
        else:
            self.user_embed_layer = Embedding(
                "user",
                self.n_users,
                self.embed_size,
                multi_sparse_combiner,
                self.feat_info,
            )
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

        self.h_cnn_layers = nn.ModuleList(
            [
                # channels_last transposed input shape: B * C * seq_len
                ConvLayer(
                    in_channels=self.item_embeds_dim,
                    out_channels=self.nh_channels,
                    kernel_size=i,
                    use_max_pool=True,
                    use_bn=use_bn,
                    channels_last=True,
                )
                for i in range(1, max_seq_len + 1)
            ]
        )
        # input shape: B * seq_len * C
        self.v_cnn_layer = ConvLayer(
            in_channels=self.max_seq_len,
            out_channels=self.nv_channels,
            kernel_size=1,
            use_max_pool=False,
            use_bn=use_bn,
            channels_last=False,
        )
        self.user_output_layer = nn.Linear(self.final_input_dim(), self.embed_size)
        self.item_output_layer = nn.Linear(self.item_embeds_dim, self.embed_size)
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.user_output_layer.weight)
        nn.init.zeros_(self.user_output_layer.bias)
        nn.init.xavier_uniform_(self.item_output_layer.weight)
        nn.init.zeros_(self.item_output_layer.bias)

    def get_user_embeddings(self, inputs: Mapping[str, torch.Tensor]) -> torch.Tensor:
        if self.use_hash:
            user_embeds = self.hash_embed_layer(inputs, is_user=True)
            seq_embeds = self.hash_embed_layer(inputs, is_seq=True)
        else:
            user_embeds = self.user_embed_layer(inputs)
            seq_embeds = self.seq_embed_layer(inputs, is_seq=True)
        out = [user_embeds]
        for cnn_layer in self.h_cnn_layers:
            h_conv_out = cnn_layer(seq_embeds)
            out.append(h_conv_out)

        out.append(self.v_cnn_layer(seq_embeds))
        user_embeds = self.user_output_layer(torch.cat(out, dim=1))
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

    def final_input_dim(self):
        h_cnn_dim = self.max_seq_len * self.nh_channels
        v_cnn_dim = self.nv_channels * ConvLayer.output_len(
            input_len=self.item_embeds_dim, kernel_size=1, stride=1
        )
        return self.user_embeds_dim + h_cnn_dim + v_cnn_dim
