from collections.abc import Mapping, Sequence

import torch
from torch import nn

from endrs.bases.torch_base import TorchBase
from endrs.data.data_info import DataInfo
from endrs.feature.feat_info import FeatInfo
from endrs.torchops.dnn import DNN
from endrs.torchops.embedding import HashEmbedding, UnionEmbedding
from endrs.torchops.layers import fm as fm_layer


class DeepFM(TorchBase):
    """DeepFM Recommendation model.

    Parameters
    ----------
    task : {'rating', 'ranking'}
        The recommendation task type. See :ref:`Task`.
    data_info : :class:`~endrs.data.DataInfo`
        Object that contains useful information about the dataset.
    feat_info : :class:`~endrs.feature.FeatInfo` or None, default: None
        Object that contains information about features used in the model.
    loss : str, default: 'cross_entropy'
        The loss to use for training (e.g., 'bce', 'focal', etc.).
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
        Sequence of hidden layer sizes for the deep component.
    multi_sparse_combiner : str, default: 'sqrtn'
        Method to combine multiple sparse features.
    seed : int, default: 42
        Random seed for reproducibility.
    accelerator : str, default: 'auto'
        Hardware accelerator type for training (e.g., 'cpu', 'gpu', 'auto', etc.).
    devices : Sequence[int] or str or int, default: 'auto'
        Devices to use for training.

    References
    ----------
    *Huifeng Guo et al.* `DeepFM: A Factorization-Machine based Neural Network for CTR Prediction
    <https://arxiv.org/pdf/1703.04247.pdf>`_.
    """

    def __init__(
        self,
        task: str,
        data_info: DataInfo,
        feat_info: FeatInfo | None = None,
        loss: str = "cross_entropy",
        embed_size: int = 16,
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
        self.hidden_units = hidden_units
        self.all_embeds_dim_lin = self.embed_output_dim("all", add_embed=False)
        self.all_embeds_dim = self.embed_output_dim("all")
        if self.use_hash:
            n_bins = data_info.n_hash_bins
            self.linear_embed_layer = HashEmbedding(n_bins, 1, feat_info)
            self.embed_layer = HashEmbedding(n_bins, self.embed_size, feat_info)
        else:
            self.linear_embed_layer = UnionEmbedding(
                self.n_users,
                self.n_items,
                embed_size=1,
                multi_sparse_combiner=multi_sparse_combiner,
                feat_info=feat_info,
            )
            self.embed_layer = UnionEmbedding(
                self.n_users,
                self.n_items,
                self.embed_size,
                multi_sparse_combiner,
                feat_info,
            )
        self.linear_layer = nn.Linear(self.all_embeds_dim_lin, 1)
        self.dnn_layer = DNN(self.all_embeds_dim, hidden_units, use_bn, dropout_rate)
        self.output_layer = nn.Linear(self.final_input_dim(), 1)
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.linear_layer.weight)
        nn.init.zeros_(self.linear_layer.bias)
        nn.init.xavier_uniform_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)

    def forward(self, inputs: Mapping[str, torch.Tensor]) -> torch.Tensor:
        linear_emb = self.linear_embed_layer(inputs, is_user=True, is_item=True)
        linear_term = self.linear_layer(linear_emb)
        pairwise_emb = self.embed_layer(
            inputs, flatten=False, is_user=True, is_item=True
        )
        pairwise_term = fm_layer(pairwise_emb)
        deep_emb = pairwise_emb.flatten(start_dim=1)
        deep_term = self.dnn_layer(deep_emb)
        output = torch.cat([linear_term, pairwise_term, deep_term], dim=1)
        # only squeeze dim 1 since the first dimension will also be removed if batch_size=1
        return self.output_layer(output).squeeze(dim=1)

    def final_input_dim(self):
        linear_dim = 1
        pair_dim = self.embed_size
        deep_dim = self.hidden_units[-1]
        return linear_dim + pair_dim + deep_dim
