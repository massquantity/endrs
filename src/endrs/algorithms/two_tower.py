from collections.abc import Mapping, Sequence

import torch
from torch import nn

from endrs.bases.dyn_embed_base import DynEmbedBase
from endrs.data.data_info import DataInfo
from endrs.feature.feat_info import FeatInfo
from endrs.torchops.dnn import DNN
from endrs.torchops.embedding import Embedding, HashEmbedding, normalize_embeds


# TODO: self-supervised Learning
class TwoTower(DynEmbedBase):
    """TwoTower Recommendation model.

    Parameters
    ----------
    task : {'rating', 'ranking'}
        The recommendation task type. See :ref:`Task`.
    data_info : :class:`~endrs.data.DataInfo`
        Object that contains useful information about the dataset.
    feat_info : :class:`~endrs.feature.FeatInfo` or None, default: None
        Object that contains information about features used in the model.
    loss : str, default: 'softmax'
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
    [1] *Xinyang Yi et al.* `Sampling-Bias-Corrected Neural Modeling for Large Corpus Item Recommendations
    <https://storage.googleapis.com/gweb-research2023-media/pubtools/5716.pdf>`_.

    [2] *Tiansheng Yao et al.* `Self-supervised Learning for Large-scale Item Recommendations
    <https://arxiv.org/pdf/2007.12865.pdf>`_.
    """

    def __init__(
        self,
        task: str,
        data_info: DataInfo,
        feat_info: FeatInfo | None = None,
        loss: str = "softmax",
        embed_size: int = 16,
        norm_embed: bool = False,
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
            self.item_embed_layer = Embedding(
                "item",
                self.n_items,
                self.embed_size,
                multi_sparse_combiner,
                self.feat_info,
            )
        self.user_dnn_layer = DNN(
            self.embed_output_dim("user"), hidden_units, use_bn, dropout_rate
        )
        self.item_dnn_layer = DNN(
            self.embed_output_dim("item"), hidden_units, use_bn, dropout_rate
        )
        self.user_output_layer = nn.Linear(hidden_units[-1], self.embed_size)
        self.item_output_layer = nn.Linear(hidden_units[-1], self.embed_size)
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.user_output_layer.weight)
        nn.init.zeros_(self.user_output_layer.bias)
        nn.init.xavier_uniform_(self.item_output_layer.weight)
        nn.init.zeros_(self.item_output_layer.bias)

    def get_user_embeddings(self, inputs: Mapping[str, torch.Tensor]) -> torch.Tensor:
        if self.use_hash:
            user_embeds = self.hash_embed_layer(inputs, is_user=True)
        else:
            user_embeds = self.user_embed_layer(inputs)
        user_embeds = self.user_dnn_layer(user_embeds)
        user_embeds = self.user_output_layer(user_embeds)
        if self.norm_embed:
            user_embeds = normalize_embeds(user_embeds)
        return user_embeds

    def get_item_embeddings(self, inputs: Mapping[str, torch.Tensor]) -> torch.Tensor:
        if self.use_hash:
            item_embeds = self.hash_embed_layer(inputs, is_item=True)
        else:
            item_embeds = self.item_embed_layer(inputs)
        item_embeds = self.item_dnn_layer(item_embeds)
        item_embeds = self.item_output_layer(item_embeds)
        if self.norm_embed:
            item_embeds = normalize_embeds(item_embeds)
        return item_embeds

    def forward(self, inputs: Mapping[str, torch.Tensor]) -> torch.Tensor:
        user_embeds = self.get_user_embeddings(inputs)
        item_embeds = self.get_item_embeddings(inputs)
        return torch.sum(user_embeds * item_embeds, dim=1)
