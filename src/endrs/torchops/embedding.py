from collections.abc import Mapping, Sequence
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from endrs.feature.feat_info import FeatInfo
from endrs.utils.constants import ITEM_KEY, OOV_IDX, SEQ_KEY, USER_KEY


class UnionEmbedding(nn.Module):
    """Combines user and item embeddings into a unified representation.
    
    This class wraps both user and item embedding layers and concatenates their outputs,
    creating a combined embedding for joint user-item representation learning.

    Parameters
    ----------
    n_users : int
        Number of unique users.
    n_items : int
        Number of unique items.
    embed_size : int
        Dimension of the embedding vectors.
    multi_sparse_combiner : str
        Method for combining multiple sparse features.
        Options include 'sum', 'mean', or 'sqrtn'.
    feat_info : :class:`~endrs.feature.FeatInfo` or None, default: None
        Feature information object providing details about available features 
        and their dimensions. Default is None.
    """

    def __init__(
        self,
        n_users: int,
        n_items: int,
        embed_size: int,
        multi_sparse_combiner: str,
        feat_info: FeatInfo | None = None,
    ):
        super().__init__()
        self.user_embeds = Embedding(
            "user", n_users, embed_size, multi_sparse_combiner, feat_info
        )
        self.item_embeds = Embedding(
            "item", n_items, embed_size, multi_sparse_combiner, feat_info
        )

    # def forward(self, batch: Mapping[str, torch.Tensor], flatten: bool = True, **_):
    #     user_feats = self.user_embeds(batch, flatten=False)
    #     item_feats = self.item_embeds(batch, flatten=False)
    #     return torch.cat([user_feats, item_feats], dim=1)
    #     # (B or B*seq) * (F*E) or (B or B*seq) * F * E
    #     outputs = torch.cat([user_feats, item_feats], dim=-2)
    #     if flatten:
    #         outputs = outputs.flatten(start_dim=-2)
    #     return outputs

    def forward(self, batch: Mapping[str, torch.Tensor], flatten: bool = True, **_):
        """Generate combined embeddings for users and items in the batch.

        Parameters
        ----------
        batch : Mapping[str, torch.Tensor]
            Dictionary containing input tensors.
        flatten : bool, optional
            Whether to flatten feature dimension with embedding dimension.
            Default is True.
                
        Returns
        -------
        torch.Tensor
            Combined embeddings tensor with shape:
            - If flatten=True: (batch_size, (n_user_features+n_item_features)*embed_size)
            - If flatten=False: (batch_size, n_user_features+n_item_features, embed_size)
        """
        user_feats = self.user_embeds(batch, flatten)
        item_feats = self.item_embeds(batch, flatten)
        # B * (FE) or B * F * E
        return torch.cat([user_feats, item_feats], dim=1)

    def assign_oovs(self):
        self.user_embeds.assign_oovs()
        self.item_embeds.assign_oovs()


class Embedding(nn.Module):
    """Embedding layer for user/item IDs and their associated features.

    Supports embedding for:
    - ID fields (user_id or item_id)
    - Sparse categorical features
    - Multi-sparse categorical features (multiple values per instance)
    - Dense numerical features

    Parameters
    ----------
    mode : {'user', 'item'}
        Whether this embedding is for users or items.
    num : int
        Number of unique entities (users or items).
    embed_size : int
        Dimension of the embedding vectors.
    multi_sparse_combiner : str
        Method for combining multiple sparse features.
        Options include 'sum', 'mean', or 'sqrtn'.
    feat_info : :class:`~endrs.feature.FeatInfo` or None, default: None
        Feature information object providing details about available features 
        and their dimensions.
    """

    def __init__(
        self,
        mode: Literal["user", "item"],
        num: int,
        embed_size: int,
        multi_sparse_combiner: str,
        feat_info: FeatInfo | None = None,
    ):
        super().__init__()
        self.embed_size = embed_size
        self.multi_sparse_combiner = multi_sparse_combiner
        self.id_embeds = nn.Embedding(num + 1, embed_size, padding_idx=OOV_IDX)
        if mode.endswith("user"):
            self.id_key = USER_KEY
            self.sparse_cols = feat_info and feat_info.user_sparse_feats
            self.dense_cols = feat_info and feat_info.user_dense_feats
            self.multi_sparse_cols = feat_info and feat_info.user_multi_sparse_feats
        elif mode.endswith("item"):
            self.id_key = ITEM_KEY
            self.sparse_cols = feat_info and feat_info.item_sparse_feats
            self.dense_cols = feat_info and feat_info.item_dense_feats
            self.multi_sparse_cols = feat_info and feat_info.item_multi_sparse_feats
        else:
            raise ValueError("`mode` must be `user` or `item`")

        self.no_feature = not (
            self.sparse_cols or self.dense_cols or self.multi_sparse_cols
        )
        if self.sparse_cols:
            self.sparse_embed_dict = nn.ModuleDict(
                {
                    feat: nn.Embedding(
                        feat_info.sparse_size(feat) + 1,
                        embed_size,
                        padding_idx=OOV_IDX,
                    )
                    for feat in self.sparse_cols
                }
            )
        if self.dense_cols:
            self.dense_param = nn.Parameter(
                torch.empty(len(self.dense_cols), embed_size).float()
            )
        if self.multi_sparse_cols:
            mode = "sum" if multi_sparse_combiner in ("sum", "sqrtn") else "mean"
            self.multi_sparse_embed_dict = nn.ModuleDict(
                {
                    feat: nn.EmbeddingBag(
                        feat_info.sparse_size(feat) + 1,
                        embed_size,
                        mode=mode,
                        padding_idx=OOV_IDX,
                    )
                    for feat in self.multi_sparse_cols
                }
            )
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.id_embeds.weight)
        if self.sparse_cols:
            for embed in self.sparse_embed_dict.values():
                nn.init.xavier_uniform_(embed.weight)
        if self.dense_cols:
            nn.init.xavier_uniform_(self.dense_param)
        if self.multi_sparse_cols:
            for embed in self.multi_sparse_embed_dict.values():
                nn.init.xavier_uniform_(embed.weight)

    def forward(
        self,
        batch: Mapping[str, torch.Tensor],
        flatten: bool = True,
        is_seq: bool = False,
        **_,
    ) -> torch.Tensor:
        """Generate embeddings for entities (users or items) in the batch.

        Processes ID fields, sparse features, multi-sparse features, and dense features,
        combining them into a single representation.

        Parameters
        ----------
        batch : Mapping[str, torch.Tensor]
            Dictionary containing input tensors.
        flatten : bool, optional
            Whether to flatten feature dimension with embedding dimension.
            Default is True.
        is_seq : bool, optional
            Whether the input is a sequence. Default is False.
                
        Returns
        -------
        torch.Tensor
            Embeddings tensor with shape:
            - If flatten=True: (batch_size, n_features*embed_size)
            - If flatten=False: (batch_size, n_features, embed_size)
            
            For sequence inputs, batch_size is replaced with batch_size*seq_length.
        """
        key = SEQ_KEY if is_seq else self.id_key
        batch_size = batch[key].shape[0]
        id_embed = self.id_embeds(batch[key])
        if self.no_feature:
            # (B or B*seq) * E or (B or B*seq) * 1 * E
            return id_embed if flatten else id_embed.unsqueeze(dim=-2)

        outputs = [id_embed]
        if self.sparse_cols:
            sparse_output = self.compute_sparse(batch, is_seq)
            outputs.extend(sparse_output)
        if self.multi_sparse_cols:
            multi_sparse_output = self.compute_multi_sparse(batch, is_seq)
            outputs.extend(multi_sparse_output)
        # (B or B*seq) * (1 or F_sparse) * E
        outputs = [torch.stack(outputs, dim=-2)]

        if self.dense_cols:
            dense_output = self.compute_dense(batch, batch_size, is_seq)
            outputs.append(dense_output)

        outputs = torch.cat(outputs, dim=-2) if len(outputs) > 1 else outputs[0]
        if flatten:
            outputs = outputs.flatten(start_dim=-2)
        return outputs

    def compute_sparse(
        self, batch: Mapping[str, torch.Tensor], is_seq: bool = False
    ) -> list[torch.Tensor]:
        outputs = []
        for feat in self.sparse_cols:
            embed_module = self.sparse_embed_dict[feat]
            key = SEQ_KEY + feat if is_seq else feat
            out = embed_module(batch[key])
            outputs.append(out)
        return outputs

    def compute_multi_sparse(
        self, batch: Mapping[str, torch.Tensor], is_seq: bool = False
    ) -> list[torch.Tensor]:
        outputs = []
        for feat in self.multi_sparse_cols:
            embed_module = self.multi_sparse_embed_dict[feat]
            key = SEQ_KEY + feat if is_seq else feat
            # (B or B*seq) * F_multi_sparse
            multi_sparse_input = batch[key]
            if is_seq:
                assert multi_sparse_input.dim() == 3
                batch_size, seq_len, feat_num = multi_sparse_input.shape
                total_size = multi_sparse_input.numel()
                embed_bag_input = multi_sparse_input.view(-1)
                offsets = torch.arange(
                    0, total_size, feat_num, device=embed_bag_input.device
                )
                out = embed_module(embed_bag_input, offsets)
                out = out.view(batch_size, seq_len, self.embed_size)
            else:
                assert multi_sparse_input.dim() == 2
                out = embed_module(multi_sparse_input)

            if self.multi_sparse_combiner == "sqrtn":
                out *= self.multi_sparse_rsqrt_len(multi_sparse_input)
            outputs.append(out)
        return outputs

    @staticmethod
    def multi_sparse_rsqrt_len(multi_sparse_indices: torch.Tensor) -> torch.Tensor:
        """Calculate reciprocal square root of non-OOV indices length for scaling.

        Used for 'sqrtn' combiner, which normalizes by the square root of the
        number of non-OOV elements.
        """
        # noinspection PyUnresolvedReferences
        length = (multi_sparse_indices != OOV_IDX).float().sum(dim=-1, keepdim=True)
        length = torch.where(length == 0, 1.0, length)
        return torch.rsqrt(length)

    def compute_dense(
        self, batch: Mapping[str, torch.Tensor], batch_size: int, is_seq: bool = False
    ) -> torch.Tensor:
        # (B or B*seq) * F_dense * E
        dense_embeds = self.dense_param.repeat(batch_size, 1, 1)
        if is_seq:
            seq_len = batch[SEQ_KEY].shape[1]
            dense_embeds = dense_embeds.unsqueeze(1).repeat(1, seq_len, 1, 1)

        dense_vals = []
        for feat in self.dense_cols:
            key = SEQ_KEY + feat if is_seq else feat
            dense_vals.append(batch[key])
        # (B or B*seq) * F_dense * 1
        dense_vals = torch.stack(dense_vals, dim=-1).unsqueeze(dim=-1)
        return dense_embeds * dense_vals

    def assign_oovs(self):
        self.id_embeds.weight[OOV_IDX] = _get_mean_embed(self.id_embeds)
        if self.sparse_cols:
            for embed in self.sparse_embed_dict.values():
                embed.weight[OOV_IDX] = _get_mean_embed(embed)
        if self.multi_sparse_cols:
            for embed in self.multi_sparse_embed_dict.values():
                embed.weight[OOV_IDX] = _get_mean_embed(embed)


class HashEmbedding(nn.Module):
    """Embedding layer that uses hash-based approach to map features to embeddings.

    Instead of maintaining separate embedding tables for each feature, uses a single
    shared embedding table accessed via hash functions. Only sparse features are used.

    Parameters
    ----------
    n_bins : int
        Number of hash bins (size of the shared embedding table).
    embed_size : int
        Dimension of the embedding vectors.
    feat_info : :class:`~endrs.feature.FeatInfo` or None, default: None
        Feature information object providing details about available features.
        Default is None.
    """

    def __init__(self, n_bins: int, embed_size: int, feat_info: FeatInfo | None):
        super().__init__()
        if feat_info:
            self.user_sparse_cols = feat_info.user_sparse_feats
            self.item_sparse_cols = feat_info.item_sparse_feats
        else:
            self.user_sparse_cols = self.item_sparse_cols = []
        self.hash_embeds = nn.Embedding(n_bins + 1, embed_size, padding_idx=OOV_IDX)
        nn.init.xavier_uniform_(self.hash_embeds.weight)

    def forward(
        self,
        batch: Mapping[str, torch.Tensor],
        flatten: bool = True,
        is_user: bool = False,
        is_item: bool = False,
        is_seq: bool = False,
    ) -> torch.Tensor:
        """Generate embeddings using hash-based approach.

        Parameters
        ----------
        batch : Mapping[str, torch.Tensor]
            Dictionary containing input tensors.
        flatten : bool, optional
            Whether to flatten feature dimension with embedding dimension.
            Default is True.
        is_user : bool, optional
            Whether to process user features. Default is False.
        is_item : bool, optional
            Whether to process item features. Default is False.
        is_seq : bool, optional
            Whether the input is a sequence. Default is False.

        Returns
        -------
        torch.Tensor
            Embeddings tensor with shape:
            - If flatten=True: (batch_size, n_features*embed_size)
            - If flatten=False: (batch_size, n_features, embed_size)

            For sequence inputs, batch_size is replaced with batch_size*seq_length.

        Raises
        ------
        AssertionError
            If none of is_user, is_item, or is_seq is True.
        """
        assert any([is_user, is_item, is_seq])
        # (B or B*seq) * F * E
        if is_user and is_item:
            assert not is_seq  # no seq in union embedding
            user_embeds = self.compute_embedding(batch, USER_KEY, self.user_sparse_cols)
            item_embeds = self.compute_embedding(batch, ITEM_KEY, self.item_sparse_cols)
            outputs = torch.cat([user_embeds, item_embeds], dim=1)
        elif is_user:
            outputs = self.compute_embedding(batch, USER_KEY, self.user_sparse_cols)
        elif is_item:
            outputs = self.compute_embedding(batch, ITEM_KEY, self.item_sparse_cols)
        else:
            outputs = self.compute_embedding(
                batch, SEQ_KEY, self.item_sparse_cols, is_seq=True
            )

        if flatten:
            outputs = outputs.flatten(start_dim=-2)
        return outputs

    def compute_embedding(
        self,
        batch: Mapping[str, torch.Tensor],
        main_key: str,
        sparse_cols: Sequence[str],
        is_seq: bool = False,
    ) -> torch.Tensor:
        outputs = [self.hash_embeds(batch[main_key])]
        for feat in sparse_cols:
            key = SEQ_KEY + feat if is_seq else feat
            out = self.hash_embeds(batch[key])
            outputs.append(out)
        # (B or B*seq) * (1 or F_sparse) * E
        return torch.stack(outputs, dim=-2)

    def assign_oovs(self):
        ...


def _get_mean_embed(embeddings: nn.Embedding) -> torch.Tensor:
    """Calculate the mean of all embeddings except the OOV embedding."""
    rest_indices = list(range(embeddings.num_embeddings))
    rest_indices.remove(OOV_IDX)
    return torch.mean(embeddings.weight[rest_indices], dim=0)


def normalize_embeds(*embeds: torch.Tensor | np.ndarray, backend: str = "torch"):
    """Normalize embeddings to unit length (L2 norm)."""
    normed_embeds = []
    for e in embeds:
        if backend.endswith("torch"):
            ne = F.normalize(e, p=2, dim=1)
        else:
            norms = np.linalg.norm(e, axis=1, keepdims=True)
            ne = e / norms
        normed_embeds.append(ne)
    return normed_embeds[0] if len(embeds) == 1 else normed_embeds
