from typing import Sequence

import torch
import torch.nn.functional as F
from torch import nn


class DNN(nn.Module):
    """Deep Neural Network module.

    Parameters
    ----------
    input_dim : int
        Input dimension size.
    hidden_units : Sequence[int]
        Sizes of hidden layers.
    use_bn : bool, default: True
        Whether to use batch normalization.
    dropout_rate : float, default: 0.0
        Probability of an element to be zeroed. If it is 0.0, dropout is not used.
    bn_after_activation : bool, default: True
        Whether to apply batch normalization after activation function.
    activation : str, default: "relu"
        Activation function to use. Currently only supports 'relu'.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_units: Sequence[int],
        use_bn: bool = True,
        dropout_rate: float = 0.0,
        bn_after_activation: bool = True,
        activation: str = "relu",
    ):
        super().__init__()
        all_dims = (input_dim, *hidden_units)
        self.layers = nn.ModuleList(
            [
                nn.Linear(in_dim, out_dim)
                for in_dim, out_dim in zip(all_dims, all_dims[1:])
            ]
        )
        self.activation = F.relu if activation == "relu" else None
        self.use_bn = use_bn
        self.bn_after_activation = bn_after_activation
        if self.use_bn:
            self.bn = nn.ModuleList([nn.BatchNorm1d(dim) for dim in all_dims])  # all_dims[1:-1]
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0.0 else None
        self.init_weights()

    def init_weights(self):
        for layer in self.layers:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor):
        # if self.use_bn:
        #     x = self.bn[0](x)
        for i, layer in enumerate(self.layers, start=1):
            x = layer(x)
            if i < len(self.layers):  # Skip for last layer
                if self.use_bn and not self.bn_after_activation:
                    x = self.bn[i](x)
                if self.activation is not None:
                    x = self.activation(x)
                if self.use_bn and self.bn_after_activation:
                    x = self.bn[i](x)
                if self.dropout is not None:
                    x = self.dropout(x)

        return x
