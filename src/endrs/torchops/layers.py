import torch
import torch.nn.functional as F
from torch import nn


# B * F * E -> B * E
def fm(inputs: torch.Tensor) -> torch.Tensor:
    """Factorization Machine part.

    Parameters
    ----------
    inputs : torch.Tensor
        Input tensor with shape (batch_size, field_size, embedding_size).

    Returns
    -------
    torch.Tensor
        Factorization Machine interaction result with shape (batch_size, embedding_size).

    References
    ----------
    *Steffen Rendle.* `Factorization Machines 
    <https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf>`_.
    """
    square_of_sum = inputs.sum(dim=1).square()
    sum_of_square = inputs.square().sum(dim=1)
    return 0.5 * (square_of_sum - sum_of_square)


class ConvLayer(nn.Module):
    """1D Convolutional layer with optional batch normalization and max pooling.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : int
        Size of the convolutional kernel.
    use_max_pool : bool
        Whether to apply max pooling after convolution.
    use_bn : bool
        Whether to use batch normalization.
    channels_last : bool
        Whether input has channels as the last dimension (batch_size, seq_len, channels).
        If False, expects input in the format (batch_size, channels, seq_len).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        use_max_pool: bool,
        use_bn: bool,
        channels_last: bool,
    ):
        super().__init__()
        # `channels_last` means inputs and outputs shape are (B * seq * C)
        # CNN and BN inputs shape are (B * C * seq)
        self.use_max_pool = use_max_pool
        self.use_bn = use_bn
        self.channels_last = channels_last
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size, stride=1, padding="valid"
        )
        if self.use_bn:
            self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        nn.init.kaiming_normal_(self.conv.weight, mode="fan_out", nonlinearity="relu")
        nn.init.zeros_(self.conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the convolutional layer.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor with shape (batch_size, seq_len, channels) if channels_last is True,
            otherwise (batch_size, channels, seq_len).

        Returns
        -------
        torch.Tensor
        """
        if self.channels_last:
            x = torch.transpose(x, 1, 2)
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        x = self.relu(x)
        if self.use_max_pool:
            kernel_size = x.shape[-1]
            x = F.max_pool1d(x, kernel_size, stride=1)

        if self.channels_last:
            # horizontal CNN with max_pool final shape: (B * C * 1) -> (B * C)
            x = torch.squeeze(x, dim=2)
        else:
            # vertical CNN final shape: (B * C * out_seq_len) -> (B * (out_seq_len*C)
            x = torch.flatten(x, start_dim=1)
        return x

    @staticmethod
    def output_len(input_len: int, kernel_size: int, stride: int) -> int:
        return (input_len - (kernel_size - 1) - 1) // stride + 1
