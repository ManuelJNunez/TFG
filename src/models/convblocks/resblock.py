"""ResNet block for 1-dimensional data"""
from typing import Optional
import torch.nn as nn
from torch import Tensor


class ResBlock(nn.Module):
    """
    This class contains the implementation of a Residual Block.

    Attributtes
    -----------
    conv1 : nn.Conv1d
        First 1x3 convolution with a ``padding`` of 1.
    bn1 : nn.BatchNorm1d
        Batch normalization for the first convolution output values.
    conv2 : nn.Conv1d
        Second 1x3 convolution with a ``padding`` of 1.
    bn2 : nn.BatchNorm1d
        Batch normalization for the second convolution output values.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: Optional[int] = 1,
        downsample: Optional[nn.Module] = None,
    ):
        """
        ResBlock initializer.

        Parameters
        ----------
        in_channels : int
            Number of channels of the input data.
        out_channels : int
            Number of channels of the output data.
        stride : int
            Stride of the first 3x3 convolution. The default value is 1.
        downsample : Optional[nn.Module]
            Downsample method for identity data.
        """
        super().__init__()
        kernel_size = 3
        padding_size = 1
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size, padding=padding_size, stride=stride
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size, padding=padding_size
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.downsample = downsample

    def forward(self, data: Tensor) -> Tensor:
        """
        Given a batch of `data`, computes the output of the Residual Block.

        Parameters
        ----------
        data : Tensor
            Input of the Residual Block.

        Returns
        -------
        Tensor
            The output of the forward progragation in the Residual Block layers.
        """
        addition = data

        out = self.conv1(data)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            addition = self.downsample(data)

        out += addition
        out = self.relu(out)

        return out
