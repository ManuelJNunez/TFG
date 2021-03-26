"""Implementation of a Convolution Autoencoder Block"""
from typing import Optional
import torch
import torch.nn as nn


class CAEBlock(nn.Module):
    """
    This class defines a block of a convolutional autoencoder.

    Attributes
    ----------
    modules : nn.ModuleList
        List of layers of the Convolutional Block.
    block_type : str
        Type of block. It only have two possible values: `up` or `down`. Use `down` for a
        down-sampling block and `up` for an up-sampling block.
    """

    def __init__(self, in_channels: int, out_channels: int, block_type: str):
        """
        Initializes a Convolutional Autoencoder Block.

        Parameters
        ----------
        in_channels : int
            Number of channels of the input tensor.
        out_channels : int
            Number of channels of the output tensor.
        block_type : str
            Type of block: it only have two possible values: `up` or `down`. Use `down` for a
            down-sampling block and `up` for an up-sampling block.

        Raises
        ------
        ValueError
            If the value of block_type is different of `up` or `down`.
            If the number of output channels of a down-sampling block should be greater than
            the number of input channels.
            If the number of input channels of an up-sampling block should be greater than the
            number of output channels.
        """
        super().__init__()
        self.modules = nn.ModuleList([])
        self.block_type = block_type
        upconv_kernel_size = 2
        upconv_stride = 2
        kernel_size = 3
        padding = 1

        if block_type not in ("down", "up"):
            raise ValueError(
                "The block_type parameter of CAEBlock only have two possible values: down or up"
            )

        if block_type == "down" and in_channels < out_channels:
            # pylint: disable=C0301
            raise ValueError(
                "The number of output channels of a down-sampling block should be greater than the number of input channels"
            )

        if block_type == "up" and in_channels > out_channels:
            # pylint: disable=C0301
            raise ValueError(
                "The number of input channels of an up-sampling block should be greater than the number of output channels"
            )

        if block_type == "up":
            self.modules.append(
                nn.ConvTranspose1d(
                    in_channels,
                    in_channels,
                    kernel_size=upconv_kernel_size,
                    stride=upconv_stride,
                )
            )

        self.modules.append(
            nn.Conv1d(
                in_channels if block_type == "down" else in_channels * 2,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
            )
        )
        self.modules.append(nn.ReLU())

        self.modules.append(
            nn.Conv1d(
                out_channels, out_channels, kernel_size=kernel_size, padding=padding
            )
        )
        self.modules.append(nn.ReLU())

    def forward(
        self,
        block_input: torch.Tensor,
        concatenated_block: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Given a block_input, computes the output for the Convolutional Autoencoder Block.

        Parameters
        ----------
        block_input : torch.Tensor
            Input of the Convolutional Autoencoder Block.
        concatenated_block : Optional[torch.Tensor]
            It is mandatory for an up-sampling block (see Unet's design).

        Returns
        -------
        torch.Tensor
            The output of the forward progragation in the Convolutional Autoencoder Block.

        Raises
        ------
        TypeError
            If you do not pass the `concatenated_block` parameter to an up-sampling block.
        """
        next_input = block_input

        if self.block_type == "up":
            if concatenated_block is None:
                # pylint: disable=C0301
                raise TypeError(
                    "The forward method of a CAEBlock with block_type 'up' requires concatenated_block parameter"
                )

            next_input = self.modules[0](next_input)
            next_input = torch.cat([next_input, concatenated_block], dim=1)
            modules = self.modules[1:]
        else:
            modules = self.modules

        for layer in modules:
            next_input = layer(next_input)

        return next_input
