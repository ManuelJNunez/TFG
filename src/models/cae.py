"""Implementation of a Convolutional Autoencoder with classifier"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .convblocks.cae_conv_block import CAEBlock
from .neuralnet import NeuralNetworkModel


class ConvAutoencoder(nn.Module):
    """
    This class defines a convolutional autoencoder and a classifier of the latent-code.
    It has two parts, an encoder (reduces the dimensionality of the data) and a decoder
    (reconstructs the dimensionally-reducted data with the smallest error). The
    architecture of the autoencoder is based on UNet's architecture.

    Attributes
    ----------
    encoder : nn.ModuleList
        Contains the layers of the encoder section.
    decoded : nn.ModuleList
        Contains the layers of the decoder section.
    flatten : nn.Flatten
        Flatten layer for the classifier fully-connected neural network.
    classifier : NeuralNetworkModel
        Model that classifies the latent-code (output of the encoder).
    """

    def __init__(self, in_channels: int, initial_channels: int, depth: int, **kwargs):
        """
        ConvAutoencoder class initializer.

        Parameters
        ----------
        in_channels : int
            Number of channels of the input data.
        initial_channels : int
            Number of channels of the output of the first ConvBlock.
        depth : int
            Depth of the encoder and autoencoder (number of ConvBlocks on each part).
        classifier_layers : [int]
            Sizes of each layer of the classifier model.
        classes : int
            Number of classes in the dataset you are using.

        Raises
        ------
        TypeError
            If the parameters `classifier_layers` or `classes` are missing.
        """
        super().__init__()
        prev_channels = in_channels
        self.encoder = nn.ModuleList([])

        if "classifier_layers" not in kwargs:
            raise TypeError("missing 1 required argument: 'classifier_layers'")

        if "classes" not in kwargs:
            raise TypeError("missing 1 required argument: 'classes'")

        # Initialize encoder
        for i in range(depth):
            next_channels = 2 * prev_channels if i != 0 else initial_channels
            self.encoder.append(CAEBlock(prev_channels, next_channels, "down"))
            prev_channels = next_channels

        self.encoder.append(CAEBlock(next_channels, next_channels, "down"))

        self.decoder = nn.ModuleList([])
        # Initialize decoder
        for i in reversed(range(depth)):
            next_channels = prev_channels // 2 if i != 0 else in_channels
            self.decoder.append(CAEBlock(prev_channels, next_channels, "up"))
            prev_channels = next_channels

        self.flatten = nn.Flatten()

        self.classifier = NeuralNetworkModel(
            kwargs["classifier_layers"], kwargs["classes"]
        )

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """
        Given a batch of `data`, computes the output of the ConvAutoencoder and the output
        of the latent-code classifier.

        Parameters
        ----------
        data : torch.Tensor
            Data which you need to classify.

        Returns
        -------
        tuple
            This tuple contains two tensors: the first with autoencoder's forward propagation output
            and a second one with classifier's forward propagation output.
        """
        block_outputs = []
        next_input = data

        for i, layer in enumerate(self.encoder):
            next_input = layer(next_input)

            if i != len(self.encoder) - 1:
                block_outputs.append(next_input)
                next_input = F.max_pool1d(next_input, kernel_size=2)

        class_output = self.classifier(self.flatten(next_input))

        for layer in self.decoder:
            next_input = layer(next_input, block_outputs.pop())

        return next_input, class_output
