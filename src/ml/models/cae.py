"""Implementation of a Convolutional Autoencoder with classifier"""
from typing import Tuple, List
import torch
from torch import Tensor
import torch.nn as nn
from .neuralnet import NeuralNetworkModel


class ConvAutoencoder(nn.Module):
    """
    This class defines a convolutional autoencoder and a classifier of the latent-code.
    It has two parts, an encoder (reduces the dimensionality of the data) and a decoder
    (reconstructs the dimensionally-reducted data with the smallest error).

    Attributes
    ----------
    encoder : nn.ModuleList
        Contains the layers of the encoder section.
    decoder : nn.ModuleList
        Contains the layers of the decoder section.
    classifier : NeuralNetworkModel
        Model that classifies the latent-code (output of the encoder).
    """

    def __init__(
        self,
        classifier_layers: List[int],
        first_conv_out_channels: int,
        depth: int,
        classes: int,
        data_size: Tensor,
        latent_size: int,
    ):
        """
        ConvAutoencoder class initializer.

        Parameters
        ----------
        in_channels : int
            Number of channels of the input data.
        first_conv_out_channels : int
            Number of channels of the output of the first ConvBlock. The number of channels of the
            next convolutional layers will be ``first_conv_out_channels * 2 * i`` being `i` the
            layer order number.
        depth : int
            Depth of the encoder and autoencoder (number of ConvBlocks on each part).
        classifier_layers : List[int]
            Sizes of each layer of the classifier model.
        classes : int
            Number of classes in the dataset you are using.
        data_size : Tensor
            Size of the data that you need to process.
        latent_size : int
            Size of the latent vector.

        Raises
        ------
        Value
            If any parameter has value less than 1.
            If ``classifier_layers`` or ``data_size`` have length 0.
            If ``classifier_layers[0]`` is not equal to ``latent_size``.
            If the length of ``data_size`` is less than 4.
        """
        super().__init__()
        encoder_kernel_size = 3
        decoder_kernel_size = 2
        stride = 2
        padding = 1

        if len(classifier_layers) < 1:
            raise ValueError("classifier_layers be grater than 1")

        if classifier_layers[0] != latent_size:
            raise ValueError(
                "The size of the first classifier layer should be equal to the latent_space size"
            )

        if len(data_size) != 4:
            raise ValueError("data_size must have length of 4")

        if classes < 1:
            raise ValueError("The number of classes cannot be 0 or less")

        if latent_size < 1:
            raise ValueError("latent_size cannot be less than 1")

        if first_conv_out_channels < 1:
            raise ValueError("first_conv_out_channels cannot be less than 1")

        if depth < 1:
            raise ValueError("in_channels cannot be less than 1")

        input_channels = data_size[1]
        encoder_layers = []

        for i in range(depth):
            output_channels = input_channels * 2 if i != 0 else first_conv_out_channels
            encoder_layers.append(
                nn.Conv2d(
                    input_channels,
                    output_channels,
                    encoder_kernel_size,
                    stride,
                    padding,
                )
            )
            encoder_layers.append(nn.ReLU())
            input_channels = output_channels

        self.encoder = nn.Sequential(*encoder_layers)

        output_size1 = data_size[2] // (2 ** depth)
        output_size2 = data_size[3] // (2 ** depth)
        output_size = output_size1 * output_size2
        output_channels = first_conv_out_channels * (2 ** (depth - 1))

        self.linear1 = nn.Linear(output_size * output_channels, latent_size)
        self.linear2 = nn.Linear(latent_size, output_size * output_channels)

        decoder_layers = []

        for i in range(depth):
            output_channels = (
                (input_channels // 2) if i != (depth - 1) else data_size[1]
            )
            decoder_layers.append(
                nn.ConvTranspose2d(
                    input_channels, output_channels, decoder_kernel_size, stride
                )
            )
            decoder_layers.append(nn.ReLU())
            input_channels = output_channels

        self.decoder = nn.Sequential(*decoder_layers)
        self.classifier = NeuralNetworkModel(classifier_layers, classes)

    def forward(self, data: torch.Tensor) -> Tuple[torch.Tensor]:
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
        output = self.encoder(data)
        original_size = output.size()
        output = torch.flatten(output, start_dim=1)
        latent_code = self.linear1(output)
        output = self.linear2(latent_code)
        reshaped_latent_code = torch.reshape(output, original_size)

        return self.decoder(reshaped_latent_code), self.classifier(latent_code)

    def training_step(self, batch: torch.Tensor, loss_func: callable) -> torch.Tensor:
        """
        This method computes the training batch loss.

        Parameters
        ----------
        batch : torch.Tensor
            A tensor with a portion of data.
        loss_func : callable
            Function that computes the training loss in each batch of data.

        Returns
        -------
        torch.Tensor
            The result of the loss functions for the batch of data, the output size depends on
            the used loss function.
        """
        data, label = batch
        cae_output, predictions = self(data)
        loss = loss_func(cae_output, data, predictions, label)

        return loss
