"""
VAE implementation using PyTorch. Based in this implementation:
https://github.com/pytorch/examples/tree/master/vae
"""
from typing import Tuple
import torch
import torch.nn as nn
from .neuralnet import NeuralNetworkModel

class VAE(nn.Module):
    """
    Varitional Autoencoder (VAE) implementation with variable depth and layer sizes.
    This Autoencoder has two parts: the encoder (reduces the dimensionality of the data) and the
    decoded (reconstructs the data with the smallest possible error). The VAE learns the mean
    and the variance of the data and construct the latent-code using the reparametrization trick.
    A fully-connected neural network is used for classifying the latent-code.

    Attributes
    ----------
    encoder : nn.ModuleList
        Contains the layers of the encoder section.
    mean : nn.Linear
        The output of this layer is the mean learnt from the data.
    logvar : nn.Linear
        The output of this layer is the logvariance learnt from the data.
    decoder : nn.ModuleList
        Contains the layers of the decoder section.
    classifier : NeuralNetworkModel
        Model that classifies the latent-code (output of the encoder).
    """

    def __init__(self, autoencoder_sizes: [int], classifier_sizes: [int]):
        """
        Initialize a new VAE with a classifier.

        Parameters
        ----------
        autoencoder_sizes : [int]
            The size of the layers of the encoder of the VAE (the decoder is symmetric).
        classifier_sizes : [int]
            The size of the layers of the latent-code classifier.
        """
        super().__init__()

        self.encoder = nn.ModuleList([])

        for i in range(len(autoencoder_sizes) - 2):
            input_size = autoencoder_sizes[i]
            output_size = autoencoder_sizes[i + 1]
            self.encoder.append(
                nn.Linear(in_features=input_size, out_features=output_size)
            )
            self.encoder.append(nn.ReLU())

        self.mean = nn.Linear(
            in_features=autoencoder_sizes[len(autoencoder_sizes) - 2],
            out_features=autoencoder_sizes[len(autoencoder_sizes) - 1],
        )
        self.logvar = nn.Linear(
            in_features=autoencoder_sizes[len(autoencoder_sizes) - 2],
            out_features=autoencoder_sizes[len(autoencoder_sizes) - 1],
        )

        self.decoder = nn.ModuleList([])

        for i in reversed(range(1, len(autoencoder_sizes))):
            input_size = autoencoder_sizes[i]
            output_size = autoencoder_sizes[i - 1]
            self.decoder.append(
                nn.Linear(in_features=input_size, out_features=output_size)
            )
            self.decoder.append(nn.ReLU())

        self.classifier = NeuralNetworkModel(
            classifier_sizes[:-1], classifier_sizes[-1]
        )

    def encode(self, encoder_input: torch.Tensor) -> Tuple[torch.Tensor]:
        """
        This method computes the encoder's output.

        Parameters
        ----------
        encoder_input : torch.Tensor
            Data to which we want to reduce the dimensionality.

        Returns
        -------
        tuple
            The first position of the tuple contains the tensor with the learnt mean and the
            second position contains the learnt logvariance.
        """
        next_input = encoder_input

        for layer in self.encoder:
            next_input = layer(next_input)

        return self.mean(next_input), self.logvar(next_input)

    # pylint: disable=R0201
    def reparametrize(self, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        This method computes the output of the reparametrization trick.

        Parameters
        ----------
        mean : torch.Tensor
            Tensor that contains the mean learnt from the data.
        logvar : torch.Tensor
            Tensor that contains the logvariance learnt from the data.

        Returns
        -------
        torch.Tensor
            Retrieves the latent-code using the reparametrization trick.
        """
        std = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(std)
        return mean + epsilon * std

    def decode(self, latent_code: torch.Tensor) -> torch.Tensor:
        """
        This method computes the output of the decoder.

        Parameters
        ----------
        latent code : torch.Tensor
            Output tensor from the encoder and reparametrization trick.

        Returns
        -------
        torch.Tensor
            Tensor that contains the output of the decoder (should have the same size as the
            input).
        """
        next_input = latent_code

        for layer in self.decoder:
            next_input = layer(next_input)

        return next_input

    def classify(self, latent_code: torch.Tensor) -> torch.Tensor:
        """
        This method computes the output of the classifier.

        Parameters
        ----------
        latent code : torch.Tensor
            Output from the encoder and reparametrization trick.

        Returns
        -------
        torch.Tensor
            The output of the classifier whose size is `num samples` x `num clases`.
        """
        return self.classifier(latent_code)

    def forward(self, vae_input: torch.Tensor) -> Tuple[torch.Tensor]:
        """
        Method that computes the output of the Autoencoder and the classifier.

        Parameters
        ----------
        vae_input: input of the neural network.

        Returns
        -------
        tuple
            This tuple contains the next objects:
                1. Decoder output
                2. Classifier output
                3. Mean learnt from data
                4. Logvariance learnt from data
        """
        mean, logvar = self.encode(vae_input)

        latent_code = self.reparametrize(mean, logvar)

        return self.decode(latent_code), self.classify(latent_code), mean, logvar

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
        decoder_output, classifier_output, mean, logvar = self(data)
        loss = loss_func(decoder_output, data, classifier_output, label, mean, logvar)

        return loss

    def validation_step(self, batch: torch.Tensor, loss_func: callable) -> torch.Tensor:
        """
        This method computes the validation batch loss.

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
        decoder_output, classifier_output, mean, logvar = self(data)
        loss = loss_func(decoder_output, data, classifier_output, label, mean, logvar)

        return loss
