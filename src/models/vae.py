"""
VAE implementation using PyTorch. Based in this implementation:
https://github.com/pytorch/examples/tree/master/vae
"""
import torch
import torch.nn as nn
from .neuralnet import NeuralNetworkModel


class VAE(nn.Module):
    """VAE class with variable depth and layer sizes"""

    def __init__(self, autoencoder_sizes: [int], classifier_sizes: [int]):
        """
        Initialize a new VAE with a classifier.

        Parameters:
            autoencoder_sizes ([int]): the size of the layers of the encoder of the
                VAE (the decoder is symmetric).
            classifier_sizes ([int]): the size of the layers of the latent code classifier.
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

    def encode(self, encoder_input: torch.Tensor) -> torch.Tensor:
        """
        This method computes the encoder output.

        Parameters:
            encoder_input: data to which we want to reduce the dimensionality.
        """
        next_input = encoder_input

        for layer in self.encoder:
            next_input = layer(next_input)

        return self.mean(next_input), self.logvar(next_input)

    # pylint: disable=R0201
    def reparametrize(self, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        This method computes the output of the reparametrization trick.
        """
        std = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(std)
        return mean + epsilon * std

    def decode(self, latent_code: torch.Tensor) -> torch.Tensor:
        """
        This method computes the output of the decoder.

        Parameters:
            latent code: output from the encoder and reparametrization trick.
        """
        next_input = latent_code

        for layer in self.decoder:
            next_input = layer(next_input)

        return next_input

    def classify(self, latent_code: torch.Tensor) -> torch.Tensor:
        """
        This method computes the output of the classifier.

        Parameters:
            latent code: output from the encoder and reparametrization trick.
        """
        return self.classifier(latent_code)

    def forward(self, vae_input: torch.Tensor) -> torch.Tensor:
        """
        Method that computes the output of the Autoencoder and the classifier.

        Parameter:
            vae_input: input of the neural network.
        """
        mean, logvar = self.encode(vae_input)

        latent_code = self.reparametrize(mean, logvar)

        return self.decode(latent_code), self.classify(latent_code), mean, logvar

    def training_step(self, batch: torch.Tensor, loss_func: callable) -> torch.Tensor:
        """
        This method computes the training batch loss.

        Parameters:
            batch: a smaller portion of data.
            loss_func: function that computes the training loss in each batch of data.
        """
        data, label = batch
        decoder_output, classifier_output, mean, logvar = self(data)
        loss = loss_func(decoder_output, data, classifier_output, label, mean, logvar)

        return loss

    def validation_step(self, batch: torch.Tensor, loss_func: callable) -> torch.Tensor:
        """
        This method computes the validation batch loss.

        Parameters:
            batch: a smaller portion of data.
            loss_func: function that computes the validation loss in each batch of data.
        """
        data, label = batch
        decoder_output, classifier_output, mean, logvar = self(data)
        loss = loss_func(decoder_output, data, classifier_output, label, mean, logvar)

        return loss
