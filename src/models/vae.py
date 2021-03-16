"""
VAE implementation using PyTorch. Based in this implementation:
https://github.com/pytorch/examples/tree/master/vae
"""
import torch
import torch.nn as nn


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

        encoder_layers = []

        for i in range(len(autoencoder_sizes) - 2):
            input_size = autoencoder_sizes[i]
            output_size = autoencoder_sizes[i + 1]
            encoder_layers.append(
                nn.Linear(in_features=input_size, out_features=output_size)
            )
            encoder_layers.append(nn.ReLU())

        self.encoder = nn.ModuleList(encoder_layers)

        self.mean = nn.Linear(
            in_features=autoencoder_sizes[len(autoencoder_sizes) - 2],
            out_features=autoencoder_sizes[len(autoencoder_sizes) - 1],
        )
        self.logvar = nn.Linear(
            in_features=autoencoder_sizes[len(autoencoder_sizes) - 2],
            out_features=autoencoder_sizes[len(autoencoder_sizes) - 1],
        )

        decoder_layers = []

        for i in reversed(range(len(autoencoder_sizes))):
            input_size = autoencoder_sizes[i]
            output_size = classifier_sizes[i - 1]
            decoder_layers.append(
                nn.Linear(in_features=input_size, out_features=output_size)
            )
            decoder_layers.append(nn.ReLU())

        self.decoder = nn.ModuleList(decoder_layers)

        classifier_layers = []

        for i in range(len(classifier_sizes) - 2):
            input_size = classifier_sizes[i]
            output_size = classifier_sizes[i + 1]
            classifier_layers.append(
                nn.Linear(in_features=input_size, out_features=output_size)
            )
            classifier_layers.append(nn.BatchNorm1d(num_features=output_size))
            classifier_layers.append(nn.ReLU())

        classifier_layers.append(
            nn.Linear(
                in_features=classifier_sizes[len(classifier_sizes) - 2],
                out_features=classifier_sizes[len(classifier_sizes) - 1],
            )
        )
        classifier_layers.append(nn.Softmax(dim=1))

        self.classifier = nn.ModuleList(classifier_layers)

    def encode(self, encoder_input):
        """
        This method computes the encoder output.

        Parameters:
            encoder_input: data to which we want to reduce the dimensionality.
        """
        next_input = encoder_input

        for layer in self.encoder:
            next_input = layer(next_input)

        return self.mean(next_input), self.logvar(next_input)

    def reparametrize(self):
        """
        This method computes the output of the reparametrization trick.
        """
        std = torch.exp(0.5 * self.logvar)
        epsilon = torch.randn_like(std)
        return self.mean + epsilon * std

    def decode(self, latent_code):
        """
        This method computes the output of the decoder and of the classifier.

        Parameters:
            latent code: output from the encoder and reparametrization trick.
        """
        next_input_decoder = latent_code

        for layer in self.decoder:
            next_input_decoder = layer(next_input_decoder)

        next_input_classifier = latent_code

        for layer in self.classifier:
            next_input_classifier = layer(next_input_classifier)

        return next_input_decoder, next_input_classifier

    def forward(self, vae_input):
        """
        Method that computes the output of the Autoencoder and the classifier.

        Parameter:
            vae_input: input of the neural network.
        """
        mean, logvar = self.encode(vae_input)

        latent_code = self.reparametrize()

        return self.decode(latent_code), mean, logvar

    def training_step(self, batch, loss_func):
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

    def validation_step(self, batch, loss_func):
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
