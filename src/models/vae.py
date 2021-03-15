import torch.nn as nn


class VAE(nn.Module):
    def __init__(self, autoencoder_sizes: [int], classifier_sizes: [int]):
        super().__init__()

        encoder_layers = []

        for i in range(len(layer_sizes) - 2):
            input_size = autoencoder_sizes[i]
            output_size = autoencoder_sizes[i + 1]
            encoder_layers.append(
                nn.Linear(in_features=input_size, out_features=output_size)
            )
            encoder_layers.append(nn.ReLU())

        self.encoder = nn.ModuleList(encoder_layers)

        self.mu = nn.Linear(
            in_features=autoencoder_sizes[len(autoencoder_sizes) - 2],
            out_features=autoencoder_sizes[len(autoencoder_sizes) - 1],
        )
        self.logvar = nn.Linear(
            in_features=autoencoder_sizes[len(autoencoder_sizes) - 2],
            out_features=autoencoder_sizes[len(autoencoder_sizes) - 1],
        )

        decoder_layers = []

        for i in reversed(range(len(layer_sizes))):
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
        next_input = encoder_input

        for layer in self.encoder:
            next_input = layer(next_input)

        return self.mu(next_input), self.logvar(next_input)

    def reparametrize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(std)
        return mu + epsilon * std

    def decode(self, latent_code):
        next_input_decoder = latent_code

        for layer in self.decoder:
            next_input_decoder = layer(next_input_decoder)

        next_input_classifier = latent_code

        for layer in self.classifier:
            next_input_classifier = layer(next_input_classifier)

        return next_input_decoder, next_input_classifier

    def forward(self, vae_input):
        mu, logvar = self.encode(vae_input)

        latent_code = self.reparametrize(mu, logvar)

        return decode(latent_code), mu, logvar

    def training_step(self, batch, loss_func):
        data, label = batch
        decoder_output, classifier_output, mu, logvar = self(data)
        loss = loss_func(decoder_output, data, classifier_output, label, mu, logvar)

        return loss

    def validation_step(self, batch, loss_func):
        data, label = batch
        decoder_output, classifier_output, mu, logvar = self(data)
        loss = loss_func(decoder_output, data, classifier_output, label, mu, logvar)

        return loss
