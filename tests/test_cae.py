import pytest
import torch
import torch.nn as nn
from src.ml.models.cae import ConvAutoencoder

in_channels = 1
initial_channels = 8
depth = 3
classifier_layers = [10, 5]
latent_size = 10
classes = 2

samples = 200
height = width = 128


@pytest.fixture
def data():
    return torch.rand((samples, in_channels, height, width))


@pytest.fixture
def model(data):
    return ConvAutoencoder(
        classifier_layers,
        initial_channels,
        depth,
        classes,
        data.size(),
        latent_size,
    )


def test_encoder(model):
    encoder = model.encoder

    assert len(encoder) // 2 == depth
    assert encoder[0].in_channels == in_channels
    assert encoder[0].out_channels == initial_channels
    assert isinstance(encoder[1], nn.ReLU)

    expected_size = initial_channels

    for i in range(2, len(encoder), 2):
        assert encoder[i].in_channels == expected_size
        expected_size *= 2
        assert encoder[i].out_channels == expected_size
        assert isinstance(encoder[i + 1], nn.ReLU)


def test_decoder(model):
    decoder = model.decoder

    assert len(decoder) // 2 == depth

    expected_size = initial_channels * 2 ** (depth - 1)

    for i in range(0, len(decoder) - 2, 2):
        assert decoder[i].in_channels == expected_size
        expected_size //= 2
        assert decoder[i].out_channels == expected_size
        assert isinstance(decoder[i + 1], nn.ReLU)

    assert decoder[-2].in_channels == initial_channels
    assert decoder[-2].out_channels == in_channels
    assert isinstance(decoder[-1], nn.ReLU)


def test_linear_1(model, data):
    linear1 = model.linear1

    expected_size = data.size(2) // (2 ** depth)
    expected_channels = initial_channels * 2 ** (depth - 1)

    assert linear1.in_features == (expected_size ** 2) * expected_channels
    assert linear1.out_features == latent_size


def test_linear_2(model, data):
    linear2 = model.linear2

    expected_size = data.size(2) // (2 ** depth)
    expected_channels = initial_channels * 2 ** (depth - 1)

    assert linear2.in_features == latent_size
    assert linear2.out_features == (expected_size ** 2) * expected_channels


def test_classifier(model):
    classifier = model.classifier

    assert classifier.layer_sizes == classifier_layers
    assert classifier.output_units == classes


def test_forward(model, data):
    decoder_output, classifier_output = model(data)

    assert decoder_output.size() == torch.Size((samples, in_channels, height, width))
    assert classifier_output.size() == torch.Size((samples, classes))


def test_training_step(model, data):
    # Fake loss function
    def loss_func(decoder_output, data, classifier_output, label):
        return torch.ones(data.size(0))

    labels = torch.ones(samples)

    batch = (data, labels)

    loss = model.training_step(batch, loss_func)

    assert loss.equal(torch.ones(data.size(0)))


def test_initializar_exception_1(data):
    with pytest.raises(ValueError):
        ConvAutoencoder([], initial_channels, depth, classes, data.size(), latent_size)


def test_initializar_exception_2(data):
    with pytest.raises(ValueError):
        ConvAutoencoder(classifier_layers, 0, depth, classes, data.size(), latent_size)


def test_initializar_exception_3(data):
    with pytest.raises(ValueError):
        ConvAutoencoder(
            classifier_layers, initial_channels, 0, classes, data.size(), latent_size
        )


def test_initializar_exception_4(data):
    with pytest.raises(ValueError):
        ConvAutoencoder(
            classifier_layers, initial_channels, depth, 0, data.size(), latent_size
        )


def test_initializar_exception_5(data):
    with pytest.raises(ValueError):
        ConvAutoencoder(
            classifier_layers,
            initial_channels,
            depth,
            classes,
            torch.Size((1, 2, 3)),
            latent_size,
        )


def test_initializar_exception_6(data):
    with pytest.raises(ValueError):
        ConvAutoencoder(
            classifier_layers, initial_channels, depth, classes, data.size(), 0
        )
