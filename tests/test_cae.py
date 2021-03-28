import pytest
import torch
import torch.nn as nn
from src.models.cae import ConvAutoencoder


in_channels = 1
initial_channels = 8
depth = 3
classifier_layers = [50, 25, 10]
classes = 2

samples = 200
length = 128


@pytest.fixture
def model():
    first_classifier_size = initial_channels * (2 ** (depth - 1))
    first_classifier_size *= length // (2 ** depth)

    classifier_layers.insert(0, first_classifier_size)

    return ConvAutoencoder(
        in_channels,
        initial_channels,
        depth,
        classifier_layers=classifier_layers,
        classes=classes,
    )


@pytest.fixture
def data():
    return torch.rand((samples, in_channels, length))


def test_initializer(model, data):
    encoder = model.encoder

    # Assertions about encoder layers
    for i, convblock in enumerate(encoder[:-1]):
        assert convblock.in_channels == i * initial_channels if i != 0 else in_channels
        assert (
            convblock.out_channels == 2 * i * initial_channels
            if i != 0
            else initial_channels
        )
        assert convblock.block_type == "down"

    assert encoder[-1].in_channels == (2 * (depth - 1) * initial_channels)
    assert encoder[-1].out_channels == (2 * (depth - 1) * initial_channels)

    decoder = model.decoder
    encoder_output_channels = 2 * (depth - 1) * initial_channels

    # Assertions about decoder layers
    for i, convblock in enumerate(decoder):
        assert (
            convblock.in_channels == (encoder_output_channels // (2 * i))
            if i != 0
            else encoder_output_channels
        )
        assert convblock.out_channels == (
            (encoder_output_channels // (2 * (i + 1)))
            if i < (depth - 1)
            else in_channels
        )

    # Assertions about the classier
    classifier = model.classifier

    assert classifier.layer_sizes == classifier_layers
    assert classifier.output_units == classes
    assert isinstance(model.flatten, nn.Flatten)


def test_initializer_exception_1():
    with pytest.raises(TypeError):
        ConvAutoencoder(in_channels, initial_channels, depth)


def test_initializer_exception_2():
    with pytest.raises(TypeError):
        ConvAutoencoder(in_channels, initial_channels, depth, classes=2)


def test_initializer_exception_3():
    with pytest.raises(TypeError):
        ConvAutoencoder(
            in_channels, initial_channels, depth, classifier_layers=classifier_layers
        )


def test_forward(model, data):
    decoded, prediction = model(data)

    assert decoded.size(0) == samples
