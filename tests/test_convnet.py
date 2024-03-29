import pytest
import torch
import torch.nn as nn
from src.ml.models.convnet import ConvClassifier
from src.ml.models.neuralnet import NeuralNetworkModel

samples = 200
height = width = 80
out_channels = [2, 6]
first_layer = int(((((height - 4) / 2) - 4) / 2))
classifier_sizes = [(first_layer ** 2) * out_channels[1], 100, 50, 10]
classes = 2
in_channels = 1


@pytest.fixture
def model():
    return ConvClassifier(classifier_sizes, out_channels, classes, in_channels)


@pytest.fixture
def data():
    return torch.rand((samples, in_channels, height, width))


def test_initializer(model):
    layers = model.layers

    assert isinstance(layers[-2], nn.Flatten)
    assert isinstance(layers[-1], NeuralNetworkModel)
    assert layers[-1].layer_sizes == classifier_sizes
    assert layers[-1].output_units == classes


def test_forward(model, data):
    output = model(data)

    assert output.size(0) == samples
    assert output.size(1) == classes


def test_training_step(model, data):
    # Fake loss function
    def loss_func(predicted, label):
        return torch.ones(predicted.size(0))

    y = torch.randint(0, 2, (samples,))
    batch = (data, y)

    loss = model.training_step(batch, loss_func)

    assert loss.equal(torch.ones(data.size(0)))
