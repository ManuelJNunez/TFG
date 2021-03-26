import pytest
import torch
import torch.nn as nn
from src.models.neuralnet import NeuralNetworkModel

SIZES = [5, 4, 3]
OUTPUT_UNITS = 2
SAMPLES = 2


@pytest.fixture
def model():
    return NeuralNetworkModel(SIZES, OUTPUT_UNITS)


def test_initializer(model):
    modules = model.layer_list

    for i in range(len(SIZES) - 1):
        expected_input = SIZES[i]
        expected_output = SIZES[i + 1]
        assert modules[i * 3].in_features == expected_input
        assert modules[i * 3].out_features == expected_output
        assert isinstance(modules[i * 3 + 1], nn.BatchNorm1d)
        assert modules[i * 3 + 1].num_features == expected_output
        assert isinstance(modules[i * 3 + 2], nn.ReLU)

    assert modules[-2].in_features == SIZES[-1]
    assert modules[-2].out_features == OUTPUT_UNITS
    assert isinstance(modules[-1], nn.Softmax)


def test_forward():
    x = torch.rand(SAMPLES, SIZES[0])

    model = NeuralNetworkModel(SIZES, OUTPUT_UNITS)

    predictions = model(x)

    assert predictions.sum(dim=1).allclose(torch.ones(SAMPLES))


def test_training_step():
    x = torch.rand(SAMPLES, SIZES[0])
    y = 0

    batch = (x, y)

    def loss_func(pred, y):
        return pred.sum()

    model = NeuralNetworkModel(SIZES, OUTPUT_UNITS)
    loss = model.training_step(batch, loss_func)

    assert loss == loss_func(model(x), y)


def test_validation_step():
    x = torch.rand(SAMPLES, SIZES[0])
    y = 0

    batch = (x, y)

    def loss_func(pred, y):
        return pred.sum()

    model = NeuralNetworkModel(SIZES, OUTPUT_UNITS)
    loss = model.validation_step(batch, loss_func)

    assert loss == loss_func(model(x), y)
