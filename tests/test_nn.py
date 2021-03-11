import pytest
import torch.nn as nn
from torch import Tensor
from torch import sum
from src.models.nn import NeuralNetworkModel

def test_initializer():
    sizes = [5, 4, 3]
    output_units = 2

    model = NeuralNetworkModel(sizes, output_units)
    modules = [module for module in model.layer_list]

    for i in range(len(sizes)-1):
        expected_input = sizes[i]
        expected_output = sizes[i+1]
        assert modules[i*3].in_features == expected_input
        assert modules[i*3].out_features == expected_output
        assert type(modules[i*3+1]) == type(nn.BatchNorm1d(0))
        assert modules[i*3+1].weight.size()[0] == expected_output
        assert type(modules[i*3+2]) == type(nn.ReLU())

    assert modules[-2].in_features == sizes[-1]
    assert modules[-2].out_features == output_units
    assert type(modules[-1]) == type(nn.Softmax())

def test_forward():
    sizes = [5, 4, 3]
    output_units = 2
    x = Tensor([[1,2,3,4,5],[6,7,8,9,10]])

    model = NeuralNetworkModel(sizes, output_units)
    
    predictions = model(x)

    for i in range(len(predictions)):
        assert predictions[i].sum().item() >= 0.99 and predictions[i].sum().item() <= 1.01

def test_training_step():
    x = Tensor([[1,2,3,4,5],[6,7,8,9,10]])
    sizes = [5, 4, 3]
    output_units = 2
    y = 0

    batch = (x, y)

    def loss_func(pred, y):
        return pred.sum()

    model = NeuralNetworkModel(sizes, output_units)
    loss = model.training_step(batch, loss_func)

    assert loss == loss_func(model(x), y)

def test_validation_step():
    x = Tensor([[1,2,3,4,5],[6,7,8,9,10]])
    sizes = [5, 4, 3]
    output_units = 2
    y = 0

    batch = (x, y)

    def loss_func(pred, y):
        return pred.sum()

    model = NeuralNetworkModel(sizes, output_units)
    loss = model.validation_step(batch, loss_func)

    assert loss == loss_func(model(x), y)
