import pytest
import torch
import torch.nn as nn
from src.models.convblocks.resblock import ResBlock

in_channels = 1
out_channels = 32
downsample = nn.Conv1d(in_channels, out_channels, 1)
samples = 200
length = 100


@pytest.fixture
def model():
    return ResBlock(in_channels, out_channels, downsample)


@pytest.fixture
def data():
    return torch.randn((samples, in_channels, length))


def test_initializer(model):
    assert isinstance(model.conv1, nn.Conv1d)
    assert isinstance(model.bn1, nn.BatchNorm1d)
    assert isinstance(model.relu, nn.ReLU)
    assert isinstance(model.conv2, nn.Conv1d)
    assert isinstance(model.bn2, nn.BatchNorm1d)
    assert model.downsample == downsample


def test_forward(monkeypatch, model, data):
    output = model(data)

    assert output.size(0) == samples
    assert output.size(1) == out_channels
    assert output.size(2) == length