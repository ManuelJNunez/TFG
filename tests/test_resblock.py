import pytest
import torch
import torch.nn as nn
from src.ml.models.convblocks.resblock import ResBlock

in_channels = 1
out_channels = 32
downsample = nn.Conv2d(in_channels, out_channels, 1)
downsample_with_stride = nn.Conv2d(in_channels, out_channels, 1, 2)
samples = 50
heigth = width = 80


@pytest.fixture
def model():
    return ResBlock(in_channels, out_channels, stride=1, downsample=downsample)


@pytest.fixture
def model_with_stride():
    return ResBlock(
        in_channels, out_channels, stride=2, downsample=downsample_with_stride
    )


@pytest.fixture
def data():
    return torch.randn((samples, in_channels, heigth, width))


def test_initializer(model):
    assert isinstance(model.conv1, nn.Conv2d)
    assert isinstance(model.bn1, nn.BatchNorm2d)
    assert isinstance(model.relu, nn.ReLU)
    assert isinstance(model.conv2, nn.Conv2d)
    assert isinstance(model.bn2, nn.BatchNorm2d)
    assert model.downsample == downsample


def test_forward(model, data):
    output = model(data)

    assert output.size(0) == samples
    assert output.size(1) == out_channels
    assert output.size(2) == heigth
    assert output.size(3) == width


def test_forward_with_stride(model_with_stride, data):
    output = model_with_stride(data)

    assert output.size(0) == samples
    assert output.size(1) == out_channels
    assert output.size(2) == heigth // 2
    assert output.size(3) == width // 2
