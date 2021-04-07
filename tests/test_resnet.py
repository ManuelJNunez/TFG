import pytest
import torch
import math
import torch.nn as nn
from src.models.resnet import ResNetModel

num_channels = [2, 4, 8, 16, 32]
number_of_blocks = [2, 2, 2, 2]
classifier_sizes = [10, 2]
in_channels = 1
classes = 2
samples = 500
height = width = 80


@pytest.fixture
def model():
    return ResNetModel(
        num_channels, number_of_blocks, classifier_sizes, in_channels, classes
    )


@pytest.fixture
def data():
    return torch.rand((samples, in_channels, height, width))


def test_initializer(model):
    # The first convolution should increment the number of channels and divide by 2 the H & W
    conv1 = model.conv1
    output_height = math.floor(
        (height + 2 * conv1.padding[0] - conv1.kernel_size[0]) / conv1.stride[0] + 1
    )

    assert output_height == height // 2

    output_width = math.floor(
        (width + 2 * conv1.padding[1] - conv1.kernel_size[1]) / conv1.stride[1] + 1
    )

    assert output_width == width // 2

    # The number of features of the first BN should be the number of channels of the conv1 output
    assert model.bn1.num_features == num_channels[0]

    # The activation of ResNet is ReLU
    assert isinstance(model.relu, nn.ReLU)

    # The MaxPool should divide by 2 the length
    maxpool = model.maxpool
    output_length = math.floor(
        (height + 2 * maxpool.padding - maxpool.kernel_size) / maxpool.stride + 1
    )

    assert output_length == height // 2

    # Assertions about group of layers
    assert isinstance(model.layer1, nn.Sequential)
    assert isinstance(model.layer2, nn.Sequential)
    assert isinstance(model.layer3, nn.Sequential)
    assert isinstance(model.layer4, nn.Sequential)

    # The average pool should reduce the length of each channel to 1
    assert model.avgpool.output_size == (1, 1)

    # The classifier should be correctly initialized
    classifier_sizes.insert(0, num_channels[-1])
    assert model.classifier.layer_sizes == classifier_sizes
    assert model.classifier.output_units == classes


def test_initializer_exception_1():
    with pytest.raises(ValueError):
        num_channels = [4]
        ResNetModel(
            num_channels, number_of_blocks, classifier_sizes, in_channels, classes
        )


def test_initializer_exception_2():
    with pytest.raises(ValueError):
        number_of_blocks = [4]
        ResNetModel(
            num_channels, number_of_blocks, classifier_sizes, in_channels, classes
        )


def test_create_layer(model):
    out_channels = 32
    blocks = 2
    stride = 2

    sequential = model._create_layer(in_channels, out_channels, blocks, stride)

    assert len(sequential) == blocks


def test_forward(model, data):
    output = model(data)

    assert output.size(0) == samples
    assert output.size(1) == classes
    assert output.sum(dim=1).allclose(torch.ones(samples))


def test_training_step(model, data):
    # Fake loss function
    def loss_func(predicted, label):
        return torch.ones(predicted.size(0))

    y = torch.randint(0, 2, (samples,))
    batch = (data, y)

    loss = model.training_step(batch, loss_func)

    assert loss.equal(torch.ones(data.size(0)))
