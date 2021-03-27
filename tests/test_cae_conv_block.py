import pytest
import torch
import torch.nn as nn
from src.models.convblocks.cae_conv_block import CAEBlock

down_in_channels = 1
down_out_channels = 122

up_in_channels = 122
up_out_channels = 1

samples = 200
length = 100

@pytest.fixture
def downblock():
    return CAEBlock(down_in_channels, down_out_channels, "down")


@pytest.fixture
def upblock():
    return CAEBlock(up_in_channels, up_out_channels, "up")


@pytest.fixture
def data_down_sample():
    return torch.rand((samples, down_in_channels, length))

@pytest.fixture
def data_up_sample():
    return torch.rand((samples, up_in_channels, length))


@pytest.fixture
def concatenated_block():
    return torch.rand((samples, up_in_channels, length * 2))


def test_down_block_init(downblock):
    layers = downblock.layers

    # Asserts about the first conv layer
    assert layers[0].in_channels == down_in_channels
    assert layers[0].out_channels == down_out_channels

    # Assert about the next ReLU
    assert isinstance(layers[1], nn.ReLU)

    # Asserts about the second conv layer
    assert layers[2].in_channels == down_out_channels
    assert layers[2].out_channels == down_out_channels

    # Assert about the next ReLU
    assert isinstance(layers[3], nn.ReLU)


def test_up_block_init(upblock):
    layers = upblock.layers

    # Asserts about the upconv (should duplicate the array length)
    assert layers[0].in_channels == up_in_channels
    assert layers[0].out_channels == up_in_channels
    assert layers[0].kernel_size == (2,)
    assert layers[0].stride == (2,)

    # Asserts about the first conv layer
    assert layers[1].in_channels == up_in_channels * 2
    assert layers[1].out_channels == up_out_channels

    # Assert about the next ReLU
    assert isinstance(layers[2], nn.ReLU)

    # Asserts about the second conv layer
    assert layers[3].in_channels == up_out_channels
    assert layers[3].out_channels == up_out_channels

    # Assert about the next ReLU
    assert isinstance(layers[4], nn.ReLU)


def test_init_value_error_0():
    with pytest.raises(ValueError):
        CAEBlock(down_in_channels, down_out_channels, "hello")


def test_init_value_error_1():
    with pytest.raises(ValueError):
        CAEBlock(up_in_channels, up_out_channels, "down")


def test_init_value_error_2():
    with pytest.raises(ValueError):
        CAEBlock(down_in_channels, down_out_channels, "up")


def test_forward_down(downblock, data_down_sample):
    output = downblock(data_down_sample)

    # Asserts about the output size
    assert output.size(0) == samples
    assert output.size(1) == down_out_channels
    assert output.size(2) == length


def test_forward_up(upblock, data_up_sample, concatenated_block):
    output = upblock(data_up_sample, concatenated_block)

    assert output.size(0) == samples
    assert output.size(1) == up_out_channels
    assert output.size(2) == length * 2

def test_forward_up_exception(upblock, data_up_sample):
    with pytest.raises(TypeError):
        upblock.forward(data_up_sample)
