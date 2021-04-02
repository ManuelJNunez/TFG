"""Convolutional Classifier model based on LeNet-5"""
from typing import Optional
import torch.nn as nn
from .neuralnet import NeuralNetworkModel
from .basemodel import BaseModel


class ConvClassifier(BaseModel):
    """
    Convolutional uni-dimensional classifier
    """

    def __init__(
        self,
        classifier_sizes: [int],
        classes: int,
        in_channels: Optional[int] = 1,
    ):
        super().__init__()
        kernel_size = 5
        max_pool_kernel = 2
        out_channels = (6, 16)
        self.layers = nn.ModuleList([])

        self.layers.append(nn.Conv1d(in_channels, out_channels[0], kernel_size))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.MaxPool1d(max_pool_kernel))
        self.layers.append(nn.Conv1d(out_channels[0], out_channels[1], kernel_size))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.MaxPool1d(max_pool_kernel))

        self.layers.append(nn.Flatten())
        self.layers.append(NeuralNetworkModel(classifier_sizes, classes))
