"""Convolutional Classifier model based on LeNet-5"""
from typing import Optional, List
import torch.nn as nn
from .neuralnet import NeuralNetworkModel
from .basemodel import BaseModel


class ConvClassifier(BaseModel):
    """
    Convolutional uni-dimensional classifier

    Attributes
    ----------
    layers : nn.ModuleList
        Layers of this convolutional model
    """

    def __init__(
        self,
        classifier_sizes: List[int],
        out_channels: List[int],
        classes: int,
        in_channels: Optional[int] = 1,
    ):
        """
        ConvClassifier initializer

        Parameters
        ----------
        classifier_sizes : List[int]
            Size of each linear layer.
        out_channels : List[int]
            Output channels of each convolutional layer. Should be of length 2.
        classes : int
            Number of classes.
        in_channels : int
            Number of channels of the input data.
        """
        super().__init__()
        kernel_size = 5
        max_pool_kernel = 2

        if len(out_channels) != 2:
            raise ValueError("out_channels should have a length of 2")

        self.layers = nn.ModuleList([])

        self.layers.append(nn.Conv2d(in_channels, out_channels[0], kernel_size))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.MaxPool2d(max_pool_kernel))
        self.layers.append(nn.Conv2d(out_channels[0], out_channels[1], kernel_size))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.MaxPool2d(max_pool_kernel))

        self.layers.append(nn.Flatten())
        self.layers.append(NeuralNetworkModel(classifier_sizes, classes))
