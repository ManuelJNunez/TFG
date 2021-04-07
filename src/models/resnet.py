"""Residual Network model for 1-dimensional data"""
from typing import List, Optional
import torch
from torch import Tensor
import torch.nn as nn
from .convblocks.resblock import ResBlock
from .neuralnet import NeuralNetworkModel
from .basemodel import BaseModel


class ResNetModel(BaseModel):
    """
    Residual Network implementation for 1-dimensional data.

    Attributes
    ----------
    conv1 : nn.Conv2d
        First convolution of the ResNet which changes the number of channels (2nd dimension) and
        divide by 2 the data length (3rd dimension).
    bn1 : nn.BatchNorm2d
        Normalizes the output of conv1.
    relu : nn.ReLU
        Activation function for the normalized output of conv1.
    layer1 : nn.Sequential
        First ResBlock
    layer2 : nn.Sequential
        Second ResBlock
    layer3 : nn.Sequential
        Third ResBlock
    layer4 : nn.Sequential
        Fourth ResBlock
    avgpool: nn.AdaptativeAvgPool1d
        Pooling layer for the layer4 output
    classifier: NeuralNetworkModel
        Takes the output of the convolutional neural network as input and predicts the class
        of the instance.
    """

    def __init__(
        self,
        num_channels: List[int],
        number_of_blocks: List[int],
        classifier_sizes: List[int],
        in_channels: int,
        classes: int,
    ):
        """
        Residual Network initializer.

        Parameters
        ----------
        num_channels : List[int]
            Number of channels for each group of blocks. Should be a list of length 5.
        number_of_blocks : List[int]
            Number of blocks for each group. Should be a list of length 4.
        classifier_sizes : List[int]
            Layer sizes for the fully connected network.
        in_channels : int
            Number of channels of the input data.
        classes : int
            Number of classes.

        Raises
        ------
        ValueError
            If num_channels does not have a length of 4 or if number_of_blocks doesn't have
            a length of 5.
        """
        super().__init__()

        if len(num_channels) != 5:
            raise ValueError("num_channels should have a length of 5.")
        if len(number_of_blocks) != 4:
            raise ValueError("number_of_blocks should have a length of 4.")

        if classifier_sizes[0] != num_channels[-1]:
            classifier_sizes.insert(0, num_channels[-1])

        self.conv1 = nn.Conv2d(
            in_channels, num_channels[0], kernel_size=7, stride=2, padding=3
        )
        self.bn1 = nn.BatchNorm2d(num_channels[0])
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.layer1 = self._create_layer(
            num_channels[0], num_channels[1], number_of_blocks[0]
        )
        self.layer2 = self._create_layer(
            num_channels[1], num_channels[2], number_of_blocks[1], stride=2
        )
        self.layer3 = self._create_layer(
            num_channels[2], num_channels[3], number_of_blocks[2], stride=2
        )
        self.layer4 = self._create_layer(
            num_channels[3], num_channels[4], number_of_blocks[3], stride=2
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = NeuralNetworkModel(classifier_sizes, classes)

    # pylint: disable=R0201
    def _create_layer(
        self,
        in_channels: int,
        out_channels: int,
        blocks: int,
        stride: Optional[int] = 1,
    ) -> nn.Sequential:
        """
        Creates a group of ResBlocks.

        Parameters
        ----------
        in_channels : int
            Number of channels of the group's input.
        out_channels : int
            Number of channels of the group's output.
        blocks : int
            Number of ResBlocks in the new created group.
        stride : int
            Stride for the first ResBlock.

        Returns
        -------
        nn.Sequential
            Contains the layers of the ResNet group.
        """
        downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, stride),
            nn.BatchNorm2d(out_channels),
        )

        layers = []
        layers.append(ResBlock(in_channels, out_channels, stride, downsample))

        for _ in range(1, blocks):
            layers.append(ResBlock(out_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, data: Tensor) -> Tensor:
        """
        Given a batch of `data`, computes the output of the Residual Network.

        Parameters
        ----------
        data : Tensor
            Data which you need to classify.

        Returns
        -------
        Tensor
            Output of the Residual Network.
        """
        out = self.conv1(data)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.classifier(out)

        return out
