"""Neural Network Model classifier implementation using PyTorch"""
import torch
import torch.nn as nn
from .basemodel import BaseModel


class NeuralNetworkModel(BaseModel):
    """
    Neural Network Model Class with variable number of layers. It uses batch normalization and
    softmax.

    Attributes
    ----------
    layer_sizes : [int]
        This list contains the size of each layer.
    output_units : int
        The number of output units of the neural network, if the number of classes is 3, it should
        be 3 too.
    layer_list : nn.ModuleList
        The list of layers composing the network.
    """

    def __init__(self, layer_sizes: [int], output_units: int):
        """
        Initializes a new Neural Network classifier with the given layer sizes and output units.

        Parameters
        ----------
        layer_sizes : [int]
            This list contains the size of each layer of the network.
        output_units : int
            Number of output units of the neural network. If the number of classes is 3, it should
            be 3 too.
        """
        super().__init__()
        self.layer_sizes = layer_sizes
        self.output_units = output_units
        self.layers = nn.ModuleList([])

        for i in range(len(layer_sizes) - 1):
            input_size = layer_sizes[i]
            output_size = layer_sizes[i + 1]
            self.layers.append(
                nn.Linear(in_features=input_size, out_features=output_size)
            )
            self.layers.append(nn.BatchNorm1d(num_features=output_size))
            self.layers.append(nn.ReLU())

        self.layers.append(nn.Linear(layer_sizes[-1], output_units))
        self.layers.append(nn.Softmax(dim=1))

    def validation_step(self, batch: torch.Tensor, loss_func: callable) -> torch.Tensor:
        """
        This method computes the validation batch loss.

        Parameters
        ----------
        batch : torch.Tensor
            A tensor with a portion of data.
        loss_func : callable
            Function that computes the training loss in each batch of data.

        Returns
        -------
        torch.Tensor
            The result of the loss functions for the batch of data, the output size depends on
            the used loss function.
        """
        data, label = batch
        pred = self(data)
        loss = loss_func(pred, label)

        return loss
