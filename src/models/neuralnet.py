"""Neural Network Model classifier implementation using PyTorch"""
import torch
import torch.nn as nn


class NeuralNetworkModel(nn.Module):
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
        self.layer_list = nn.ModuleList([])

        for i in range(len(layer_sizes) - 1):
            input_size = layer_sizes[i]
            output_size = layer_sizes[i + 1]
            self.layer_list.append(
                nn.Linear(in_features=input_size, out_features=output_size)
            )
            self.layer_list.append(nn.BatchNorm1d(num_features=output_size))
            self.layer_list.append(nn.ReLU())

        self.layer_list.append(nn.Linear(layer_sizes[-1], output_units))
        self.layer_list.append(nn.Softmax(dim=1))

    def forward(self, classifier_input: torch.Tensor) -> torch.Tensor:
        """
        Computes the output of the classifier.

        Parameters
        ----------
        classifier_input : torch.Tensor
            Data to which we want to classify.

        Returns
        -------
        torch.Tensor
            A tensor whose size is `num samples` x `num clases`. Each value of a row represents
            the probability of beloging to a class.
        """
        layer_output = classifier_input

        for layer in self.layer_list:
            layer_output = layer(layer_output)

        return layer_output

    def training_step(self, batch: torch.Tensor, loss_func: callable) -> torch.Tensor:
        """
        This method computes the training batch loss.

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
