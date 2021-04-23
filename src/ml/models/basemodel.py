"""Base Model for some models"""
import torch
import torch.nn as nn


class BaseModel(nn.Module):
    """
    Base Model from which other classes will inherit for avoiding duplicated code.
    """

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

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """
        Computes the output of the convolutional and fully connected classifier.

        Parameters
        ----------
        data : torch.Tensor
            Data to which we want to classify.

        Returns
        -------
        torch.Tensor
            A tensor whose size is `num samples` x `num clases`. Each value of a row represents
            the probability of beloging to a class.
        """
        next_input = data

        for layer in self.layers:
            next_input = layer(next_input)

        return next_input
