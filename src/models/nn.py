"""Neural Network Model implementation using PyTorch"""
import torch
import torch.nn as nn


class NeuralNetworkModel(nn.Module):
    """Neural Network Model Class with variable number of layers"""

    def __init__(self, layer_sizes: [int], output_units: int):
        """Class inatializer with layer sizes and output_units as parameters"""
        super().__init__()
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

    def forward(self, fordward_input: torch.Tensor) -> torch.Tensor:
        """Neural Network Forward Pass"""
        layer_output = fordward_input

        for layer in self.layer_list:
            layer_output = layer(layer_output)

        return layer_output

    def training_step(self, batch: torch.Tensor, loss_func: callable) -> torch.Tensor:
        """Neural Network training step (calculates batch loss)"""
        data, label = batch
        pred = self(data)
        loss = loss_func(pred, label)

        return loss

    def validation_step(self, batch: torch.Tensor, loss_func: callable) -> torch.Tensor:
        """Neural Network validation step (calculates batch loss for validation)"""
        data, label = batch
        pred = self(data)
        loss = loss_func(pred, label)

        return loss
