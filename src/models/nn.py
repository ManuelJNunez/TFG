"""Neural Network implementation using PyTorch"""
import torch.nn as nn

class NeuralNetworkModel(nn.Module):
    """Neural Network Model Class"""

    def __init__(self, layer_sizes: [int], output_units: int):
        """Class inatializer with layer sizes and output_units as parameters"""
        super().__init__()
        layers = []

        for i in range(len(layer_sizes)-1):
            input_size = layer_sizes[i]
            output_size = layer_sizes[i+1]
            layers.append(nn.Linear(in_features=input_size, out_features=output_size))
            layers.append(nn.BatchNorm1d(num_features=output_size))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(layer_sizes[-1], output_units))
        layers.append(nn.Softmax(dim=1))

        self.layer_list = nn.ModuleList(layers)

    def forward(self, fordward_input):
        """Neural Network Forward Pass"""
        layer_output = fordward_input

        for layer in self.layer_list:
            layer_output = layer(layer_output)

        return layer_output

    def training_step(self, batch, loss_func):
        """Neural Network training step (calculates batch loss)"""
        data, label = batch
        pred = self(data)
        loss = loss_func(pred, label)

        return loss

    def validation_step(self, batch, loss_func):
        """Neural Network validation step (calculates batch loss for validation)"""
        data, label = batch
        pred = self(data)
        loss = loss_func(pred, label)

        return loss
