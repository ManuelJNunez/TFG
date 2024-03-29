"""Data Loader Wrapped class to send model and dataset to the training device (CPU OR CUDA)"""
from typing import Iterable, List
import torch
from torch.utils.data import DataLoader


class DeviceDataLoader:
    """
    This is a wrapped DataLoader used for loading the data to training device (CPU or CUDA)
    """

    def __init__(self, data_loader: DataLoader, device: torch.device):
        """
        Initializes the DeviceDataLoader object with the data and the device where this class
        will load it.

        Parameters:
            data_loader: the original data loader that contains all the data distributed in batches.
            device: the device we will load the data to.
        """
        self.data_loader = data_loader
        self.device = device

    def __iter__(self) -> Iterable[List[torch.Tensor]]:
        """
        This method is for iterating over the batches while it loads them to the training device.
        """
        for batch in self.data_loader:
            data, label = batch
            yield [data.to(self.device), label.to(self.device)]

    def __len__(self) -> int:
        """
        This method retrieves the DataLoader's length
        """
        return len(self.data_loader)
