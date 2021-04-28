"""The snapperml_data_loader contains a DataLoader for SnapperML"""
from pathlib import Path
import snapper_ml
import torch
from torch.utils.data import TensorDataset, DataLoader
from .utils import read_data
from .device_data_loader import DeviceDataLoader


class SnapperDataLoader(snapper_ml.DataLoader):
    """DataLoader for SnapperML, contains the PyTorch's DataLoaders"""
    @classmethod
    def load_data(cls, dev: torch.device, batch_size: int):
        train_path = Path("data/Train_EnergyGround_alt5200m_qgsii_fluka_N44971.h5")
        test_path = Path("data/Test_EnergyGround_alt5200m_qgsii_fluka_N14989.h5")

        train_data, train_labels = read_data(train_path)
        test_data, test_labels = read_data(test_path)

        train_data.unsqueeze_(1)
        test_data.unsqueeze_(1)

        # Create a Tensor Dataset and a Data Loader for the training data
        train_ds = TensorDataset(train_data, train_labels)
        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        train_dl = DeviceDataLoader(train_dl, dev)

        # Create a Tensor Dataset and a Data Loader for the test data
        test_ds = TensorDataset(test_data, test_labels)
        test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=True)
        test_dl = DeviceDataLoader(test_dl, dev)

        return train_dl, test_dl, train_data.size()
