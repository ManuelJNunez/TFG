import pytest
import torch
from torch.utils.data import TensorDataset, DataLoader
from src.ml.utils.device_data_loader import DeviceDataLoader

NUM_SAMPLES = 500
BATCH_SIZE = 64


@pytest.fixture
def data():
    return torch.rand(NUM_SAMPLES, 50), torch.randint(2, (NUM_SAMPLES, 1))


@pytest.fixture
def dataset(data):
    x, y = data
    return TensorDataset(x, y)


@pytest.fixture
def data_loader(dataset):
    return DataLoader(dataset, batch_size=BATCH_SIZE)


@pytest.fixture
def device_data_loader(data_loader):
    return DeviceDataLoader(data_loader, torch.device("cpu"))


def test_initializer(data_loader, device_data_loader):
    assert device_data_loader.data_loader == data_loader
    assert device_data_loader.device == torch.device("cpu")


def test_iter(data_loader, device_data_loader):
    for dl_batch, device_dl_batch in zip(data_loader, device_data_loader):
        x_dl, y_dl = dl_batch
        x_device_dl, y_device_dl = device_dl_batch
        assert (x_dl == x_device_dl).all()
        assert (y_dl == y_device_dl).all()


def test_len(data_loader, device_data_loader):
    assert len(data_loader) == len(device_data_loader)
