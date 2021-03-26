import pytest
import torch
from torch.utils.data import TensorDataset, DataLoader
from src.utils.utils import validate, fit, default_device
from src.models.neuralnet import NeuralNetworkModel
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class FakeModel(nn.Module):
    def validation_step(self, batch, loss_func):
        return loss_func(batch[1])


NUM_SAMPLES = 500
NUM_FEATURES = 2
BATCH_SIZE = 64


@pytest.fixture
def data():
    data = torch.rand(NUM_SAMPLES, NUM_FEATURES)
    labels = torch.zeros(NUM_SAMPLES)

    labels[data[:, 0] > 0.5] = 1

    labels = labels.type(torch.LongTensor)

    return data, labels


@pytest.fixture
def dataset(data):
    data, labels = data
    return TensorDataset(data, labels)


@pytest.fixture
def data_loader(dataset):
    return DataLoader(dataset, batch_size=BATCH_SIZE)


def test_validate(data_loader, data):
    model = FakeModel()
    _, labels = data

    labels_float = labels.type(torch.FloatTensor)

    def loss_func(batch):
        return batch.sum()

    val_loss = validate(model, data_loader, loss_func, len(data[1]))

    assert val_loss == labels_float.mean()


def test_fit(data_loader, data):
    EPOCHS = 3
    features, labels = data

    # Compute CEloss before training
    model = NeuralNetworkModel([features.size(1)], 2)
    outputs_before = model(features)
    loss_before = F.cross_entropy(outputs_before, labels)

    # Fit the model
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    loss_func = nn.CrossEntropyLoss()
    fit(EPOCHS, model, loss_func, optimizer, data_loader)

    # Compute CEloss after training
    outputs_after = model(features)
    loss_after = F.cross_entropy(outputs_after, labels)

    assert loss_before > loss_after


def test_default_device_cuda(monkeypatch):
    def is_available():
        return True

    monkeypatch.setattr(torch.cuda, "is_available", is_available)

    device = default_device()

    assert device == torch.device("cuda")


def test_default_device_cpu(monkeypatch):
    def is_available():
        return False

    monkeypatch.setattr(torch.cuda, "is_available", is_available)

    device = default_device()

    assert device == torch.device("cpu")
