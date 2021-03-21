"""Utils for training models or get the data"""
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import TensorDataset, DataLoader


def validate(
    model: nn.Module, valid_dl: TensorDataset, loss_func: callable, val_length: int
) -> torch.Tensor:
    """Validation of the model"""
    batches_losses = [model.validation_step(batch, loss_func) for batch in valid_dl]
    epoch_loss = torch.stack(batches_losses).sum()

    return epoch_loss / val_length


def fit(
    epochs: int,
    model: nn.Module,
    loss_func: callable,
    opt: Optimizer,
    train_dl: DataLoader,
) -> None:
    """Function for training models"""
    for _ in range(epochs):
        model.train()
        for batch in train_dl:
            loss_batch = model.training_step(batch, loss_func)
            loss_batch.backward()
            opt.step()
            opt.zero_grad()


def default_device() -> torch.device:
    """Use CUDA if available, else CPU"""
    dev = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    return dev
