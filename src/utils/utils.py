"""Utils for training models or get the data"""
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader


def validate(
    model: nn.Module, valid_dl: DataLoader, loss_func: callable
) -> torch.Tensor:
    """Validation of the model"""
    batches_losses = [model.validation_step(batch, loss_func) for batch in valid_dl]
    epoch_loss = torch.cat(batches_losses).mean()

    return epoch_loss


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
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    return dev
