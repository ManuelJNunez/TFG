"""Utils for training models or get the data"""
from typing import Callable, Tuple, List
from pathlib import Path
from sklearn.metrics import confusion_matrix
import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import Optimizer
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import h5py
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from .device_data_loader import DeviceDataLoader


def validate(
    model: nn.Module, valid_dl: TensorDataset, loss_func: callable, val_length: int
) -> Tensor:
    """Validation of the model"""
    batches_losses = [model.validation_step(batch, loss_func) for batch in valid_dl]
    epoch_loss = torch.stack(batches_losses).sum()

    return epoch_loss / val_length


def compute_batch_loss(
    model: nn.Module,
    loss_func: Callable,
    data: Tensor,
    labels: Tensor,
) -> Tuple[Tensor]:
    """Compute the loss for a given batch"""
    pred_probs = model(data)

    if isinstance(pred_probs, tuple):
        pred_probs = pred_probs[1]

    pred_labels = pred_probs.argmax(dim=1)

    pred_labels = pred_labels.cpu()
    true_labels = labels.cpu()

    loss = loss_func(true_labels, pred_labels)

    return loss.item(), len(data)


def compute_general_loss(
    data_loader: DeviceDataLoader,
    model: nn.Module,
    loss_func: Callable,
) -> float:
    """Compute the loss in the entire DataSet"""

    losses, lens = zip(
        *[compute_batch_loss(model, loss_func, xb, yb) for xb, yb in data_loader]
    )

    train_acc = np.sum(np.multiply(losses, lens)) / np.sum(lens)

    return train_acc


def fit(
    epochs: int,
    model: nn.Module,
    loss_func: callable,
    opt: Optimizer,
    train_dl: DataLoader,
) -> None:
    """Function for training models"""
    model.train()
    for _ in range(epochs):
        for batch in train_dl:
            loss_batch = model.training_step(batch, loss_func)
            loss_batch.backward()
            opt.step()
            opt.zero_grad()

    model.eval()


def default_device() -> torch.device:
    """Use CUDA if available, else CPU"""
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    return dev


def read_data(path: Path) -> Tensor:
    """Read the dataset from the given path"""
    with h5py.File(path, "r") as file:
        data = torch.from_numpy(np.array(file["data"]))
        info = pd.read_hdf(path, key="info")
        labels = torch.from_numpy(info.loc[:, "Y_class"].values)

    return data, labels.long()


def generate_confusion_matrix(data_loader: DataLoader, model: nn.Module) -> str:
    """Generate the confusion matrix"""
    labels_pred = np.array([])
    labels_true = np.array([])

    for data, labels in data_loader:
        labels_true = np.append(labels_true, labels.cpu().detach().numpy())

        pred_prob = model(data)
        
        if isinstance(pred_prob, tuple):
            pred_prob = pred_prob[1]
        
        pred_prob = pred_prob.cpu().detach().numpy()
        labels_pred = np.append(labels_pred, np.argmax(pred_prob, axis=1))

    cmatrix = confusion_matrix(labels_true, labels_pred, normalize="true")

    return cmatrix


def save_confusion_matrix(cm_values: List, class_names: str, dest_dir: Path):
    """Saves the computed confusion matrix as a image in the dest_dir"""
    plt.figure()
    df_cm = pd.DataFrame(cm_values, index=class_names, columns=class_names)
    sn_plot = sn.heatmap(df_cm, annot=True)
    fig = sn_plot.get_figure()
    fig.savefig(dest_dir)
