"""Script for training a LeNet-5 based Network"""
from snapper_ml import job
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score
from .ml.utils.utils import compute_general_loss, default_device, fit
from .ml.utils.snapperml_data_loader import SnapperDataLoader
from .ml.models.convnet import ConvClassifier

SEED = 1234


@job(data_loader_func=SnapperDataLoader)
def main(epochs=10, seed=2342, lr=0.0001, bs=64, out_channels=None):
    """Main function for training a LeNet-5 based model with Snapper-ML"""
    # Set the seed and get the default device for training
    torch.manual_seed(seed)
    dev = default_device()

    if out_channels is None:
        out_channels = [2, 6]

    # Read the data and create the DataLoader
    train_dl, test_dl, data_size = SnapperDataLoader.load_data(dev, bs)

    # Compute the size of the first fully-connected layer
    first_layer = int(((((data_size[3] - 4) / 2) - 4) / 2))
    classifier_sizes = [(first_layer ** 2) * out_channels[1], 100, 50, 10]

    # Initialize the model and load it to the training device
    model = ConvClassifier(classifier_sizes, out_channels, 2, data_size[1])
    model.to(dev)

    # Initialize optimizer
    opt = optim.Adam(model.parameters(), lr)

    # Fit the model to the data
    fit(epochs, model, nn.CrossEntropyLoss(), opt, train_dl)

    # Compute the model losses
    train_acc = compute_general_loss(train_dl, model, accuracy_score)
    test_acc = compute_general_loss(test_dl, model, accuracy_score)

    train_f1 = compute_general_loss(train_dl, model, f1_score)
    test_f1 = compute_general_loss(test_dl, model, f1_score)

    return {
        "train_acc": train_acc,
        "test_acc": test_acc,
        "train_f1": train_f1,
        "test_f1": test_f1,
    }


if __name__ == "__main__":
    main()
