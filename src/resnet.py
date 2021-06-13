"""Script for training a ResNet based Network"""
from snapper_ml import job
from snapper_ml.logging import logger
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score
import config
from ml.utils.utils import (
    compute_general_loss,
    default_device,
    fit,
    generate_confusion_matrix,
    save_confusion_matrix,
)
from ml.utils.snapperml_data_loader import SnapperDataLoader
from ml.models.resnet import ResNetModel


@job(data_loader_func=SnapperDataLoader)
def main(
    epochs=5,
    seed=2342,
    lr=0.001,
    reg=0.01,
    bs=128,
    num_channels=None,
    number_of_blocks=None,
    optimizer="adam",
):
    """Main function for training a ResNet based model with Snapper-ML"""
    # Set the seed and get the default device for training
    torch.manual_seed(seed)
    dev = default_device()
    model_path = config.model_path

    if num_channels is None:
        num_channels = [64, 64, 128, 256, 512]

    if number_of_blocks is None:
        number_of_blocks = [3, 4, 6, 3]

    logger.info("Reading data...")

    # Read the data and create the DataLoader
    train_dl, test_dl, data_size = SnapperDataLoader.load_data(dev, bs)

    # Compute the size of the first fully-connected layer
    classifier_sizes = [200, 100, 10]

    # Initialize the model and load it to the training device
    model = ResNetModel(
        num_channels, number_of_blocks, classifier_sizes, data_size[1], 2
    )
    model.to(dev)

    # Initialize optimizer
    if optimizer == "adam":
        opt = optim.Adam(model.parameters(), lr, weight_decay=reg)
    elif optimizer == "sgd":
        opt = optim.SGD(model.parameters(), lr, weight_decay=reg)

    logger.info("Training model...")

    # Fit the model to the data
    fit(epochs, model, nn.CrossEntropyLoss(), opt, train_dl)

    logger.info("Computing model loss...")

    # Compute the model losses
    train_acc = compute_general_loss(train_dl, model, accuracy_score)
    test_acc = compute_general_loss(test_dl, model, accuracy_score)

    train_f1 = compute_general_loss(train_dl, model, f1_score)
    test_f1 = compute_general_loss(test_dl, model, f1_score)

    logger.info("Generating confusion matrix...")

    train_cmatrix = generate_confusion_matrix(train_dl, model)
    train_cm_file = config.artifacts_path.joinpath("train_confusion_matrix.png")
    save_confusion_matrix(
        train_cmatrix,
        config.class_names,
        train_cm_file,
    )

    test_cmatrix = generate_confusion_matrix(test_dl, model)
    test_cm_file = config.artifacts_path.joinpath("test_confusion_matrix.png")
    save_confusion_matrix(
        test_cmatrix,
        config.class_names,
        test_cm_file,
    )

    logger.info("Generating artifacts...")

    # Save the model
    torch.save(model.state_dict(), model_path)

    return {
        "train_acc": train_acc,
        "test_acc": test_acc,
        "train_f1": train_f1,
        "test_f1": test_f1,
    }, {
        "model": str(model_path),
        "train_confusion_matrix": str(train_cm_file),
        "test_confusion_matrix": str(test_cm_file),
    }


if __name__ == "__main__":
    main()
