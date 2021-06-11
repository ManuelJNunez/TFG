from snapper_ml import job
from snapper_ml.logging import logger
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch import Tensor
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
from ml.models.cae import ConvAutoencoder


def cae_loss(weight_decay: float):
    def compute_cae_loss(
        cae_output: Tensor, data: Tensor, predictions: Tensor, true_labels: Tensor
    ) -> Tensor:
        lamb = weight_decay
        mse_loss = F.mse_loss(cae_output, data)
        cross_entropy = F.cross_entropy(predictions, true_labels)

        return cross_entropy + lamb * mse_loss

    return compute_cae_loss


@job(data_loader_func=SnapperDataLoader)
def main(
    epochs=10,
    seed=2342,
    lr=0.001,
    norm=0.01,
    bs=128,
    first_conv_out_channels=2,
    depth=3,
    latent_size=10,
    optimizer="adam",
):
    """Main function for training a LeNet-5 based model with Snapper-ML"""
    # Set the seed and get the default device for training
    torch.manual_seed(seed)
    dev = default_device()
    model_path = config.model_path

    logger.info("Reading data...")

    # Read the data and create the DataLoader
    train_dl, test_dl, data_size = SnapperDataLoader.load_data(dev, bs)

    # Compute the size of the first fully-connected layer
    classifier_sizes = [10, 5]

    # Initialize the model and load it to the training device
    model = ConvAutoencoder(
        classifier_sizes, first_conv_out_channels, depth, 2, data_size, latent_size
    )
    model.to(dev)

    # Initialize optimizer
    if optimizer == "adam":
        opt = optim.Adam(model.parameters(), lr)
    elif optimizer == "sgd":
        opt = optim.SGD(model.parameters(), lr)

    logger.info("Training model...")

    # Fit the model to the data
    fit(epochs, model, cae_loss(norm), opt, train_dl)

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
