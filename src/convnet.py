from snapper_ml import job
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from ml.utils.utils import read_data, default_device, fit
from ml.utils.device_data_loader import DeviceDataLoader
from ml.utils.snapperml_data_loader import SnapperDataLoader
from ml.models.convnet import ConvClassifier
from torch.utils.data import TensorDataset, DataLoader


@job(data_loader_func=SnapperDataLoader)
def main(epochs=10, seed=2342, lr=0.0001, bs=64, out_channels=[2, 6]):
    torch.manual_seed(seed)

    train_data, train_labels, test_data, test_labels = SnapperDataLoader.load_data()

    # Process data and load to the default device
    dev = default_device()
    train_data.unsqueeze_(1)
    train_data = train_data.to(dev)
    train_labels = train_labels.to(dev)

    # Create a Tensor Dataset and a Data Loader
    train_ds = TensorDataset(train_data, train_labels)
    train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True)

    first_layer = int(((((train_data.size(3) - 4) / 2) - 4) / 2))
    classifier_sizes = [(first_layer ** 2) * out_channels[1], 100, 50, 10]

    model = ConvClassifier(classifier_sizes, out_channels, 2, train_data.size(1))

    model.to(dev)
    opt = optim.Adam(model.parameters())

    fit(epochs, model, nn.CrossEntropyLoss(), opt, train_dl)

    train_acc = (model(train_data).argmax(dim=1) == train_labels).float().mean()

    print(train_acc)

    return {"train_acc": train_acc.item()}


if __name__ == "__main__":
    main()
