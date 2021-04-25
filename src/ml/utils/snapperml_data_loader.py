from snapper_ml import DataLoader
from .utils import read_data
from pathlib import Path


class SnapperDataLoader(DataLoader):
    @classmethod
    def load_data(cls):
        train_path = Path("data/Train_EnergyGround_alt5200m_qgsii_fluka_N44971.h5")
        test_path = Path("data/Test_EnergyGround_alt5200m_qgsii_fluka_N14989.h5")

        train_data, train_labels = read_data(train_path)

        test_data, test_labels = read_data(test_path)

        return train_data, train_labels, test_data, test_labels
