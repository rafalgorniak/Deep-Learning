import torch
import numpy as np
from torch_geometric.datasets import MoleculeNet, QM9
from torch_geometric.transforms import TargetIndegree

def compute_class_weights(dataset):
    labels = np.array([int(data.y.item()) for data in dataset])
    count_class0 = np.sum(labels == 0)
    count_class1 = np.sum(labels == 1)

    weight_class0 = 1.0 / count_class0
    weight_class1 = 1.0 / count_class1

    return torch.tensor([weight_class0, weight_class1], dtype=torch.float)


def _get_bace_datasets():
    dataset = MoleculeNet(root="data", name="BACE")

    class_weights = compute_class_weights(dataset)

    # Split dataset into train, validation, and test sets
    torch.manual_seed(42)  # For reproducibility
    train_dataset = dataset[:int(0.8 * len(dataset))]
    val_dataset = dataset[int(0.8 * len(dataset)):int(0.9 * len(dataset))]
    test_dataset = dataset[int(0.9 * len(dataset)):]
    return train_dataset, val_dataset, test_dataset, class_weights


def get_bace_datasets():
    dataset = MoleculeNet(root="data", name="BACE")

    class_weights = compute_class_weights(dataset)
    dataset_len = len(dataset)

    indices = np.random.permutation(dataset_len)
    torch.manual_seed(42)  # For reproducibility

    train_index = indices[:int(0.8 * dataset_len)].tolist()
    val_index = indices[int(0.8 * dataset_len):int(0.9 * dataset_len)].tolist()
    test_index = indices[int(0.9 * dataset_len):].tolist()

    train_dataset = dataset.index_select(train_index)
    val_dataset = dataset.index_select(val_index)
    test_dataset = dataset.index_select(test_index)
    return train_dataset, val_dataset, test_dataset, class_weights


def get_qm9_datasets(transform=None, pre_transform=None):
    dataset = QM9(root="data/QM9", transform=transform, pre_transform=pre_transform)

    dataset_len = len(dataset)

    indices = np.random.permutation(dataset_len).astype(np.int64)
    torch.manual_seed(42)  # For reproducibility

    train_index = indices[:int(0.8 * dataset_len)].tolist()
    val_index = indices[int(0.8 * dataset_len):int(0.9 * dataset_len)].tolist()
    test_index = indices[int(0.9 * dataset_len):].tolist()

    train_dataset = dataset.index_select(train_index)
    val_dataset = dataset.index_select(val_index)
    test_dataset = dataset.index_select(test_index)

    return train_dataset, val_dataset, test_dataset
