import random

import numpy as np
import torch


def train_val_test_split(dataset, val_ratio=0.1, test_ratio=0.1, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    n = len(dataset)
    indices = np.random.permutation(n).astype(np.int64)

    n_train = int((1 - val_ratio - test_ratio) * n)
    n_val = int(val_ratio * n)

    train_index = indices[:n_train].tolist()
    val_index = indices[n_train:n_train + n_val].tolist()
    test_index = indices[n_train + n_val:].tolist()

    train_dataset = dataset.index_select(train_index)
    val_dataset = dataset.index_select(val_index)
    test_dataset = dataset.index_select(test_index)

    return train_dataset, val_dataset, test_dataset


def compute_class_weights(dataset):
    labels = np.array([int(data.y.item()) for data in dataset])
    count_class0 = np.sum(labels == 0)
    count_class1 = np.sum(labels == 1)

    weight_class0 = 1.0 / count_class0
    weight_class1 = 1.0 / count_class1

    return torch.tensor([weight_class0, weight_class1], dtype=torch.float)