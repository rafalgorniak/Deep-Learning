import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset


def load_celeba_dataset(data_dir, batch_size=32, image_size=64, split='train', subset_size=None):
    # Transformacje dla obrazów, zmiana rozdzielczości i jakaś normalizacja do wartości [-1,1]
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    def target_transform(target):
        return target[20].item()

    # Pobieranie zbioru CelebA, możliwe 3 opcje: train, test oraz validacyjny
    dataset = datasets.CelebA(
        root=data_dir,
        split=split,
        transform=transform,
        target_transform=target_transform,
        download=True
    )

    if subset_size is not None:
        dataset = reduce_data_number(dataset, subset_size)

    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return data_loader


def load_wiederface_dataset(data_dir, batch_size=32, image_size=64, split='train', subset_size=None):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    dataset = datasets.WIDERFace(
        root=data_dir,
        split=split,
        target_transform=transform,
        download=True
    )

    if subset_size is not None:
        dataset = reduce_data_number(dataset, subset_size)

    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    return data_loader


def reduce_data_number(data_set, subset_size):
    total_samples = len(data_set)
    subset_size = int(total_samples * subset_size)

    selected_indices = np.random.choice(total_samples, subset_size, replace=False)
    reduced_subset = Subset(data_set, selected_indices)

    return reduced_subset


def load_dataset(dataset, subset_size=1):
    if dataset == 'CelebA':
        train_loader = load_celeba_dataset(data_dir="./data", batch_size=64, split='train', subset_size=subset_size)
        valid_loader = load_celeba_dataset(data_dir="./data", batch_size=64, split='valid', subset_size=subset_size)
        test_loader = load_celeba_dataset(data_dir="./data", batch_size=64, split='test', subset_size=subset_size)
    elif dataset == 'WIDERFace':
        train_loader = load_wiederface_dataset(data_dir="./data", batch_size=64, split='train', subset_size=subset_size)
        valid_loader = load_wiederface_dataset(data_dir="./data", batch_size=64, split='val', subset_size=subset_size)
        test_loader = load_wiederface_dataset(data_dir="./data", batch_size=64, split='test', subset_size=subset_size)
    else:
        raise NotImplementedError

    return train_loader, valid_loader, test_loader

