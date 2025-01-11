import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

from project_1.classes.FacesDataset import FacesDataset


def load_celeba_dataset(data_dir, batch_size=32, image_size=64, split='train', subset_size=None):
    # Transformacje dla obrazów, zmiana rozdzielczości i jakaś normalizacja do wartości [-1,1]

    if split == 'train':
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomVerticalFlip(0.5),
            transforms.RandomRotation((-20, 20)),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    else:
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
        download=False
    )

    if subset_size is not None:
        dataset = reduce_data_number(dataset, subset_size)

    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return data_loader


def load_celeba_pretrained_dataset(data_dir, batch_size=32, image_size=224, split='train', subset_size=None):
    # Transformacje dla obrazów, zmiana rozdzielczości i jakaś normalizacja do wartości [-1,1]

    if split == 'train':
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomVerticalFlip(0.5),
            transforms.RandomRotation((-20, 20)),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def target_transform(target):
        return target[7].item()

    # Pobieranie zbioru CelebA, możliwe 3 opcje: train, test oraz validacyjny
    dataset = datasets.CelebA(
        root=data_dir,
        split=split,
        transform=transform,
        target_transform=target_transform,
        download=False
    )

    if subset_size is not None:
        dataset = reduce_data_number(dataset, subset_size)

    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return data_loader


def load_wiederface_dataset():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    images_dir = "./data/widerFace/faces"
    labels_file = "./data/widerFace/faces_gender_labels.txt"
    dataset = FacesDataset(images_dir=images_dir, labels_file=labels_file, transform=transform)

    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    return dataloader


def load_wiederface_pretrained_dataset():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    images_dir = "./data/widerFace/faces"
    labels_file = "./data/widerFace/faces_nose_labels.txt"
    dataset = FacesDataset(images_dir=images_dir, labels_file=labels_file, transform=transform)

    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    return dataloader


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
        train_loader = None
        valid_loader = None
        test_loader = load_wiederface_dataset()
    elif dataset == 'CelebA_pretrained':
        train_loader = load_celeba_pretrained_dataset(data_dir="./data", batch_size=64, split='train', subset_size=subset_size)
        valid_loader = load_celeba_pretrained_dataset(data_dir="./data", batch_size=64, split='valid', subset_size=subset_size)
        test_loader = load_celeba_pretrained_dataset(data_dir="./data", batch_size=64, split='test', subset_size=subset_size)
    elif dataset == 'WIDERFace_pretrained':
        train_loader = None
        valid_loader = None
        test_loader = load_wiederface_pretrained_dataset()
    else:
        raise NotImplementedError

    return train_loader, valid_loader, test_loader
