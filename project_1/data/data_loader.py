from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def load_celeba_dataset(data_dir, batch_size=32, image_size=64, split='train'):

    # Transformacje dla obrazów, zmiana rozdzielczości i jakaś normalizacja do wartości [-1,1]
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])



    # Pobieranie zbioru CelebA, możliwe 3 opcje: train, test oraz validacyjny
    dataset = datasets.CelebA(
        root=data_dir,
        split=split,
        transform=transform,
        download=True
    )

    # Tworzenie DataLoadera, ustaiwanie rozmiaru paczek oraz tasowanie
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return data_loader


def load_wiederface_dataset(data_dir, batch_size=32, image_size=64, split='train'):

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

    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return data_loader


def load_dataset(dataset):
    if dataset == 'CelebA':
        train_loader = load_celeba_dataset(data_dir="./data", batch_size=64, split='train')
        valid_loader = load_celeba_dataset(data_dir="./data", batch_size=64, split='valid')
        test_loader = load_celeba_dataset(data_dir="./data", batch_size=64, split='test')
    elif dataset == 'WIDERFace':
        train_loader = load_wiederface_dataset(data_dir="./data", batch_size=64, split='train')
        valid_loader = load_wiederface_dataset(data_dir="./data", batch_size=64, split='val')
        test_loader = load_wiederface_dataset(data_dir="./data", batch_size=64, split='test')
    else:
        raise NotImplementedError

    return train_loader, valid_loader, test_loader
