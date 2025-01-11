from torch.utils.data import Dataset
import cv2
import os


class FacesDataset(Dataset):
    def __init__(self, images_dir, labels_file, transform=None):
        self.images_dir = images_dir
        self.transform = transform

        self.image_paths = []
        self.labels = []

        with open(labels_file, 'r') as f:
            for line in f.readlines():
                img_name, label = line.strip().split()
                self.image_paths.append(img_name)
                self.labels.append(int(label))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_name = self.image_paths[idx]
        img_path = str(os.path.join(self.images_dir, img_name))
        image = cv2.imread(img_path)

        if image is None:
            raise FileNotFoundError(f"Nie udało się wczytać obrazu: {img_path}")

        if self.transform:
            image = self.transform(image)

        label = self.labels[idx]

        return image, label
