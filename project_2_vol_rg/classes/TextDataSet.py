import torch
from torch.utils.data import Dataset

class TextDataSet(Dataset):
    def __init__(self, encoded_data_list: list, sequence_length: int):
        self.encoded_data_list: list = encoded_data_list
        self.sequence_length: int = sequence_length

    def __len__(self) -> int:
        return len(self.encoded_data_list) - self.sequence_length

    def __getitem__(self, index):
        x = self.encoded_data_list[index : index + self.sequence_length]
        y = self.encoded_data_list[index + 1 : index + 1 + self.sequence_length]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)