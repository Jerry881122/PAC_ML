import torch
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, feature, label):
        self.feature = feature
        self.label = label

    def __len__(self):
        return len(self.feature)

    def __getitem__(self, index):
        return self.feature[index], self.label[index]