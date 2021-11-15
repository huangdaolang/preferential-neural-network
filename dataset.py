from torch.utils.data import Dataset
import torch


class pref_dataset(Dataset):
    def __init__(self, x_duels, pref):
        self.x_duels = x_duels
        self.pref = pref

    def __getitem__(self, index):
        entry = {"x1": torch.tensor([self.x_duels[index][0]]),
                 "x2": torch.tensor([self.x_duels[index][1]]),
                 "pref": self.pref[index]}
        return entry

    def __len__(self):
        return len(self.pref)


class inducing_dataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __getitem__(self, index):
        entry = {"x": torch.tensor([self.x[index]]),
                 "y": self.y[index]}
        return entry

    def __len__(self):
        return len(self.y)
