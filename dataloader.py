from torch.utils.data import Dataset, DataLoader
import torch

class CreditcardData(Dataset):
    def __init__(self, features, labels):
        self.x = torch.tensor(features, dtype=torch.float32)
        self.y = torch.tensor(labels, dtype=torch.float32)
        self.length = self.x.shape[0]


    def __getitem__(self,idx):
        return self.x[idx], self.y[idx]

    def __len__(self):
        return self.length
