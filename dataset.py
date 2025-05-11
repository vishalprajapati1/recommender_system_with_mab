import torch
from torch.utils.data import Dataset


class MovieLensDataset(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        user, item, rating = self.data[idx]
        return (torch.tensor(user, dtype=torch.long), torch.tensor(item, dtype=torch.long),
                torch.tensor(rating, dtype=torch.float))
