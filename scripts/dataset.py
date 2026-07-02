import torch
from torch.utils.data import Dataset
import numpy as np
import glob

class MotionNetDataset(Dataset):
    def __init__(self, folder="data/sequences/"):
        self.files = sorted(glob.glob(f"{folder}/*.npy"))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # Carrega a sequência (Time, H, W)
        data = np.load(self.files[idx])
        # Adiciona dimensão de canal: (Time, Channel, H, W) -> (5, 1, 200, 200)
        return torch.tensor(data, dtype=torch.float32).unsqueeze(1) 
