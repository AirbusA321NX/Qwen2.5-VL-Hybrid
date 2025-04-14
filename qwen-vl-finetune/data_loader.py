import torch
from torch.utils.data import DataLoader as TorchDataLoader, TensorDataset
import numpy as np

class DataLoader:
    def __init__(self, config):
        self.batch_size = config.get("batch_size", 32)
        # Generate dummy features and labels for training
        x = np.random.randn(1000, 100).astype("float32")
        y = np.random.randint(0, 10, size=(1000,)).astype("int64")
        self.dataset = TensorDataset(torch.tensor(x), torch.tensor(y))
    
    def get_loader(self):
        return TorchDataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)
