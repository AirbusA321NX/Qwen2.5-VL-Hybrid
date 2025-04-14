import torch
import torch.nn as nn
import torch.optim as optim

class SLTrainer:
    def __init__(self, model, data_loader, config):
        self.model = model
        self.data_loader = data_loader.get_loader()
        self.epochs = config.get("epochs", 50)
        self.lr = config.get("learning_rate", 0.001)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = nn.CrossEntropyLoss()
    
    def train(self):
        self.model.train()
        for epoch in range(self.epochs):
            for data, target in self.data_loader:
                self.optimizer.zero_grad()
                sl_out, _ = self.model(data)
                loss = self.criterion(sl_out, target)
                loss.backward()
                self.optimizer.step()
