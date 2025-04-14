import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        # Shared layer
        self.fc = nn.Linear(100, 50)
        self.relu = nn.ReLU()
        # Supervised learning head: outputs logits for a 10-class classification
        self.sl_head = nn.Linear(50, 10)
        # Reinforcement learning head: outputs a single value (e.g. for value estimation)
        self.rl_head = nn.Linear(50, 1)
    
    def forward(self, x):
        x = self.relu(self.fc(x))
        sl_out = self.sl_head(x)
        rl_out = self.rl_head(x)
        return sl_out, rl_out
