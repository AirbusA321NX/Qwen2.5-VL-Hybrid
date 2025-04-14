import torch
import torch.optim as optim
from agent_rl import RLAgent

class RLTrainer:
    def __init__(self, model, data_loader, config_rl):
        self.model = model
        self.data_loader = data_loader.get_loader()
        self.epochs = config_rl.get("rl_epochs", 20)
        self.lr = config_rl.get("rl_learning_rate", 0.0001)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.agent = RLAgent(model, config_rl)
    
    def train(self):
        self.model.train()
        for epoch in range(self.epochs):
            for data, target in self.data_loader:
                # Obtain action using the RL head from the current state.
                action = self.agent.get_action(data)
                # Compute a reward based on the action and target
                reward = self.agent.compute_reward(action, target.float().unsqueeze(1))
                loss = -reward.mean()  # Negative reward to maximize actual reward
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
