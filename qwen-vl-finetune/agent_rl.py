import torch
import torch.nn.functional as F

class RLAgent:
    def __init__(self, model, config_rl):
        self.model = model
        self.reward_scaling = config_rl.get("rl_reward_scaling", 1.0)
    
    def get_action(self, state):
        # Use the RL head to produce an action (activation via tanh)
        with torch.no_grad():
            _, rl_out = self.model(state)
            action = torch.tanh(rl_out)
        return action
    
    def compute_reward(self, prediction, target):
        # Compute a negative MSE as reward (scaled by a factor)
        reward = -F.mse_loss(prediction, target)
        return reward * self.reward_scaling
