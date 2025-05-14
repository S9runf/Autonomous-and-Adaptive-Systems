import torch
import torch.nn as nn
from torch.distributions import Categorical

class ActorCritic(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh()
        )

        self.out = nn.Linear(64, output_dim)

    def forward(self, input):
        x = self.shared(input)
        return self.out(x)  
        
"""     def get_action(self, logits):
        probs = Categorical(logits=logits)
        action = probs.sample()
        return action.item(), probs.log_prob(action)
    
    def action_distribution(self, logits):
        probs = Categorical(logits=logits)
        return probs """
    