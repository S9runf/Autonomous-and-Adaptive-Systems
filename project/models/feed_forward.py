import torch
import torch.nn as nn
from torch.distributions import Categorical

class FeedForward(nn.Module):
    def __init__(self, input_dim, output_dim, critic=False):
        super().__init__()
        self.max_distance = 5.0
        self.critic = critic
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh()
        )

        self.out = nn.Linear(64, output_dim)

    def forward(self, input):
        x = self.shared(input)
        return self.out(x)  
    