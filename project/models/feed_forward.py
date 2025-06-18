import torch.nn as nn

class FeedForward(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.Tanh()
        )

        self.out = nn.Linear(128, output_dim)

    def forward(self, input):
        x = self.norm(input)
        x = self.shared(x)
        return self.out(x)
    