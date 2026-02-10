import torch.nn as nn

class HeartModel(nn.Module):
    def __init__(self, input_size=13):
        super(HeartModel, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.net(x)
