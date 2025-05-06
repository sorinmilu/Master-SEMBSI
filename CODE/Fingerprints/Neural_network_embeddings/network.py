import torch
import torch.nn as nn
import torch.nn.functional as F

class SiameseNetwork(nn.Module):
    def __init__(self, embedding_dim=256):
        super(SiameseNetwork, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7),  # Output: [64, 99, 99]
            nn.ReLU(),
            nn.MaxPool2d(2),                 # -> [64, 49, 49]

            nn.Conv2d(64, 128, kernel_size=5),  # -> [128, 45, 45]
            nn.ReLU(),
            nn.MaxPool2d(2),                   # -> [128, 22, 22]

            nn.Conv2d(128, 128, kernel_size=3),  # -> [128, 20, 20]
            nn.ReLU(),
            nn.MaxPool2d(2)                     # -> [128, 10, 10]
        )

        self.fc = nn.Sequential(
            nn.Linear(128 * 10 * 10, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, embedding_dim)  # Final output is the embedding
        )

    def forward_one(self, x):
        x = self.conv(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def forward(self, input1, input2):
        output1 = self.forward_one(input1)
        output2 = self.forward_one(input2)
        return output1, output2
