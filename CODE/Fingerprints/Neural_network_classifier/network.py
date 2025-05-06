import torch
import torch.nn as nn

class FingerprintClassifier(nn.Module):
    def __init__(self, image_size=128, embedding_size=512, num_classes=3):
        super(FingerprintClassifier, self).__init__()
        self.image_size = image_size

        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),  # 1 input channel for grayscale
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.conv_output_size = self._get_conv_output_size()

        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.conv_output_size, embedding_size),
            nn.ReLU(),
            nn.Linear(embedding_size, num_classes)  # No sigmoid for multi-class logits
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

    def _get_conv_output_size(self):
        dummy_input = torch.ones(1, 1, self.image_size, self.image_size)  # 1 channel for grayscale
        x = self.conv_layers(dummy_input)
        return x.numel()
