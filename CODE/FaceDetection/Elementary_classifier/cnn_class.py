import torch
import torch.nn as nn

class CNNClassifier(nn.Module):
    def __init__(self, image_size = 128, embedding_size=512):
        super(CNNClassifier, self).__init__()
        self.image_size = image_size

        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  
        )

        self.conv_output_size = self._get_conv_output_size()
        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.conv_output_size, embedding_size),  # Fully connected layer after conv layers            
            nn.ReLU(),
            nn.Linear(embedding_size, 1),  # Output for binary classification
            nn.Sigmoid()  # Sigmoid activation for binary classification
        )

    def forward(self, x):
        x = self.conv_layers(x)  # Pass through convolutional layers
        x = self.fc_layers(x)  # Pass through fully connected layers
        return x

    def _get_conv_output_size(self):
        # Create a dummy input tensor to calculate the output size after conv layers
        dummy_input = torch.ones(1, 3, self.image_size, self.image_size)  
        x = self.conv_layers(dummy_input)
        return x.numel()  # Return the number of elements in the tensor after conv layers
    