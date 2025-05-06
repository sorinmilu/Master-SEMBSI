import torch
import torch.nn as nn
import torch.nn.functional as F

class GridCNNModel(nn.Module):
    def __init__(self, griddim=7, num_classes=1, image_size = 224):
        super(GridCNNModel, self).__init__()
        self.griddim = griddim
        self.num_classes = num_classes
        self.image_size = image_size

        # Define the convolutional layers in a Sequential block
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3), nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 192, kernel_size=3, padding=1), nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(192, 128, kernel_size=1), nn.LeakyReLU(0.1),
            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.LeakyReLU(0.1),
            nn.Conv2d(256, 256, kernel_size=1), nn.LeakyReLU(0.1),
            nn.Conv2d(256, 512, kernel_size=3, padding=1), nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Calculate the output size after the convolutional layers
        self.conv_output_size = self._get_conv_output_size()

        # Define the fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.conv_output_size, 4096),  # Fully connected layer after conv layers
            nn.ReLU(),
            nn.Linear(4096, self.griddim ** 2),  # Output grid cells
            nn.Sigmoid()  # Sigmoid to output values between 0 and 1
        )

    def forward(self, x):
        # Apply the convolutional layers
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        x = x.view(x.size(0), self.griddim, self.griddim)
        return x

    # def _get_conv_output_size(self):
    #     # Create a dummy input tensor to calculate the output size after conv layers
    #     dummy_input = torch.ones(1, 3, 224, 224)  # Batch size 1, 3 channels, 224x224 image
    #     x = self.conv_layers(dummy_input)
    #     return x.numel()  # Return the number of elements in the tensor after conv layers

    def _get_conv_output_size(self):
            # Create a dummy input tensor to calculate the output size after conv layers
            dummy_input = torch.ones(1, 3, self.image_size, self.image_size)  # Batch size 1, 3 channels, 224x224 image
            x = self.conv_layers(dummy_input)
            return x.numel()  # Return the number of elements in the tensor after conv layers


class GridSimpleCNNModel(nn.Module):
    def __init__(self, griddim=7, num_classes=1, image_size = 224):
        super(GridSimpleCNNModel, self).__init__()
        self.griddim = griddim
        self.num_classes = num_classes
        self.image_size = image_size

        # Define the convolutional layers in a Sequential block
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3), nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: (64, 56, 56)
            nn.Conv2d(64, 192, kernel_size=3, padding=1), nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: (192, 28, 28)
            nn.Conv2d(192, 256, kernel_size=3, padding=1), nn.LeakyReLU(0.1),  # Output: (256, 28, 28)
            nn.Conv2d(256, 512, kernel_size=3, padding=1), nn.LeakyReLU(0.1),  # Output: (512, 28, 28)
            nn.MaxPool2d(kernel_size=2, stride=2)  # Output: (512, 14, 14)
        )

        # Calculate the output size after the convolutional layers
        self.conv_output_size = self._get_conv_output_size()

        # Define the fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.conv_output_size, 4096),  # Fully connected layer after conv layers
            nn.ReLU(),
            nn.Linear(4096, self.griddim ** 2),  # Output grid cells
            nn.Sigmoid()  # Sigmoid to output values between 0 and 1
        )

    def forward(self, x):
        # Apply the convolutional layers
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        x = x.view(x.size(0), self.griddim, self.griddim)
        return x

    def _get_conv_output_size(self):
            # Create a dummy input tensor to calculate the output size after conv layers
            dummy_input = torch.ones(1, 3, self.image_size, self.image_size)  # Batch size 1, 3 channels, 224x224 image
            x = self.conv_layers(dummy_input)
            return x.numel()  # Return the number of elements in the tensor after conv layers


class GridYOLOCNNModel(nn.Module):
    def __init__(self, griddim=7, num_classes=1, image_size = 448):
        super(GridYOLOCNNModel, self).__init__()
        self.griddim = griddim
        self.num_classes = num_classes
        self.image_size = image_size

        # Define the convolutional layers in a Sequential block
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3), nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: (64, 112, 112)
            nn.Conv2d(64, 192, kernel_size=3, padding=1), nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: (192, 56, 56)
            nn.Conv2d(192, 256, kernel_size=3, padding=1), nn.LeakyReLU(0.1),
            nn.Conv2d(256, 512, kernel_size=3, padding=1), nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: (512, 28, 28)
            nn.Conv2d(512, 1024, kernel_size=3, padding=1), nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: (1024, 14, 14)
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1), nn.LeakyReLU(0.1),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1), nn.LeakyReLU(0.1),  # Output: (1024, 7, 7)
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1), nn.LeakyReLU(0.1),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1), nn.LeakyReLU(0.1),
        )

        # Calculate the output size after the convolutional layers
        self.conv_output_size = self._get_conv_output_size()
        print(self.conv_output_size)

        # Define the fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.conv_output_size, 4096),  # Fully connected layer after conv layers
            nn.ReLU(),
            nn.Linear(4096, self.griddim ** 2),  # Output grid cells
            nn.Sigmoid()  # Sigmoid to output values between 0 and 1
        )

    def forward(self, x):
        # Apply the convolutional layers
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        x = x.view(x.size(0), self.griddim, self.griddim)
        return x

    def _get_conv_output_size(self):
            # Create a dummy input tensor to calculate the output size after conv layers
            dummy_input = torch.ones(1, 3, self.image_size, self.image_size)  # Batch size 1, 3 channels, 224x224 image
            x = self.conv_layers(dummy_input)
            return x.numel()  # Return the number of elements in the tensor after conv layers
