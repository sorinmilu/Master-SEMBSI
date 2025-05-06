import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class YOLOFaceCNNOld(nn.Module):
    def __init__(self, grid_size=7):
        super(YOLOFaceCNN, self).__init__()
        self.grid_size = grid_size  

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

        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 14 * 14, 4096),  
            nn.BatchNorm1d(4096),
            nn.LeakyReLU(0.1),
            nn.Linear(4096, self.grid_size ** 2 * 5),  # Predicts (p, x, y, w, h) per grid cell
            nn.Sigmoid()  
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x.view(-1, self.grid_size, self.grid_size, 5)  # Now predicts bounding boxes

        # x = self.fc_layers[:-1](x)  # Forward pass without the last Sigmoid
        # raw_logits = x.clone()  # Store raw logits before sigmoid
        # x = torch.sigmoid(x)  # Apply sigmoid manually
        # return raw_logits, x  # Return both


class YOLOFaceCNN(nn.Module):
    def __init__(self, grid_size=7):
        super(YOLOFaceCNN, self).__init__()
        self.grid_size = grid_size

        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3), nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 192, kernel_size=3, padding=1), nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(192, 128, kernel_size=1), nn.LeakyReLU(0.1),
            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.LeakyReLU(0.1),
            nn.Conv2d(256, 256, kernel_size=1), nn.LeakyReLU(0.1),
            nn.Conv2d(256, 512, kernel_size=3, padding=1), nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(512, 1024, kernel_size=3, padding=1), nn.LeakyReLU(0.1),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1), nn.LeakyReLU(0.1),
            nn.Conv2d(1024, 1024, kernel_size=3, stride=2, padding=1), nn.LeakyReLU(0.1),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1), nn.LeakyReLU(0.1),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1), nn.LeakyReLU(0.1)
        )
        
        self.yolo_head = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, padding=1), nn.LeakyReLU(0.1),
            nn.Conv2d(512, 5, kernel_size=1),  # Output: (x, y, w, h, confidence)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.yolo_head(x)
        x = x.permute(0, 2, 3, 1)  # Reshape to (batch, grid_size, grid_size, 5)
        x[..., 0:2] = torch.sigmoid(x[..., 0:2])  # (x, y)
        x[..., 2:4] = torch.exp(x[..., 2:4])      # (w, h)
        x[..., 4] = torch.sigmoid(x[..., 4])      # Confidence
        return x

# Test
if __name__ == "__main__":
    model = YOLOFaceCNN(grid_size=7)
    sample_input = torch.randn(1, 3, 224, 224)  # Sample image
    output = model(sample_input)
    print(output.shape)  # Expected: (1, 7, 7) for a 7x7 grid output
