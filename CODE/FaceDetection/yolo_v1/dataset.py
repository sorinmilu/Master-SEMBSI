import os
from PIL import Image
from torch.utils.data import Dataset
import glob
import torch

class FaceDataset(Dataset):
    def __init__(self, image_dir, positive_prefix, transform=None):
        self.image_dir = image_dir
        self.positive_prefix = positive_prefix
        self.transform = transform
        self.image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir)]
        self.labels = [1 if self.positive_prefix in image_path else 0 for image_path in self.image_paths]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx])
        label = self.labels[idx]
        # print(self.image_paths[idx], label )
        if self.transform:
            image = self.transform(image)

        return image, label

class YOLODataset(Dataset):
    def __init__(self, img_dir, label_dir, transform=None):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.images = sorted(glob.glob(os.path.join(img_dir, "*.jpg")) + glob.glob(os.path.join(img_dir, "*.png")))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        img_name = os.path.basename(img_path).split(".")[0]
        label_path = os.path.join(self.label_dir, f"{img_name}.txt")

        # Load image with PIL to get its size
        image = Image.open(img_path).convert("RGB")
        original_width, original_height = image.size

        # Apply transformations if available
        if self.transform:
            image = self.transform(image)

        # After transformation, image is a tensor with shape [C, H, W]
        if isinstance(image, torch.Tensor):
            new_width, new_height = image.size(2), image.size(1)
        else:
            new_width, new_height = image.size

        # Initialize labels grid
        S = 7  # Grid size
        labels_grid = torch.zeros((S, S, 5))  # Shape: [S, S, 5] -> [x, y, w, h, confidence]

        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                for line in f:
                    values = list(map(float, line.strip().split()))
                    _, norm_x_center, norm_y_center, norm_width, norm_height = values

                    # Convert normalized coordinates to pixel values
                    x_center = norm_x_center * original_width
                    y_center = norm_y_center * original_height
                    width = norm_width * original_width
                    height = norm_height * original_height

                    # Rescale to new image size
                    x_center = x_center * new_width / original_width
                    y_center = y_center * new_height / original_height
                    width = width * new_width / original_width
                    height = height * new_height / original_height

                    # Normalize to grid cell
                    grid_x = int(x_center * S / new_width)
                    grid_y = int(y_center * S / new_height)
                    x_offset = (x_center * S / new_width) - grid_x
                    y_offset = (y_center * S / new_height) - grid_y

                    # Assign the bounding box and confidence to the grid cell
                    labels_grid[grid_y, grid_x, 0] = x_offset
                    labels_grid[grid_y, grid_x, 1] = y_offset
                    labels_grid[grid_y, grid_x, 2] = width / new_width  # Normalize width
                    labels_grid[grid_y, grid_x, 3] = height / new_height  # Normalize height
                    labels_grid[grid_y, grid_x, 4] = 1.0  # Confidence score

        return image, labels_grid