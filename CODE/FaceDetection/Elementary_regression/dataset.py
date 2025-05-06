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
        # print(img_path)
        
        image = Image.open(img_path).convert("RGB")
        
        # Store the original size of the image before any transformations
        original_width, original_height = image.size

        # Apply transformations if available (such as resizing, etc.)
        if self.transform:
            image = self.transform(image)

        # After the transformation, image is a tensor with shape [C, H, W], so:
        new_width, new_height = image.size(2), image.size(1)  # Access width and height from tensor

        # Load labels (if exists)
        boxes = []
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                for line in f:
                    values = list(map(float, line.strip().split()))
                    # Rescale the bounding box coordinates
                    norm_x_center, norm_y_center, norm_bbox_width, norm_bbox_height = values[1], values[2], values[3], values[4]

                    #recalculates the original pixel coordinates from norm
                    ox_center = norm_x_center * original_width
                    oy_center = norm_y_center * original_height
                    obbox_width = norm_bbox_width * original_width
                    obbox_height = norm_bbox_height * original_height


                    # Rescale bounding box coordinates to the new image size (after transformation)
                    x_center = ox_center * new_width / original_width
                    y_center = oy_center * new_height / original_height
                    width = obbox_width * new_width / original_width
                    height = obbox_height * new_height / original_height

                    #renormalize the coordinates

                    n_x_center = x_center/new_width
                    n_y_center = y_center/new_height
                    n_width = width/new_width
                    n_height = width/new_height

                    # Append the adjusted box
                    boxes.append([values[0], n_x_center, n_y_center, n_width, n_height])

        return image, torch.tensor(boxes)