import torch
import torch.utils.data as data
import os
from PIL import Image
import numpy as np

GRIDSIZE = 7  # Example grid size, you can change this

class GridDataset(data.Dataset):
    def __init__(self, img_dir, label_dir, grid_size=7, transform=None):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.grid_size = grid_size
        self.transform = transform
        self.img_names = [f for f in os.listdir(img_dir) if f.endswith('.jpg')]

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = os.path.join(self.img_dir, img_name)
        label_path = os.path.join(self.label_dir, img_name.replace('.jpg', '.txt'))
        
        # Load the image
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        
        # Initialize grid
        grid = np.zeros((self.grid_size, self.grid_size))

        # Parse ground truth label file
        with open(label_path, 'r') as f:
            for line in f.readlines():
                label_data = line.strip().split()
                class_id, cx, cy, w, h = map(float, label_data)

                cx_cell = int(cx * self.grid_size)
                cy_cell = int(cy * self.grid_size)
                
                # Convert box dimensions to grid cells
                half_w = int((w * self.grid_size) / 2)
                half_h = int((h * self.grid_size) / 2)
                
                # Find the bounding box grid cells
                for i in range(max(0, cx_cell - half_w), min(self.grid_size, cx_cell + half_w + 1)):
                    for j in range(max(0, cy_cell - half_h), min(self.grid_size, cy_cell + half_h + 1)):
                        grid[i, j] = 1  # Set the cell as containing the object


                # # Set objectness to 1
                # for row in range(self.grid_size):
                #     for col in range(self.grid_size):
                #         print(grid[row, col], end = " ")
                #     print()    
                # print('--------------------')
        # Convert grid to tensor
        grid = torch.tensor(grid, dtype=torch.float32)

        return img, grid
