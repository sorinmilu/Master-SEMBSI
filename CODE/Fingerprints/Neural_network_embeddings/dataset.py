import os
import random
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class FingerprintDataset(Dataset):
    def __init__(self, data_dir, input_size=(105, 105)):
        self.data_dir = data_dir
        self.image_paths = []
        self.input_size = input_size

        # Collect all image paths
        for class_dir in os.listdir(data_dir):
            class_path = os.path.join(data_dir, class_dir)
            if os.path.isdir(class_path):
                for image_name in os.listdir(class_path):
                    self.image_paths.append(os.path.join(class_path, image_name))

        # Define two different transform pipelines
        self.transform1 = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize(self.input_size),
            transforms.RandomRotation(10),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        
        self.transform2 = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize(self.input_size),
            transforms.RandomRotation(10),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])        
        ])

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img1_path = self.image_paths[idx]
        img1 = Image.open(img1_path).convert('RGB')

        is_positive = random.choice([True, False])

        if is_positive:
            img2_path = img1_path
            img2 = Image.open(img2_path).convert('RGB')

            # Apply different augmentations for each image in the same pair
            img1 = self.transform1(img1)
            img2 = self.transform2(img2)
        else:
            img2_path = random.choice(self.image_paths)
            while img2_path == img1_path:
                img2_path = random.choice(self.image_paths)
            img2 = Image.open(img2_path).convert('RGB')

            img1 = self.transform1(img1)
            img2 = self.transform2(img2)

        return img1, img2, int(is_positive)

class FingerprintDatasetDirect(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_paths = []
        
        # Collecting all images in the dataset
        for class_dir in os.listdir(data_dir):
            class_path = os.path.join(data_dir, class_dir)
            if os.path.isdir(class_path):
                for image_name in os.listdir(class_path):
                    self.image_paths.append(os.path.join(class_path, image_name))
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load the first image
        img1_path = self.image_paths[idx]
        img1 = Image.open(img1_path).convert('RGB')

        # Randomly decide if we want a positive or negative pair
        is_positive = random.choice([True, False])

        if is_positive:
            # Positive pair: same image loaded twice
            img2_path = img1_path
            img2 = Image.open(img2_path).convert('RGB')
        else:
            # Negative pair: randomly select a different image
            img2_path = random.choice(self.image_paths)
            while img2_path == img1_path:  # Ensure it's a different image
                img2_path = random.choice(self.image_paths)
            img2 = Image.open(img2_path).convert('RGB')

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2, int(is_positive)  # Return 1 for positive, 0 for negative
