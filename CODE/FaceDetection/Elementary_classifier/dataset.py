import os
from PIL import Image
from torch.utils.data import Dataset

class FaceDataset(Dataset):
    def __init__(self, image_dir, positive_prefix, transform=None):
        self.image_dir = image_dir
        self.positive_prefix = positive_prefix
        self.transform = transform
        self.image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg'))]
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
