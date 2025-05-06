import argparse
import torch
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from cnn_class import CNNClassifier
from dataset import FaceDataset
from PIL import Image

# Define transformations (resize, normalization)
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


parser = argparse.ArgumentParser(
        prog="train", 
        description="Training program for a simple yes/no CNN classifier",
        usage="python3 train.py -d <data_dir> -m <model_name> -e <embedding_size>",
        epilog="This program trains a simple CNN model using files found in <data_dir> directory. Those images have to be positive and negative, the positive images having in their names the prefix given in <positive_prefix> argument"
        )
parser.add_argument('-e', "--epochs", default=10, help="Number of epochs", type=int)
parser.add_argument('-d', "--data_dir", default=10, help="The directory containing images", required=True)
parser.add_argument('-p', "--positive_prefix", default="face", help="the part found in the names of the positive files")
parser.add_argument('-es', "--embedding_size", default=512, help="size of the last fully connected layer", type=int)
parser.add_argument('-m', "--model", default="models/face_model", help="Path to output model")

args = parser.parse_args()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create dataset and dataloader
train_dataset = FaceDataset(image_dir=args.data_dir, transform=transform, positive_prefix=args.positive_prefix)
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)

# Initialize the model, loss function, and optimizer
model = CNNClassifier(embedding_size=args.embedding_size).to(DEVICE)
criterion = torch.nn.BCELoss().to(DEVICE)  # Binary Cross Entropy loss
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Training loop
num_epochs = args.epochs
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs,labels = inputs.to(DEVICE), labels.to(DEVICE)
        labels = labels.float().view(-1, 1)  # Ensure labels are float for BCE loss
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader)}")

# Save the trained model
    torch.save(model.state_dict(), f"{args.model}_{running_loss/len(train_loader)}.pth")
