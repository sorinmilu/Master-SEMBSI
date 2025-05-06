import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import argparse
import os

from network import FingerprintClassifier
from dataset import FingerprintClassificationDataset
from loss import ClassificationLoss

def train(input_directory, model_file, num_epochs, batch_size, learning_rate):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    dataset = FingerprintClassificationDataset(
        root_dir=input_directory,
        transform=transform
    )

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = FingerprintClassifier().to(device)
    criterion = ClassificationLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        loop = tqdm(dataloader, desc=f"Epoch [{epoch+1}/{num_epochs}]", leave=False)
        for images, labels in loop:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        print(f"Epoch {epoch+1}, Loss: {epoch_loss / len(dataloader)}")

    torch.save(model.state_dict(), model_file)
    print("Training complete. Model saved.")


parser = argparse.ArgumentParser(prog='train.py',
                                 description="Program de antrenament pentru reteaua neuronala de clasificare a amprentelor.",
                                 usage='train.py -m <model file> -id <images directory> -ep 10',
                                 epilog="This program gets a directory with jpeg images as an argument, runs the model on each image and prints the class of fingerprints found in the images")


parser.add_argument('-m', "--model_file", default="face_model.pth", help="Calea intreaga (inclusiv numele fisierului) catre modelul de antrenat")
parser.add_argument('-id', "--input_directory", help="Directorul unde se afla datele de antrenament")
parser.add_argument("-ep", "--epochs", help="Number of epochs", type=int, default = 50)
parser.add_argument("-bs", "--batch_size", help="Batch size for training", type=int, default = 16)
parser.add_argument("-lr", "--learning_rate", help="Learning rate", type=float, default = 0.0001)

args = parser.parse_args()

if not os.path.exists(args.input_directory):
    raise FileNotFoundError(f"Input directory '{args.input_directory}' does not exist.")


train(args.input_directory, args.model_file, args.epochs, args.batch_size, args.learning_rate)
