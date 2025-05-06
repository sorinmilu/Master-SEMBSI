import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import FingerprintDataset
from network import SiameseNetwork
from loss import ContrastiveLoss
from torchvision import transforms
import argparse
from tqdm import tqdm



def train(input_directory, model_file, embedding_size, num_epochs, batch_size, learning_rate):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    train_dataset = FingerprintDataset(input_directory)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = SiameseNetwork(embedding_dim=embedding_size).to(device)
    loss_function = ContrastiveLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    min_loss = float('inf')  # Set to infinity initially

# Training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        for batch_idx, (img1, img2, label) in enumerate(train_loader, start=1):  # Use enumerate to get batch index
            img1, img2, label = img1.to(device), img2.to(device), label.to(device)
            optimizer.zero_grad()
            output1, output2 = model(img1, img2)
            loss = loss_function(output1, output2, label)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            print(f"Batch [{batch_idx}/{len(train_loader)}] Loss: {loss.item()}")  # Include batch number

        epoch_loss = total_loss / len(train_loader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss}')

        # Save the model checkpoint for the current epoch
        # torch.save(model.state_dict(), f'siamese_model_epoch_{epoch+1}.pth')

        # Check if this is the minimum loss so far
        if epoch_loss < min_loss:
            min_loss = epoch_loss
            torch.save(model.state_dict(), 'siamese_model_min_loss.pth')  # Save the model with minimum loss
            print(f"New minimum loss achieved: {min_loss}. Model saved as 'siamese_model_min_loss.pth'.")

    print("Training finished!")


parser = argparse.ArgumentParser(prog='train.py',
                                 description="Program de antrenament pentru reteaua neuronala siameză de identificare a amprentelor.",
                                 usage='train.py -m <model file> -id <images directory> -ep 10',
                                 epilog="Programul utilizeaza un director de imagini pentru a antrena o retea neuronala cu scopul de a identifica amprentele")


parser.add_argument('-m', "--model_file", default="face_model.pth", help="Calea intreaga (inclusiv numele fisierului) catre modelul de antrenat")
parser.add_argument('-id', "--input_directory", help="Directorul unde se afla datele de antrenament")
parser.add_argument("-ep", "--epochs", help="Numarul de epoci", type=int, default = 50)
parser.add_argument("-bs", "--batch_size", help="Dimensiunea calupului de imagini utilizate în antrenament", type=int, default = 16)
parser.add_argument('-e', "--embedding_size", default=128, help="Dimensiunea stratului de embeddings", type=int)
parser.add_argument("-lr", "--learning_rate", help="Rata de învățare", type=float, default = 0.0001)

args = parser.parse_args()

if not os.path.exists(args.input_directory):
    raise FileNotFoundError(f"Input directory '{args.input_directory}' does not exist.")


train(args.input_directory, args.model_file, args.embedding_size, args.epochs, args.batch_size, args.learning_rate)
