import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from gridmodel import GridCNNModel, GridSimpleCNNModel, GridYOLOCNNModel
from dataset import GridDataset
from loss import GridLoss
import os
import json
import time
import os
import argparse
from tabulate import tabulate

description = 'Training program for the grid cnn model'

parser = argparse.ArgumentParser(prog='train.py',
                                 description=description,
                                 usage='train.py -id <image directory> -ld <labels directory> -gs <grid_size> -ep <epochs>',
                                 epilog="",
                                 formatter_class=argparse.RawTextHelpFormatter )

parser.add_argument("-id", "--image_directory", required=True, help="The directory with training images", type=str)
parser.add_argument("-ld", "--labels_directory", required=True, help="The directory with training labels", type=str)
parser.add_argument("-o", "--output_name", help="Root name of the saved model. If empty, it is auto generated", type=str)
parser.add_argument("-ep", "--epochs", help="Number of epochs", type=int, default = 50)
parser.add_argument("-gs", "--grid_size", help="Grid size", type=int, default = 14)
parser.add_argument("-bs", "--batch_size", help="Batch size", type=int, default=16)
parser.add_argument("-ir", "--input_resolution", help="Input resolution for the modl", choices = [224, 448], default = 224)
parser.add_argument("-lmo", "--lambda_obj", help="Weight for objectness loss", default=1.0, type=float)
parser.add_argument("-lmn", "--lambda_noobj", help="Weight for lack of object loss", default=0.5, type=float)
parser.add_argument("-lr", "--learning_rate", help="Learning rate for optimizer", default=0.0001, type=float)
parser.add_argument("-pbd", "--print_batch_level_summary", help="Prints a row for each batch", default=True, type=bool)
parser.add_argument("-dd", "--subset_train", help="Use only a limited number of images for training (usually for testimg models, etc). Leave it at 0 to get all samples) ", default=0, type=int)

args = parser.parse_args()

input_dir = args.image_directory
output_dir = args.labels_directory

# Check if input directory exists
if not os.path.exists(args.image_directory):
    raise FileNotFoundError(f"Input directory '{args.image_directory}' does not exist.")

if not os.path.exists(args.labels_directory):
    raise FileNotFoundError(f"Input directory '{args.labels_directory}' does not exist.")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = f"cnn_grid_model_{args.input_resolution}_{args.grid_size}"

if args.output_name is not None:
    model_name = args.output_name


hyperparameters = {
    "learning_rate": args.learning_rate,
    "epochs": args.epochs,
    "batch_size": args.batch_size,
    "grid_size" : args.grid_size,
    "input resolution" : args.input_resolution,
    "lambda_obj": args.lambda_obj,
    "lambda_noobj": args.lambda_noobj,
    "device": str(DEVICE),
    "model_name_root": model_name
}

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

transform_2x = transforms.Compose([
    transforms.Resize((448, 448)),
    transforms.ToTensor(),
])



#Dataset and model depend on input image resolution
dataset = GridDataset(args.image_directory, args.labels_directory, grid_size=hyperparameters['grid_size'], transform=transform)
model = GridSimpleCNNModel(griddim=hyperparameters['grid_size'], image_size=224).to(DEVICE)

if args.input_resolution == 448:
    dataset = GridDataset(args.image_directory, args.labels_directory, grid_size=hyperparameters['grid_size'], transform=transform_2x)
    model = GridYOLOCNNModel(griddim=hyperparameters['grid_size'], image_size=448).to(DEVICE)


dataloader = DataLoader(dataset, batch_size=hyperparameters['batch_size'], shuffle=True)

if args.subset_train > 0:
    indices = torch.randperm(len(dataset)).tolist()[:args.subset_train]
    subset_dataset = Subset(dataset, indices)
    dataloader = DataLoader(subset_dataset, batch_size=hyperparameters['batch_size'], shuffle=True)


loss_fn = GridLoss(lambda_obj=hyperparameters['lambda_obj'], lambda_noobj=hyperparameters['lambda_noobj']).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=hyperparameters['learning_rate'])

log_file = f"{hyperparameters['model_name_root']}.json"
training_log = {"training_parameters": hyperparameters, "epochs": []}
best_model = {"epoch": None, "avg_total_loss": float("inf"), "model_path": None}

with open(log_file, "w") as f:
    json.dump(training_log, f, indent=4)

saved_models = []  

## Print a summary
print("Training parameters")
table_data = [(key, value) for key, value in hyperparameters.items()]
additional_rows = [
    ("Images", args.image_directory),  
    ("Labels", args.labels_directory), 
    ("Print Batch", args.print_batch_level_summary),
    ("Subset Train", f"YES [{args.subset_train}]" if args.subset_train > 0 else "No" ) 
]

table_data = additional_rows + table_data
print(tabulate(table_data, headers=["Hyperparameter", "Value"], tablefmt="grid"))

# Training loop
for epoch in range(hyperparameters['epochs']):
    print("Starting epoch: " + str(epoch))
    start_time = time.time()
    model.train()
    total_loss = 0.0
    total_batches = len(dataloader)
    i = 1
    for imgs, labels in dataloader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        # Forward pass
        pred = model(imgs)
        # Compute the loss
        loss = loss_fn(pred, labels)
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        print(f"    -> Batch number: {i} Batch loss: {loss} Total loss: {total_loss} total batches: {len(dataloader)}")
        i = i + 1

    print(f"Epoch [{epoch+1}/{hyperparameters['epochs']}], Loss: {total_loss/len(dataloader)}")
    end_time = time.time()  # End epoch timer
    duration = end_time - start_time
    avg_loss = total_loss / total_batches    

    model_filename = f"{hyperparameters['model_name_root']}_{epoch}_{avg_loss:.4f}.pth"
    torch.save(model.state_dict(), model_filename)
    saved_models.append(model_filename)
    # torch.save(model.state_dict(), f"yolo_big_data_ng_{epoch}_{total_loss:.4f}.pth")

    if len(saved_models) > 3:
        old_model = saved_models.pop(0)
        if old_model != best_model["model_path"]:  # Don't delete the best model
            os.remove(old_model)
        else:
            saved_models.append(old_model)  # Put it back since we don't delete it

    # Update best model if this epoch has the lowest loss
    if avg_loss < best_model["avg_total_loss"]:
        # Delete the previous best model if it's not one of the last 3
        if best_model["model_path"] and best_model["model_path"] not in saved_models:
            os.remove(best_model["model_path"])

        # Update best model info
        best_model = {
            "epoch": epoch,
            "avg_total_loss": avg_loss,
            "model_path": model_filename
        }

    epoch_data = {
        "epoch": epoch,
        "batches": len(dataloader),
        "duration": duration,
        "avg_total_loss": avg_loss,
    }

    # Update JSON file
    with open(log_file, "r+") as f:
        training_log = json.load(f)
        training_log["epochs"].append(epoch_data)
        training_log["best_model"] = best_model  # Store best model details
        f.seek(0)
        json.dump(training_log, f, indent=4)
