import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from cnn_yolo_face import YOLOFaceCNN
from loss import YOLOFaceLoss, YOLOv1Loss
from dataset import YOLODataset
from tqdm import tqdm 
import torchvision.transforms as T
from PIL import Image, ImageDraw
import json
import time
import os

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

hyperparameters = {
    "learning_rate": 0.0001,
    "epochs": 60,
    "batch_size": 2,
    "grid_size" : 7,
    "lambda_coord": 0.8,
    "lambda_noobj": 0.005,
    "device": str(DEVICE),
    "model_name_root": "yolo_like_simple_loss"
}

start_epoch = 0

# Dataset preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def collate_fn(batch):
    images, targets = zip(*batch)  # Separate images and labels
    images = torch.stack(images, 0)  # Stack images normally
    targets = torch.stack(targets, 0)
    return images, targets  # Return targets as a list (variable-length)

train_data = YOLODataset(img_dir="../../data/training/large_dataset/images", label_dir="../../data/training/large_dataset/labels", transform=transform)

train_loader = DataLoader(train_data, batch_size=hyperparameters['batch_size'], shuffle=True, collate_fn=collate_fn)
model = YOLOFaceCNN(grid_size=hyperparameters['grid_size']).to(DEVICE)

# criterion = YOLOFaceLoss()
criterion = YOLOv1Loss(lambda_coord=hyperparameters['lambda_coord'], lambda_noobj=hyperparameters['lambda_noobj'])
optimizer = optim.Adam(model.parameters(), lr=hyperparameters['learning_rate'])

log_file = f"{hyperparameters['model_name_root']}.json"
training_log = {"training_parameters": hyperparameters, "epochs": []}
best_model = {"epoch": None, "avg_total_loss": float("inf"), "model_path": None}

with open(log_file, "w") as f:
    json.dump(training_log, f, indent=4)

# Training Loop

saved_models = []  

for epoch in range(start_epoch, hyperparameters['epochs']):
    start_time = time.time()  # Start epoch timer
    model.train()
    total_loss = 0.0
    total_loc_loss = 0.0
    total_loss_obj = 0.0
    total_loss_noobj = 0.0
    total_batches = len(train_loader)
    i = 1

    for images, labels_grid in train_loader:
        images = images.to(DEVICE)
        labels_grid = labels_grid.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        # print(outputs.shape)
        # print(labels_grid.shape)

        #loc_loss Measures how well the model predicts the bounding box center (x, y) and dimensions (w, h) for grid cells containing objects.
        #conf_loss_obj (Confidence Loss for Object Cells) Definition: Measures how well the model predicts the confidence score for grid cells containing objects.
        #conf_loss_noobj (Confidence Loss for Non-Object Cells) Definition: Measures how well the model predicts the confidence score for grid cells that do not contain objects.

        loss, loc_loss, loss_obj, loss_noobj = criterion(outputs, labels_grid)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_loc_loss += loc_loss.item()
        total_loss_obj += loss_obj.item()
        total_loss_noobj += loss_noobj.item()

        print(f"Batch {i}/{total_batches}, Batch loss: {loss.item():.4f}, ")    
        # print(f"Epoch {epoch+1}/{hyperparameters['epochs']}, "
      #  f"Total Loss: {total_loss:.4f}, Loc Loss: {total_loc_loss:.4f}, "
      #  f"Conf Obj Loss: {total_conf_loss_obj:.4f}, Conf NoObj Loss: {total_conf_loss_noobj:.4f}")
        i = i + 1

    print(f"Epoch {epoch+1}/{hyperparameters['epochs']}, Loss: {total_loss/len(train_loader):.4f}")

    end_time = time.time()  # End epoch timer
    duration = end_time - start_time
    avg_loss = total_loss / total_batches
    avg_loc_loss = total_loc_loss / total_batches
    avg_loss_obj = total_loss_obj / total_batches
    avg_loss_noobj = total_loss_noobj / total_batches

# Save the trained model
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
        "batches": len(train_loader),
        "duration": duration,
        "avg_total_loss": avg_loss,
        "avg_loc_loss": avg_loc_loss,
        "avg_loss_obj": avg_loss_obj,
        "avg_loss_noobj": avg_loss_noobj
    }

    # Update JSON file
    with open(log_file, "r+") as f:
        training_log = json.load(f)
        training_log["epochs"].append(epoch_data)
        training_log["best_model"] = best_model  # Store best model details
        f.seek(0)
        json.dump(training_log, f, indent=4)

