import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from cnn_yolo_face import YOLOFaceCNN
from loss import YOLOFaceLoss
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
    "batch_size": 16,
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
    return images, list(targets)  # Return targets as a list (variable-length)

train_data = YOLODataset(img_dir="../../data/training/large_dataset/images", label_dir="../../data/training/large_dataset/labels", transform=transform)

train_loader = DataLoader(train_data, batch_size=hyperparameters['batch_size'], shuffle=True, collate_fn=collate_fn)
model = YOLOFaceCNN(grid_size=hyperparameters['grid_size']).to(DEVICE)

# criterion = YOLOFaceLoss()
criterion = YOLOFaceLoss(lambda_coord=hyperparameters['lambda_coord'], lambda_noobj=hyperparameters['lambda_noobj'])
optimizer = optim.Adam(model.parameters(), lr=hyperparameters['learning_rate'])

log_file = f"{hyperparameters['model_name_root']}.json"
training_log = {"training_parameters": hyperparameters, "epochs": []}
best_model = {"epoch": None, "avg_total_loss": float("inf"), "model_path": None}

with open(log_file, "w") as f:
    json.dump(training_log, f, indent=4)

# Training Loop

saved_models = []  

for epoch in range(start_epoch, hyperparameters['epochs']):
    print(epoch)
    start_time = time.time()  # Start epoch timer
    model.train()
    total_loss = 0.0
    total_obj_loss = 0.0
    total_coord_loss = 0.0
    total_no_obj_loss = 0.0
    total_batches = len(train_loader)
    i = 1

    for images, labels in train_loader:
        if images.size(0) == 1:  # Skip batch if batch size is 1
            continue

        images = images.to(DEVICE)
        labels = [label.to(DEVICE).float() for label in labels]

        optimizer.zero_grad()
        outputs = model(images)


        # Convert labels to a grid format
        labels_grid = torch.zeros_like(outputs).to(DEVICE)

        for j, label in enumerate(labels):
            # print(f"        {i}")
            if label.numel() > 0:  # Skip empty labels (no faces)
                # Extract the coordinates (x, y, w, h) from the label (ignoring class probability)
                x, y, w, h = label[0, 1], label[0, 2], label[0, 3], label[0, 4]

                # Compute the grid coordinates for the center of the bounding box
                center_x, center_y = int(x * hyperparameters['grid_size']), int(y * hyperparameters['grid_size'])

                # Compute the grid cell indices covered by the bounding box
                # Calculate the bounding box's top-left and bottom-right corners in terms of grid cells
                half_w = int(w * hyperparameters['grid_size'] / 2)
                half_h = int(h * hyperparameters['grid_size'] / 2)

                start_x = max(int((x - w / 2) * hyperparameters['grid_size']), 0)
                end_x = min(int((x + w / 2) * hyperparameters['grid_size']), hyperparameters['grid_size'] - 1)
                start_y = max(int((y - h / 2) * hyperparameters['grid_size']), 0)
                end_y = min(int((y + h / 2) * hyperparameters['grid_size']), hyperparameters['grid_size'] - 1)

                # Iterate over the grid cells that the bounding box covers and set them to 1
                for gx in range(start_x, end_x + 1):
                    for gy in range(start_y, end_y + 1):
                        labels_grid[j, gx, gy, 0] = 1  # The first element (index 0) represents the class probability
                        labels_grid[j, gx, gy, 1:5] = torch.tensor([x, y, w, h]).to(DEVICE)

        loss, coord_loss, obj_loss, no_obj_loss = criterion(outputs, labels_grid)

        loss.backward()
        optimizer.step()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        total_loss += loss.item()
        total_coord_loss += coord_loss.item()
        total_obj_loss += obj_loss.item()
        total_no_obj_loss += no_obj_loss.item()
        
        print(f"Batch number: {i} Batch loss: {loss} Total loss: {total_loss} total batches: {len(train_loader)}")
#        pbar.update(1)  # Display the average loss so far
        # pbar.update(1)
        i = i + 1

    print(f"Epoch {epoch+1}/{hyperparameters['epochs']}, Loss: {total_loss/len(train_loader):.4f}")

    end_time = time.time()  # End epoch timer
    duration = end_time - start_time
    avg_loss = total_loss / total_batches
    avg_coord_loss = total_coord_loss / total_batches
    avg_obj_loss = total_obj_loss / total_batches
    avg_no_obj_loss = total_no_obj_loss / total_batches

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
        "avg_coord_loss": avg_coord_loss,
        "avg_obj_loss": avg_obj_loss,
        "avg_no_obj_loss": avg_no_obj_loss
    }

    # Update JSON file
    with open(log_file, "r+") as f:
        training_log = json.load(f)
        training_log["epochs"].append(epoch_data)
        training_log["best_model"] = best_model  # Store best model details
        f.seek(0)
        json.dump(training_log, f, indent=4)

