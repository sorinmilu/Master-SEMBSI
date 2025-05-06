import argparse
import torch
from torchvision import transforms
from PIL import Image, ImageDraw
from gridmodel import GridCNNModel, GridSimpleCNNModel, GridYOLOCNNModel
import matplotlib.pyplot as plt
import os
import numpy as np
import sys
from torchvision.ops import nms
import decimal
import shutil
from tqdm import tqdm
import cv2


# Preprocess image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

transform_2x = transforms.Compose([
    transforms.Resize((448, 448)),
    transforms.ToTensor(),
])

def resize_with_aspect_ratio(width, height, max_size=1000):
    if width <= max_size and height <= max_size:
        return width, height  # No resizing needed

    if width > height:
        new_width = max_size
        new_height = int(height * (max_size / width))
    else:
        new_height = max_size
        new_width = int(width * (max_size / height))

    return new_width, new_height


def predict(image_path, output_path, model):

    original_image = Image.open(image_path).convert("RGB")
    dir_name = os.path.dirname(image_path)

    dir_name, base_name = os.path.split(output_path)
    name, ext = os.path.splitext(base_name)
    heatmap_overlay_path = os.path.join(dir_name, f"{base_name}_heatmap.png")
    grid_overlay_path = os.path.join(dir_name, f"{base_name}_grid.png")
    grid_big_overlay_path = os.path.join(dir_name, f"{base_name}_grid_big.png")


    original_width, original_height = original_image.size
    new_width, new_height = resize_with_aspect_ratio(original_width, original_height)

    # send the image to the model
    image = transform(original_image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(image)  # Shape: (1, 7, 7, 5)
        output = output.squeeze(0)  # Remove batch dim
        rows, cols = output.shape  

        resized_confidence_map = output.cpu().detach().numpy()  # Ensure it's on CPU and detached from graph
        resized_confidence_map = resized_confidence_map.astype(np.float32)  # Convert to OpenCV-compatible type
        resized_confidence_map = np.transpose(resized_confidence_map)
        grid_map = resized_confidence_map


        # Resize with OpenCV
        resized_confidence_map = cv2.resize(resized_confidence_map, (original_width, original_height), interpolation=cv2.INTER_LINEAR)


        #fig, ax = plt.subplots()
        fig, ax = plt.subplots(figsize=(new_width / 100, new_height / 100))

        ax.imshow(original_image)
        ax.imshow(resized_confidence_map, cmap='hot', interpolation='nearest', alpha=0.5, vmin=0, vmax=1)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.title('Overlayed Heatmap on Original Image')
        plt.savefig(heatmap_overlay_path)  # Save the overlay image
        plt.close()  # Close the plot to free up memory

        fig, ax = plt.subplots(figsize=(new_width / 100, new_height / 100))

        ax.imshow(original_image)
        masked_grid = np.ma.masked_where(grid_map == 0, grid_map)
        ax.imshow(masked_grid, cmap='viridis', alpha=0.5, interpolation='nearest', vmin=0, vmax=1, 
                extent=[0, original_width, original_height, 0])  # Map the grid to image size
        for i in range(rows + 1):  # Horizontal lines
            ax.axhline(i * (original_height / rows), color='black', linewidth=1, alpha=0.5)

        for j in range(cols + 1):  # Vertical lines
            ax.axvline(j * (original_width / cols), color='black', linewidth=1, alpha=0.5)
        cell_width = original_width / cols
        cell_height = original_height / rows

        for i in range(rows):
            for j in range(cols):
                if grid_map[i, j] > 0.1:  # Only display values greater than 0.1
                    ax.text((j + 0.5) * cell_width, (i + 0.5) * cell_height, f'{grid_map[i, j]:.1f}', ha='center', va='center', color='white', fontsize=10)

        ax.set_xlim(0, original_width)
        ax.set_ylim(original_height, 0)  # Invert Y-axis to match image coordinate system

        ax.set_xticks([])
        ax.set_yticks([])

        plt.savefig(grid_big_overlay_path)





description = 'Inference program for yolo-like regression model that process an entire directory'

parser = argparse.ArgumentParser(prog='infer_folder.py',
                                 description=description,
                                 usage='infer_folder.py -id <input_directory> -od <output_directory> -m <model file>' ,
                                 epilog="This program gets a directory with jpeg images as an argument, runs the model on each image replicates the input folder structure on the output folder. The output folder has to be empty")

parser.add_argument("-id", "--input_directory", required=True, help="Directory with the testing images. Can have subdiorectories", type=str)
parser.add_argument("-od", "--output_directory", required=True, help="Directory with the inference results. It has to be empty. If the given directory does not exists, it will be created. The structure of the input directory (subdirectories, etc) will be recreated", type=str)
parser.add_argument("-m", "--model", help="File that holds the model", type=str)
parser.add_argument("-ir", "--input_resolution", help="Input resolution for the modl", choices = [224, 448], default = 224, type=int)
parser.add_argument("-g", "--grid_size", help="Grid size of the model (default 7) - it depends on the model definition", type=int, default=28)

args = parser.parse_args()

input_dir = args.input_directory
output_dir = args.output_directory

# Check if input directory exists
if not os.path.exists(input_dir):
    raise FileNotFoundError(f"Input directory '{input_dir}' does not exist.")

# Check if output directory exists
if os.path.exists(output_dir):
    if os.listdir(output_dir):
        raise FileExistsError(f"Output directory '{output_dir}' is not empty.")
else:
    os.makedirs(output_dir)
    print(f"Created output directory: {output_dir}")

# Count total files
file_list = []
valid_extensions = {".jpg", ".jpeg", ".png"}
for root, _, files in os.walk(input_dir):
    for file in files:
        if os.path.splitext(file)[1].lower() in valid_extensions:
            file_list.append(os.path.join(root, file))

total_files = len(file_list)
print(f"Total files to process: {total_files}")



DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load trained model

model = GridYOLOCNNModel(griddim=args.grid_size, image_size=args.input_resolution).to(DEVICE)
#model = GridSimpleCNNModel(griddim=args.grid_size, image_size=args.input_resolution).to(DEVICE)

#model = GridCNNModel(griddim=args.grid_size, image_size=args.input_resolution).to(DEVICE)

model.load_state_dict(torch.load(args.model, map_location=torch.device('cpu'), weights_only=True))
model.eval()


# Process files with progress bar
with tqdm(total=total_files, desc="Processing files", unit="file") as pbar:
    for file_path in file_list:
        rel_path = os.path.relpath(file_path, input_dir)
        target_path = os.path.join(output_dir, os.path.dirname(rel_path))
        os.makedirs(target_path, exist_ok=True)
        output_file_path = os.path.join(target_path, os.path.basename(file_path))
        
        # Process file (for now, just copying as a placeholder)
        # shutil.copy(file_path, output_file_path)

        predict(file_path, output_file_path, model)

        pbar.update(1)

print("Processing complete.")
