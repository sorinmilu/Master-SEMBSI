import argparse
import torch
from torchvision import transforms
from PIL import Image, ImageDraw
from cnn_yolo_face import YOLOFaceCNN
import matplotlib.pyplot as plt
import os
import numpy as np
import sys
from torchvision.ops import nms
import decimal
import shutil
from tqdm import tqdm

# Preprocess the input image
def preprocess_image(image_path, image_size=(224, 224)):
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
    ])
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0)  # Add batch dimension
    return image, tensor

def postprocess_output(output, grid_size=7, threshold=0.5, image_size=(224, 224)):
    """
    Convert model output to bounding boxes.
    Args:
        output: Model output of shape [1, grid_size, grid_size, 5].
        grid_size: Size of the grid (e.g., 7 for 7x7 grid).
        threshold: Confidence threshold for filtering boxes.
        image_size: Original image size (width, height).
    Returns:
        List of bounding boxes [(x1, y1, x2, y2)].
    """
    output = output.squeeze(0)  # Remove batch dimension, shape: [grid_size, grid_size, 5]
    boxes = []
    cell_width = image_size[0] / grid_size  # Width of each grid cell
    cell_height = image_size[1] / grid_size  # Height of each grid cell

    for i in range(grid_size):
        for j in range(grid_size):
            confidence = output[i, j, 4]  # Confidence score
            if confidence > threshold:
                # Extract bounding box parameters
                x_offset, y_offset, w, h = output[i, j, :4]
                x_center = (j + x_offset) * cell_width
                y_center = (i + y_offset) * cell_height
                box_width = w * image_size[0]
                box_height = h * image_size[1]

                # Convert to (x1, y1, x2, y2) format
                x1 = max(0, x_center - box_width / 2)
                y1 = max(0, y_center - box_height / 2)
                x2 = min(image_size[0], x_center + box_width / 2)
                y2 = min(image_size[1], y_center + box_height / 2)
                boxes.append((x1, y1, x2, y2))

    return boxes

# Draw bounding boxes on the image
def draw_boxes(image, boxes):
    draw = ImageDraw.Draw(image)
    edge_width = 2 + max(1, int(image.width / 100))
    for box in boxes:
        draw.rectangle(box, outline="red", width=edge_width)
    return image

# Main inference function
def run_inference(model, device, image_path, model_path, output_path, grid_size=7, threshold=0.3):


    # Preprocess the image
    original_image, input_tensor = preprocess_image(image_path)
    input_tensor = input_tensor.to(device)

    # Run the model
    with torch.no_grad():
        output = model(input_tensor)  # Shape: [1, grid_size, grid_size, 5]

    # Postprocess the output
    boxes = postprocess_output(output, grid_size=grid_size, threshold=threshold, image_size=original_image.size)

    # Draw the bounding boxes
    image_with_boxes = draw_boxes(original_image, boxes)

    # Save the output image
    image_with_boxes.save(output_path)


description = 'Inference program for yolo-like regression model that process an entire directory'

parser = argparse.ArgumentParser(prog='infer_folder.py',
                                 description=description,
                                 usage='infer_folder.py -id <input_directory> -od <output_directory> -m <model file>' ,
                                 epilog="This program gets a directory with jpeg images as an argument, runs the model on each image replicates the input folder structure on the output folder. The output folder has to be empty")

parser.add_argument("-id", "--input_directory", required=True, help="Directory with the testing images. Can have subdiorectories", type=str)
parser.add_argument("-od", "--output_directory", required=True, help="Directory with the inference results. It has to be empty. If the given directory does not exists, it will be created. The structure of the input directory (subdirectories, etc) will be recreated", type=str)
parser.add_argument("-m", "--model", help="File that holds the model", type=str)
parser.add_argument("-g", "--grid_size", help="Grid size of the model (default 7) - it depends on the model definition", type=int, default=7)
parser.add_argument("-ot", "--object_threshold", type=float, default = 0.4,  help="Threshold above wich a cell becomes a hit. Default 0.5")

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
model = YOLOFaceCNN(grid_size=args.grid_size).to(DEVICE)
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

        run_inference(model, DEVICE, file_path, args.model, output_file_path,threshold=args.object_threshold, grid_size=args.grid_size)
        
        pbar.update(1)

print("Processing complete.")
