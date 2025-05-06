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


# Preprocess image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def predict(image_path, output_path, model, object_threshold, nms_threshold):

    original_image = Image.open(image_path).convert("RGB")
    dir_name = os.path.dirname(image_path)


    dir_name, base_name = os.path.split(output_path)
    name, ext = os.path.splitext(base_name)
    box_image_path = os.path.join(dir_name, f"{name}_box{ext}")
    nms_image_path = os.path.join(dir_name, f"{name}_nms{ext}")

    original_width, original_height = original_image.size

    # send the image to the model
    image = transform(original_image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        output = model(image)  # Shape: (1, 7, 7, 5)


    # confidence_map = output[0, :, :, 0]  # Extract confidence scores (shape: 7x7)

    output = output.squeeze(0)  # Remove batch dim
    

    rows, cols, features = output.shape  

    # Loop through each row and column
    # for i in range(rows):  
    #     for j in range(cols):  
    #         print(output[i, j])


    # Filter out boxes with probability less than 0.5
    filtered_boxes = output[output[:, :, 0] > object_threshold]

    x_center = (filtered_boxes[:, 1] * original_width).cpu().numpy()
    y_center = (filtered_boxes[:, 2] * original_height).cpu().numpy()
    box_width = (filtered_boxes[:, 3] * original_width).cpu().numpy()
    box_height = (filtered_boxes[:, 4] * original_height).cpu().numpy()

    # Create bounding boxes in the form (x1, y1, x2, y2) for NMS
    x1 = x_center - box_width / 2
    y1 = y_center - box_height / 2
    x2 = x_center + box_width / 2
    y2 = y_center + box_height / 2

    boxes = torch.tensor(np.array([x1, y1, x2, y2])).T
    scores = filtered_boxes[:, 0].clone()

    image_copy = original_image.copy()
    draw = ImageDraw.Draw(image_copy)

    for i, box in enumerate(boxes):

        x1, y1, x2, y2 = box.tolist()
        outline_color = (255, 0, 0)  # Red for example, change it as needed
        draw.rectangle([x1, y1, x2, y2], outline=outline_color, width=7)

    image_copy.save(box_image_path)

    # Apply NMS using the first column as probability score and coordinates
    indices = nms(boxes, scores, iou_threshold=nms_threshold)

    draw = ImageDraw.Draw(original_image)

    for idx in indices:
        box = boxes[idx].cpu().numpy()
        x1, y1, x2, y2 = map(int, box)
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)

    original_image.save(nms_image_path)

description = 'Inference program for yolo-like regression model that process an entire directory'

parser = argparse.ArgumentParser(prog='infer_folder.py',
                                 description=description,
                                 usage='infer_folder.py -id <input_directory> -od <output_directory> -m <model file>' ,
                                 epilog="This program gets a directory with jpeg images as an argument, runs the model on each image replicates the input folder structure on the output folder. The output folder has to be empty")

parser.add_argument("-id", "--input_directory", required=True, help="Directory with the testing images. Can have subdiorectories", type=str)
parser.add_argument("-od", "--output_directory", required=True, help="Directory with the inference results. It has to be empty. If the given directory does not exists, it will be created. The structure of the input directory (subdirectories, etc) will be recreated", type=str)
parser.add_argument("-m", "--model", help="File that holds the model", type=str)
parser.add_argument("-g", "--grid_size", help="Grid size of the model (default 7) - it depends on the model definition", type=int, default=7)
parser.add_argument("-ot", "--object_threshold", type=float, default = 0.7,  help="Threshold above wich a cell becomes a hit. Default 0.5")
parser.add_argument("-nt", "--nms_iou_threshold", type=float, default = 0.5,  help="Non maximum suppression IOU threshold. Value between 0 and 1, usually 0.5, Less will collapse boxes more")


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

        predict(file_path, output_file_path, model, args.object_threshold, args.nms_iou_threshold)

        pbar.update(1)

print("Processing complete.")
