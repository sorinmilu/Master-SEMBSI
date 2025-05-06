import argparse
import torch
from torchvision import transforms
from PIL import Image, ImageDraw
from gridmodel import GridSimpleCNNModel
import matplotlib.pyplot as plt
import os
import numpy as np
import sys
from torchvision.ops import nms
import decimal
import cv2
from collections import deque


# Preprocess image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
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


def find_bounding_boxes_normalized(matrix):
    rows, cols = len(matrix), len(matrix[0])
    visited = [[False] * cols for _ in range(rows)]
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, Down, Left, Right
    boxes = []

    def bfs(start_x, start_y):
        queue = deque([(start_x, start_y)])
        visited[start_x][start_y] = True
        x_min, x_max, y_min, y_max = start_x, start_x, start_y, start_y

        while queue:
            x, y = queue.popleft()

            # Expand bounding box
            x_min, x_max = min(x_min, x), max(x_max, x)
            y_min, y_max = min(y_min, y), max(y_max, y)

            # Explore neighbors
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if 0 <= nx < rows and 0 <= ny < cols and matrix[nx][ny] == 1 and not visited[nx][ny]:
                    visited[nx][ny] = True
                    queue.append((nx, ny))

        # Convert to normalized coordinates
        x_norm = x_min / rows
        y_norm = y_min / cols
        width_norm = (x_max - x_min + 1) / rows
        height_norm = (y_max - y_min + 1) / cols

        return (x_norm, y_norm, width_norm, height_norm)

    # Scan matrix
    for i in range(rows):
        for j in range(cols):
            if matrix[i][j] == 1 and not visited[i][j]:
                box = bfs(i, j)
                boxes.append(box)

    return boxes


def predict(image_path, model):

    original_image = Image.open(image_path).convert("RGB")
    dir_name = os.path.dirname(image_path)
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    heatmap_overlay_path = os.path.join(dir_name, f"{base_name}_heatmap.png")


    original_image = Image.open(image_path).convert("RGB")
    dir_name = os.path.dirname(image_path)

    base_name = os.path.splitext(os.path.basename(image_path))[0]
    heatmap_overlay_path = os.path.join(dir_name, f"{base_name}_heatmap.png")
    grid_big_overlay_path = os.path.join(dir_name, f"{base_name}_grid_big.png")
    boxes_overlay_path = os.path.join(dir_name, f"{base_name}_boxes.png")

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

        # Drawing extracting and drawing boxes
        threshold = 0.3
        matrix = (grid_map > threshold).astype(int)
        matrix = np.transpose(matrix)
        bounding_boxes = find_bounding_boxes_normalized(matrix)

        draw = ImageDraw.Draw(original_image)

        for box in bounding_boxes:
        # Convert normalized to absolute pixel coordinates
            x_min = int(box[0] * original_width)
            y_min = int(box[1] * original_height)
            x_max = int((box[0] + box[2]) * original_width)
            y_max = int((box[1] + box[3]) * original_height)
            draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=3)


        original_image.save(boxes_overlay_path)

        ax.set_xlim(0, original_width)
        ax.set_ylim(original_height, 0)  # Invert Y-axis to match image coordinate system

        ax.set_xticks([])
        ax.set_yticks([])

        plt.savefig(grid_big_overlay_path)
        plt.close()



description = 'Inference program for grid cnn model'

parser = argparse.ArgumentParser(prog='infer.py',
                                 description=description,
                                 usage='extract_directory_mtcnn.py -in <input_directory> -ot <output_directory>',
                                 epilog='This program gets an image file as argument and runs the prediction model on it. It will detect the "objectness" of a set of cells that are obtained by splitting the image in a n x n grid. The grid size has to be the same as the grid size with which the model was trained')
parser.add_argument("-i", "--image", required=True, help="Input image path", type=str)
parser.add_argument("-m", "--model", help="File that holds the model", type=str)
parser.add_argument("-g", "--grid_size", help="Grid size of the model (default 7) - it depends on the model definition", type=int, default=7)


args = parser.parse_args()

if args.image:
    CHECK_FILE = os.path.isfile(args.image)
    if not CHECK_FILE:
        print('Input file not found')
        sys.exit(1)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Load trained model
model = GridSimpleCNNModel(griddim=args.grid_size, image_size=224).to(DEVICE)
model.load_state_dict(torch.load(args.model, map_location=torch.device('cpu'), weights_only=True))
model.eval()

predict(args.image, model)
