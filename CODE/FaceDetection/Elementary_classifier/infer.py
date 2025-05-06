import torch
import torch.nn as nn
import argparse
import os
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from cnn_class import CNNClassifier
import numpy as np
import math
import cv2

# Define transformation (resize, normalization)
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def save_feature_maps_as_grid(feature_maps, output_file="feature_maps_last_layer.png"):
    if len(feature_maps) == 0:
        print("No feature maps captured!")
        return

    # Extract the feature map from the first (and only) entry in the list
    feature_map = feature_maps[0]  # We take the first feature map in the list

    # Check the shape of the feature map
    print(f"Feature map shape: {feature_map.shape}")

    # Ensure the feature map has 3 dimensions: [num_channels, height, width]
    if feature_map.dim() == 4:  # If it has a batch dimension, remove it
        feature_map = feature_map.squeeze(0)

    num_maps = feature_map.size(0)  # Number of feature maps (channels)
    map_height = feature_map.size(1)  # Height of each feature map
    map_width = feature_map.size(2)  # Width of each feature map

    # Dynamically calculate grid size
    grid_cols = math.ceil(math.sqrt(num_maps))  # Number of columns
    grid_rows = math.ceil(num_maps / grid_cols)  # Number of rows

    # Calculate grid dimensions
    grid_height = grid_rows * map_height + (grid_rows - 1) * 2  # Height of the grid with spacing
    grid_width = grid_cols * map_width + (grid_cols - 1) * 2  # Width of the grid with spacing

    # Create a blank canvas for the grid
    grid_img = np.ones((grid_height, grid_width), dtype=np.float32) * 255  # White background

    # Populate the grid with feature maps
    for i in range(grid_rows):
        for j in range(grid_cols):
            map_idx = i * grid_cols + j
            if map_idx < num_maps:
                # Extract feature map and normalize it for display
                feature_map_image = feature_map[map_idx].cpu().detach().numpy()
                feature_map_image = np.maximum(feature_map_image, 0)  # Remove negative values
                feature_map_image = feature_map_image / feature_map_image.max()  # Normalize to [0, 1]

                # Position in the grid
                y_offset = i * (map_height + 2)
                x_offset = j * (map_width + 2)

                # Insert feature map into the grid
                grid_img[y_offset:y_offset+map_height, x_offset:x_offset+map_width] = feature_map_image * 255  # Scale back to [0, 255]

    # Save the grid image
    plt.imshow(grid_img, cmap='gray')
    plt.axis('off')  # Hide axes
    plt.savefig(output_file, bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f"Feature maps saved to {output_file}")

# Call this function to save the feature maps as a grid of images

def draw_feature_map(model, image_path, transform, output_file, fc_layer_size, prediction):
    """
    Draws feature maps from all convolutional layers of the model and combines them into a single image,
    with the specified fully connected layer visualized as a column of red shades on the right.
    The original image is added as the first item in the combined image.

    Args:
        model: The trained model.
        image_path: Path to the input image.
        transform: The transformation to apply to the input image.
        output_file: The output file to save the combined image.
        fc_layer_size: The size (number of output units) of the fully connected layer to visualize.
    """
    # Dictionary to store feature maps for each layer
    feature_maps_dict = {}
    fc_output = None  # To store the specified fully connected layer's output

    # Register hooks for all Conv2d layers and the specified FC layer
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            feature_maps_dict[name] = []  # Initialize a list for this layer
            module.register_forward_hook(
                lambda module, input, output, layer_name=name: feature_maps_dict[layer_name].append(output)
            )
        elif isinstance(module, nn.Linear) and module.out_features == fc_layer_size:  # Identify FC layer by size
            def fc_hook(module, input, output):
                nonlocal fc_output
                print(f"Hook triggered for FC layer with size {fc_layer_size}")  # Debug: Confirm hook is triggered
                fc_output = output
            module.register_forward_hook(fc_hook)

    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)  # Add batch dimension

    # Switch model to evaluation mode
    model.eval()

    with torch.no_grad():
        # Forward pass through the model
        print("Performing forward pass...")
        model(image)

    # Check if the specified FC layer's output was captured
    if fc_output is None:
        print(f"Error: Fully connected layer with size {fc_layer_size} was not captured.")
        return

    # Extract the specified FC layer's output
    print(f"FC Output Shape: {fc_output.shape}")  # Debug: Print FC output shape
    fc_output = fc_output.cpu().detach().numpy().flatten()  # Flatten to 1D array

    # Normalize the FC layer's output to [0, 255]
    fc_output_normalized = (fc_output - fc_output.min()) / (fc_output.max() - fc_output.min()) * 255
    fc_output_normalized = fc_output_normalized.astype(np.uint8)

    # Create a vertical column of red shades for the FC layer
    fc_column_height = len(fc_output_normalized)
    fc_column = np.zeros((fc_column_height, 50, 3), dtype=np.uint8)  

    for i, value in enumerate(fc_output_normalized):
        fc_column[i, :, 0] = 0  # Set the red channel
        fc_column[i, :, 1] = value      # Set the green channel to 0
        fc_column[i, :, 2] = 0     # Set the blue channel to 0

    

    # List to store grid images for each layer
    grid_images = []

    # Save feature maps for each layer and store the grid images
    for layer_name, feature_maps in feature_maps_dict.items():
        print(f"Processing feature maps for layer: {layer_name}")
        grid_file = f"feature_maps_{layer_name}.png"
        save_feature_maps_as_grid(feature_maps, output_file=grid_file)

        # Load the saved grid image
        grid_image = cv2.imread(grid_file, cv2.IMREAD_GRAYSCALE)
        grid_images.append(grid_image)

    # Load the original image and resize it to match the height of the combined image
    
    

    # Combine all grid images into a single image
    if grid_images:
    # Calculate the maximum height of the grid images
        max_height = max(img.shape[0] for img in grid_images)

        # Load the original image and resize it to match the height of the combined image
        original_image = cv2.imread(image_path)  # Load the original image in color (BGR format)
        original_image_resized = cv2.resize(original_image, (grid_images[0].shape[1], max_height), interpolation=cv2.INTER_AREA)

        # Add the original image to the beginning of the grid images
        grid_images.insert(0, original_image_resized)

        # Recalculate the total width of the combined image to include the original image
        total_width = sum(img.shape[1] for img in grid_images) + (len(grid_images) - 1) * 4

        # Create a blank canvas for the combined image (3 channels for color)
        combined_image = np.ones((max_height, total_width, 3), dtype=np.uint8) * 255  # White background (color)

        # Place each grid image on the canvas
        x_offset = 0
        for img in grid_images:
            if len(img.shape) == 2:  # If the image is grayscale, convert it to color
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            combined_image[: img.shape[0], x_offset : x_offset + img.shape[1]] = img
            x_offset += img.shape[1] + 4  # Add 4 pixels of spacing

        # Resize the fully connected column to match the height of the combined image
        fc_column_resized = cv2.resize(fc_column, (50, combined_image.shape[0]), interpolation=cv2.INTER_NEAREST)


        number_column_resized = np.ones((combined_image.shape[0], 100, 3), dtype=np.uint8) * 255  # White background
    
        number_text = f"{prediction:.2f}"  # Format the number to 2 decimal places
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 2
        color = (0, 0, 255)  # Red color for the text
        text_size = cv2.getTextSize(number_text, font, font_scale, thickness)[0]
        text_x = (number_column_resized.shape[1] - text_size[0]) // 2  # Center the text horizontally
        text_y = (number_column_resized.shape[0] + text_size[1]) // 2  # Center the text vertically
        cv2.putText(number_column_resized, number_text, (text_x, text_y), font, font_scale, color, thickness, lineType=cv2.LINE_AA)

        # Append the fully connected column to the right of the combined image
        final_image = np.hstack((combined_image, fc_column_resized, number_column_resized))

        # Save the final image
        cv2.imwrite(output_file, final_image)
        print(f"Combined feature maps with FC layer saved to {output_file}")

def infer(model, image_path, transform):
    # Load image
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)  # Add batch dimension

    # Switch model to evaluation mode
    model.eval()

    with torch.no_grad():
        # Forward pass
        output = model(image)
        prediction = output.item()  # Get scalar value
        if prediction >= 0.5:
            print(image_path + "Face detected!: " + str(prediction))
        else:
            print(image_path + "No face detected!: "  + str(prediction))

    return prediction

parser = argparse.ArgumentParser(description="Face detection script with drawing and printing options.")
parser.add_argument('-i', "--input_image", required=True, help="Path to input image")
parser.add_argument('-e', "--embedding_size", default=512, help="size of the last fully connected layer", type=int)
parser.add_argument('-m', "--model", default="models/face_model.pth", help="Path to input image")


args = parser.parse_args()

if not os.path.isfile(args.input_image):
    print(f"Error: Input image '{args.input_image}' does not exist.")
    exit(1)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CNNClassifier(embedding_size=args.embedding_size).to(DEVICE) 
model.load_state_dict(torch.load(args.model, weights_only=True, map_location=torch.device('cpu')))  # Load trained model
prediction = infer(model, args.input_image, transform)

print('Drawing feature map')

base_name = os.path.splitext(os.path.basename(args.input_image))[0]  # Extract base name without extension
output_file =  f"{base_name}_featuremaps.png"
draw_feature_map(model, args.input_image, transform, output_file, args.embedding_size, prediction)
