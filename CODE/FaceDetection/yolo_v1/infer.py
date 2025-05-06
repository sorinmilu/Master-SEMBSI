import torch
from torchvision import transforms
from PIL import Image, ImageDraw
from cnn_yolo_face import YOLOFaceCNN
import argparse

# Load the trained model
def load_model(model_path, grid_size=7, device="cpu"):
    model = YOLOFaceCNN(grid_size=grid_size).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()  # Set the model to evaluation mode
    return model

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
    for box in boxes:
        draw.rectangle(box, outline="red", width=2)
    return image

# Main inference function
def run_inference(image_path, model_path, output_path, grid_size=7, threshold=0.5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model_path, grid_size=grid_size, device=device)

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
    print(f"Output saved to {output_path}")


description = 'Inference program for yolo-like regression model'

parser = argparse.ArgumentParser(prog='infer.py',
                                 description=description,
                                 usage='extract_directory_mtcnn.py -in <input_directory> -ot <output_directory>',
                                 epilog="This program gets a directory with jpeg images as an argument, runs the MTCNN face detection model on each image and extracts the image fragments and writes them to the output directory")
parser.add_argument("-i", "--image", required=True, help="test image path", type=str)
parser.add_argument("-m", "--model", help="File that holds the model", type=str)
parser.add_argument("-o", "--output", required=True, help="output image", type=str)

args = parser.parse_args()

run_inference(args.image, args.model, args.output, grid_size=7, threshold=0.5)