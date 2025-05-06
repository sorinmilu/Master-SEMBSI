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


# Preprocess image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def predict(image_path, object_threshold, nms_threshold):

    original_image = Image.open(image_path).convert("RGB")
    dir_name = os.path.dirname(image_path)
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    box_image_path = os.path.join(dir_name, f"{base_name}_box.png")
    nms_image_path = os.path.join(dir_name, f"{base_name}_nms.png")

    original_width, original_height = original_image.size

    # send the image to the model
    image = transform(original_image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        output = model(image)  # Shape: (1, 7, 7, 5)

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
        draw.rectangle([x1, y1, x2, y2], outline=outline_color, width=3)

    image_copy.save(box_image_path)

    # Apply NMS using the first column as probability score and coordinates
    indices = nms(boxes, scores, iou_threshold=nms_threshold)

    draw = ImageDraw.Draw(original_image)

    for idx in indices:
        box = boxes[idx].cpu().numpy()
        x1, y1, x2, y2 = map(int, box)
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)

    original_image.save(nms_image_path)


description = 'Inference program for yolo-like regression model'

parser = argparse.ArgumentParser(prog='infer.py',
                                 description=description,
                                 usage='extract_directory_mtcnn.py -in <input_directory> -ot <output_directory>',
                                 epilog="This program gets a directory with jpeg images as an argument, runs the MTCNN face detection model on each image and extracts the image fragments and writes them to the output directory")
parser.add_argument("-i", "--image", required=True, help="test image path", type=str)
parser.add_argument("-m", "--model", help="File that holds the model", type=str)
parser.add_argument("-g", "--grid_size", help="Grid size of the model (default 7) - it depends on the model definition", type=int, default=7)
parser.add_argument("-ot", "--object_threshold", type=float, default = 0.7,  help="Threshold above wich a cell becomes a hit. Default 0.5")
parser.add_argument("-nt", "--nms_iou_threshold", type=float, default = 0.5,  help="Non maximum suppression IOU threshold. Value between 0 and 1, usually 0.5, Less will collapse boxes more")


args = parser.parse_args()

if args.image:
    CHECK_FILE = os.path.isfile(args.image)
    if not CHECK_FILE:
        print('Input file not found')
        sys.exit(1)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Load trained model
model = YOLOFaceCNN(grid_size=args.grid_size).to(DEVICE)
model.load_state_dict(torch.load(args.model, map_location=torch.device('cpu'), weights_only=True))
model.eval()

predict(args.image, args.object_threshold, args.nms_iou_threshold)
