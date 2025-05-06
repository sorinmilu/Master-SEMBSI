import argparse
import torch
from torchvision import transforms
from PIL import Image, ImageDraw
from cnn_yolo_face import YOLOFaceCNN
import matplotlib.pyplot as plt
import os
import cv2
import numpy as np
from torchvision.ops import nms


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GRID_SIZE = 7

# Load trained model
model = YOLOFaceCNN(grid_size=GRID_SIZE).to(DEVICE)
model.load_state_dict(torch.load("yolo_big_data_57_2.8357.pth"))
model.eval()

# Preprocess image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# def predict(image_path): 
#     image = Image.open(image_path).convert("RGB")
#     image = transform(image).unsqueeze(0).to(DEVICE)

#     with torch.no_grad():
#         output = model(image)

#     # Convert to probabilities (if not already in range 0-1)
#     output = torch.sigmoid(output)  # Ensure sigmoid activation

#     # Check if any cell has a score above threshold
#     has_face = (output > 0.5).any().item()

#     return "Face" if has_face else "No Face"

# def predict(image_path):
#     image = Image.open(image_path).convert("RGB")
#     image = transform(image).unsqueeze(0).to(DEVICE)

#     with torch.no_grad():
#         output = model(image)

#     score = output.mean().item()
#     return "Face" if score > 0.5 else "No Face"


def predict(image_path):
    original_image = Image.open(image_path).convert("RGB")
    original_height, original_width = original_image.size
    image = transform(original_image).unsqueeze(0).to(DEVICE)

    dir_name = os.path.dirname(image_path)
    base_name = os.path.splitext(os.path.basename(image_path))[0]

    # Create the heatmap file path in the same directory
    heatmap_path = os.path.join(dir_name, f"{base_name}_heatmap.png")  # Change extension if needed
    heatmap_oar_path = os.path.join(dir_name, f"{base_name}_heatmap_oar.png")
    overlay_image_path = os.path.join(dir_name, f"{base_name}_heatmap_overlay.png")
    box_image_path = os.path.join(dir_name, f"{base_name}_box.png")
    nms_image_path = os.path.join(dir_name, f"{base_name}_nms.png")

    with torch.no_grad():
        output = model(image)  # Shape: (1, 7, 7, 5)
        # raw_logits, output = model(image)

    # print("Raw logits:", raw_logits[..., 0])  # Print logits for class probabilities
    # print("Post-sigmoid:", output[..., 0])

    # print(output.shape)

    confidence_map = output[0, :, :, 0]  # Extract confidence scores (shape: 7x7)
    resized_confidence_map = np.array(confidence_map)
    resized_confidence_map = cv2.resize(resized_confidence_map, (original_width, original_height), interpolation=cv2.INTER_LINEAR)
    resized_confidence_map = np.transpose(resized_confidence_map)

    for row in confidence_map:
        print("  ".join(f"{conf:.2f}" for conf in row))


    plt.imshow(confidence_map, cmap='hot', interpolation='nearest', vmin=0, vmax=1)
    plt.colorbar(label='Confidence')
    plt.title('Confidence Map Heatmap')
    plt.savefig(heatmap_path)  # You can change the file name and format if needed
    plt.close()  # Close the plot to free up memory


    plt.imshow(resized_confidence_map, cmap='hot', interpolation='nearest', vmin=0, vmax=1)
    plt.colorbar(label='Confidence')
    plt.title('Confidence Map Heatmap')
    plt.savefig(heatmap_oar_path)  # You can change the file name and format if needed
    plt.close()  # Close the plot to free up memory

    fig, ax = plt.subplots()
    ax.imshow(original_image)

    # Overlay the heatmap with 50% opacity
    heatmap = ax.imshow(resized_confidence_map, cmap='hot', interpolation='nearest', alpha=0.5, vmin=0, vmax=1)

    # Add a colorbar to the overlay
    fig.colorbar(heatmap, ax=ax, label='Confidence')

# Save the overlapped image
    plt.title('Overlayed Heatmap on Original Image')
    plt.savefig(overlay_image_path)  # Save the overlay image
    plt.close()  # Close the plot to free up memory


    




    output = output.squeeze(0)  # Remove batch dim
    print(output.shape)
    prob_map = output[..., 0]  # Extract probability grid
    # print(prob_map)

    #
    # list all boxes
    

    rows, cols, features = output.shape  

    # Loop through each row and column
    for i in range(rows):  
        for j in range(cols):  
            print(output[i, j])


    # Filter out boxes with probability less than 0.5
    filtered_boxes = output[output[:, :, 0] > 0.5]

    # Convert normalized coordinates to actual pixel coordinates based on the image size

    x_center = (filtered_boxes[:, 1] * original_width).cpu().numpy()
    y_center = (filtered_boxes[:, 2] * original_height).cpu().numpy()
    box_width = (filtered_boxes[:, 3] * original_width).cpu().numpy()
    box_height = (filtered_boxes[:, 4] * original_height).cpu().numpy()

    # Create bounding boxes in the form (x1, y1, x2, y2) for NMS
    x1 = x_center - box_width / 2
    y1 = y_center - box_height / 2
    x2 = x_center + box_width / 2
    y2 = y_center + box_height / 2


    #--- prenms
    # Stack the boxes and probabilities for NMS
    boxes = torch.tensor([x1, y1, x2, y2]).T
    scores = torch.tensor(filtered_boxes[:, 0])

    # Convert to tensors
    # boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
    # scores_tensor = torch.tensor(scores, dtype=torch.float32)

    image_copy = original_image.copy()
    draw = ImageDraw.Draw(image_copy)

    for box in boxes:
        print(box)

        x1, y1, x2, y2 = box
        outline_color = (255, 0, 0)  # Red for example, change it as needed
        draw.rectangle([x1, y1, x2, y2], outline=outline_color, width=3)
    
        # # Optionally, you can annotate the box with the score
        # draw.text((x1, y1), f'{score:.2f}', fill=(255, 255, 255))  # Dis

    output_image_path = "output_image_with_boxes.png"
    image_copy.save(box_image_path)


    print("--------------- post nms -------------")
    # Draw the remaining boxes on the image


    # Apply NMS using the first column as probability score and coordinates
    indices = nms(boxes, scores, iou_threshold=0.5)

    draw = ImageDraw.Draw(original_image)

    for idx in indices:
        box = boxes[idx].cpu().numpy()
        x1, y1, x2, y2 = map(int, box)


        print(x1, y1, x2, y2)
        # Draw the box on the image copy
        
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)


    # print("BOX image path: " + box_image_path)
    original_image.save(nms_image_path)



























    print(confidence_map.shape)
    print(prob_map.shape)


    y_idxs, x_idxs = np.where(prob_map > 0.5)

    # # Convert to list of coordinates (x, y)
    # bboxes = [(output[y, x, 0], output[y, x, 1], output[y, x, 2], output[y, x, 3]) for y, x in zip(y_idxs, x_idxs)]

    # # Print results
    # for (x, y), (bx, by, bw, bh) in zip(zip(x_idxs, y_idxs), bboxes):
    #     print(f"Cell ({x}, {y}) -> Box (x: {bx}, y: {by}, w: {bw}, h: {bh})")


    bboxes = [(output[y, x, 0], output[y, x, 1], output[y, x, 2], output[y, x, 3]) for y, x in zip(y_idxs, x_idxs)]
    scores = [prob_map[y, x] for y, x in zip(y_idxs, x_idxs)]  # Extract scores (probabilities)

    # Print results including probability
    for (x, y), (bx, by, bw, bh), prob in zip(zip(x_idxs, y_idxs), bboxes, scores):
        print(f"Cell ({x}, {y}) -> Box (x: {bx}, y: {by}, w: {bw}, h: {bh}), Probability: {prob:.4f}")

    best_idx = torch.argmax(prob_map)  # Find best grid cell
    best_y, best_x = divmod(best_idx.item(), model.grid_size)

    if prob_map[best_y, best_x] < 0.5:
        return "No Face", None

    x_offset, y_offset, w, h = output[best_y, best_x, 1:].tolist()
    
    img_w, img_h = image.shape[2], image.shape[3]
    cell_w, cell_h = img_w / model.grid_size, img_h / model.grid_size

    x_center = (best_x + x_offset) * cell_w
    y_center = (best_y + y_offset) * cell_h
    w, h = w * img_w, h * img_h

    x1, y1 = int(x_center - w / 2), int(y_center - h / 2)
    x2, y2 = int(x_center + w / 2), int(y_center + h / 2)

    return "Face", (x1, y1, x2, y2)


description = 'Tester'

parser = argparse.ArgumentParser(prog='extract_directory_mtcnn.py',
                                 description=description,
                                 usage='extract_directory_mtcnn.py -in <input_directory> -ot <output_directory>',
                                 epilog="This program gets a directory with jpeg images as an argument, runs the MTCNN face detection model on each image and extracts the image fragments and writes them to the output directory")
parser.add_argument("-i", "--image", required=True, help="test image path", type=str)
# parser.add_argument("-ot", "--outdir", help="Directory for writing output images", type=str)

args = parser.parse_args()

# Test on an image
# print(predict("data/images/face_000238.jpg"))
print(predict(args.image))
