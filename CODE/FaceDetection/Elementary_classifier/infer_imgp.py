import torch
import argparse
import os
from PIL import Image, ImageDraw
import torchvision.transforms as transforms
import torchvision.ops as ops
from cnn_class import CNNClassifier
from tqdm import tqdm

# Define transformation (resize, normalization)
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def iou(box1, box2):
    """Compute Intersection over Union (IoU) of two boxes."""
    x1 = max(box1['topx'], box2['topx'])
    y1 = max(box1['topy'], box2['topy'])
    x2 = min(box1['bottomx'], box2['bottomx'])
    y2 = min(box1['bottomy'], box2['bottomy'])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = box1['width'] * box1['height']
    area2 = box2['width'] * box2['height']
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0

def is_inside(inner, outer):
    """Check if inner box is fully inside outer box."""
    return (inner['topx'] >= outer['topx'] and
            inner['topy'] >= outer['topy'] and
            inner['bottomx'] <= outer['bottomx'] and
            inner['bottomy'] <= outer['bottomy'])

def nms_filter(boxes, iou_threshold=0.3):
    """Perform NMS-like filtering to remove overlapping and contained boxes."""
    # Sort boxes by prediction confidence (high to low)
    boxes = sorted(boxes, key=lambda b: b['prediction'], reverse=True)
    final_boxes = []

    while boxes:
        current = boxes.pop(0)
        final_boxes.append(current)
        
        boxes = [
            box for box in boxes 
            if not is_inside(box, current) and iou(box, current) < iou_threshold
        ]

    return final_boxes

def compute_relative_iou(box1, box2):
    """Compute IoU relative to the smaller box."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    smaller_area = min(area1, area2)

    return intersection / smaller_area if smaller_area > 0 else 0

def apply_tv_nmsa(boxes_data, iou_threshold=0.5):
    boxes = torch.tensor([[b['topx'], b['topy'], b['bottomx'], b['bottomy']] for b in boxes_data], dtype=torch.float32)
    scores = torch.tensor([b['prediction'] for b in boxes_data], dtype=torch.float32)
    indices = ops.nms(boxes, scores, iou_threshold)

    selected_boxes = [boxes_data[i] for i in indices.tolist()]
    return selected_boxes    

def apply_tv_nms(boxes_data, iou_threshold=0.5):
    boxes = torch.tensor([[b['topx'], b['topy'], b['bottomx'], b['bottomy']] for b in boxes_data], dtype=torch.float32)
    scores = torch.tensor([b['prediction'] for b in boxes_data], dtype=torch.float32)

    # Perform custom NMS with IoU relative to the smaller box
    keep = []
    indices = scores.argsort(descending=True)

    while indices.numel() > 0:
        current_idx = indices[0].item()
        keep.append(current_idx)

        remaining = indices[1:]
        new_indices = []

        for idx in remaining:
            idx = idx.item()
            if compute_relative_iou(boxes[current_idx].tolist(), boxes[idx].tolist()) < iou_threshold:
                new_indices.append(idx)

        indices = torch.tensor(new_indices, dtype=torch.long)

    selected_boxes = [boxes_data[i] for i in keep]
    return selected_boxes

# def apply_non_max_suppression(boxes, iou_threshold=0.5):
#     def iou(box1, box2):
#         x1 = max(box1["topx"], box2["topx"])
#         y1 = max(box1["topy"], box2["topy"])
#         x2 = min(box1["bottomx"], box2["bottomx"])
#         y2 = min(box1["bottomy"], box2["bottomy"])
        
#         inter_area = max(0, x2 - x1) * max(0, y2 - y1)
#         box1_area = (box1["bottomx"] - box1["topx"]) * (box1["bottomy"] - box1["topy"])
#         box2_area = (box2["bottomx"] - box2["topx"]) * (box2["bottomy"] - box2["topy"])
        
#         union_area = box1_area + box2_area - inter_area
#         return inter_area / union_area if union_area else 0
    
#     def is_inside(outer_box, inner_box):
#         """Returns True if at least 80% of inner_box is inside outer_box."""
#         # Compute inner box area
#         inner_area = (inner_box["bottomx"] - inner_box["topx"]) * (inner_box["bottomy"] - inner_box["topy"])

#         # Compute intersection area
#         x1 = max(inner_box["topx"], outer_box["topx"])
#         y1 = max(inner_box["topy"], outer_box["topy"])
#         x2 = min(inner_box["bottomx"], outer_box["bottomx"])
#         y2 = min(inner_box["bottomy"], outer_box["bottomy"])

#         intersection_width = max(0, x2 - x1)
#         intersection_height = max(0, y2 - y1)
#         intersection_area = intersection_width * intersection_height

#         # Check if at least 80% of the inner box is inside
#         return intersection_area / inner_area >= 0.8

#     boxes = [b for b in boxes if b["is_face"]]
#     boxes.sort(key=lambda b: b["prediction"], reverse=True)
    
#     selected_boxes = []
#     while boxes:
#         best_box = boxes.pop(0)
#         selected_boxes.append(best_box)
#         boxes = [b for b in boxes if iou(best_box, b) < iou_threshold and not is_inside(best_box, b)]

    
#     return selected_boxes

"""
Generates a list of boxes that have the same aspect ratio as the input image. Each level will halve both dimensions, maintaining the aspect ratio 
At each level (box size) it moves the box with <step> pixels and runs the inference, cropping and transforming the fragment as needed

Parameters:
    -----------
    model : torch.nn.Module
        The object detection model used for inference.
    input_image : PIL.Image or numpy.ndarray
        The input image on which object detection is performed.
    transform : callable
        A preprocessing function (e.g., for resizing and normalizing) 
        applied to each image before passing it to the model.
    num_levels : int, optional
        The number of pyramid levels (default is 4). Higher values allow for 
        finer-scale detection but increase computation time.
    step : int, optional
        The scaling step size between levels (default is 5). A smaller step 
        means finer granularity in the pyramid.

    Returns:
    --------
    List[Tuple[float, float, List[Detection]]]
        A list of detections at different pyramid levels, where each entry 
        contains:
        - The scale factor used.
        - The corresponding image size.
        - A list of detected objects from the model.

    Example:
    --------
    >>> detections = detect_with_image_pyramid(model, image, transform, num_levels=3, step=4)
    >>> for scale, size, objs in detections:
    >>>     print(f"Scale: {scale}, Image Size: {size}, Objects: {len(objs)}")

"""

def detect_with_image_pyramid(model, input_image, transform, num_levels=4, step=5):
    image = Image.open(input_image)
    width, height = image.size
    
    levels = [(width // (2 ** i), height // (2 ** i)) for i in range(1, num_levels + 1)]

    total_boxes = sum(((height - h) // step + 1) * ((width - w) // step + 1) for w, h in levels)
    progress_interval = max(total_boxes // 20, 1)  # 5% increments, avoid division by zero
    processed_boxes = 0
    image_pyramid = []
    with torch.no_grad():
        for level, (box_width, box_height) in enumerate(levels, start=1):
            for topy in range(0, height - box_height + 1, step):  # Slide vertically with step
                for topx in range(0, width - box_width + 1, step):  # Slide horizontally with step
                    bottomx = topx + box_width
                    bottomy = topy + box_height
                    print(bottomx, bottomy, box_width, box_height, end="")
                    cropped_image = image.crop((topx, topy, bottomx, bottomy))
                    cropped_image = transform(cropped_image).unsqueeze(0) 
                    output = model(cropped_image)
                    prediction = output.item()
                    if prediction > 0.6:
                        is_face=True
                        print (" -> 1")
                    else:
                        is_face=False    
                        print (" -> 0")
                    image_pyramid.append({
                        "level": level,
                        "topx": topx,
                        "topy": topy,
                        "bottomx": bottomx,
                        "bottomy": bottomy,
                        "width": box_width,
                        "height": box_height,
                        "prediction": prediction,
                        "is_face": is_face
                    })
                    processed_boxes += 1
                    if processed_boxes % progress_interval == 0:
                        print(f"Progress: {processed_boxes / total_boxes * 100:.1f}%")
    


    return image_pyramid


"""
Generates a list of squares starting with the smallest size of the image. The box size is reduced each level by <size_reduction> parameter 
as long as the size does not get smaller than <min_box_size> 
At each level (box size) it moves the box with <step> pixels and runs the inference, cropping and transforming the fragment as needed

Parameters:
    -----------
    model : torch.nn.Module
        The object detection model used for inference.
    input_image : PIL.Image or numpy.ndarray
        The input image on which object detection is performed.
    transform : callable
        A preprocessing function (e.g., for resizing and normalizing) 
        applied to each image before passing it to the model.
    min_box_size : int, optional
        The minimum size of the image at the smallest pyramid level (default is 128).
    size_reduction : int, optional
        The amount by which the image size is reduced at each pyramid level (default is 60).
    step : int, optional
        The scaling step size between pyramid levels (default is 10). A smaller step 
        results in finer granularity in the pyramid.

    Returns:
    --------
    List[Tuple[int, int, List[Detection]]]
        A list of detections at different pyramid levels, where each entry contains:
        - The size of the square image.
        - The corresponding image dimensions.
        - A list of detected objects from the model.

    Example:
    --------
    >>> detections = generate_square_image_pyramid(model, image, transform, min_box_size=128, size_reduction=50, step=5)
    >>> for size, dims, objs in detections:
    >>>     print(f"Size: {size}, Dimensions: {dims}, Objects: {len(objs)}")

"""

# def detect_with_square_image_pyramid(model, input_image, transform, min_box_size = 128, size_reduction=100, step=10):
#     image = Image.open(input_image)
#     width, height = image.size
#     min_edge = min(height, width)

#     print(width, height)

#     # List to hold the windows data
#     image_pyramid = []

#     move_horizontally = width > height

#     level = 0
#     current_size = min_edge

#     while current_size >= min_box_size:
#         if level == 0:
#             # The first level only moves in the smaller dimension
#             if move_horizontally:
#                 print("horizontaly")
#                 step_x = step
#                 step_y = 0  # Only move horizontally for the first level
#             else:
#                 print("verticaly")
#                 step_x = 0  # Only move vertically
#                 step_y = step
#         else:
#             step_x = step
#             step_y = step

#         print(step_y, step_x)


#         # Move window across the image
#         for topy in range(0, height - current_size + 1, step_y if step_y > 0 else 1):
#             for topx in range(0, width - current_size + 1, step_x if step_x > 0 else 1):
#                 bottomx = topx + current_size
#                 bottomy = topy + current_size
#                 print(bottomx, bottomy, current_size, end="")
#                 cropped_image = image.crop((topx, topy, bottomx, bottomy))
#                 cropped_image = transform(cropped_image).unsqueeze(0) 
#                 output = model(cropped_image)
#                 prediction = output.item()
#                 if prediction > 0.8:
#                     is_face=True
#                     print (" -> 1")
#                 else:
#                     is_face=False    
#                     print (" -> 0")

#                 # Store window data in a dictionary
#                 window_data = {
#                     'level': level,
#                     'topy': topy,
#                     'topx': topx,
#                     'bottomx': bottomx,
#                     'bottomy': bottomy,
#                     'width': current_size,
#                     'height': current_size,
#                     'prediction' : prediction,
#                     "is_face": is_face
#                 }
#                 image_pyramid.append(window_data)

#         # Reduce the size for the next level
#         level += 1
#         current_size -= size_reduction

#     return image_pyramid


def detect_with_square_image_pyramid(model, input_image, transform, min_box_size=128, size_reduction=100, step=10, batch_size=16):
    image = Image.open(input_image)
    width, height = image.size
    min_edge = min(height, width)
    
    image_pyramid = []
    move_horizontally = width > height
    
    level = 0
    current_size = min_edge
    
    batch_images = []  # List to accumulate transformed image crops
    batch_windows = []  # List to track the corresponding window metadata
    
    # Calculate total number of boxes for tqdm
    total_boxes = 0
    temp_size = min_edge
    while temp_size >= min_box_size:
        step_x, step_y = (step, 0) if move_horizontally and level == 0 else (0, step) if not move_horizontally and level == 0 else (step, step)
        total_boxes += ((max(1, (height - temp_size) // step_y) if step_y > 0 else 1) * 
                        (max(1, (width - temp_size) // step_x) if step_x > 0 else 1))
        temp_size -= size_reduction
        level += 1
    
    level = 0
    current_size = min_edge

    print(f"total boxes: {total_boxes}")

    with tqdm(total=total_boxes, desc="Processing image pyramid") as pbar:
        while current_size >= min_box_size:
            step_x, step_y = (step, 0) if move_horizontally and level == 0 else (0, step) if not move_horizontally and level == 0 else (step, step)
            
            for topy in range(0, height - current_size + 1, step_y if step_y > 0 else 1):
                for topx in range(0, width - current_size + 1, step_x if step_x > 0 else 1):
                    bottomx, bottomy = topx + current_size, topy + current_size
                    
                    cropped_image = image.crop((topx, topy, bottomx, bottomy))
                    batch_images.append(transform(cropped_image).unsqueeze(0))
                    batch_windows.append({
                        'level': level,
                        'topy': topy,
                        'topx': topx,
                        'bottomx': bottomx,
                        'bottomy': bottomy,
                        'width': current_size,
                        'height': current_size
                    })
                    
                    # Process batch when reaching batch_size
                    if len(batch_images) >= batch_size:
                        batch_tensor = torch.cat(batch_images, dim=0)  # Stack into a batch
                        outputs = model(batch_tensor).squeeze(1).tolist()  # Get predictions
                        
                        for i, prediction in enumerate(outputs):
                            batch_windows[i]['prediction'] = prediction
                            batch_windows[i]['is_face'] = prediction > 0.8
                            image_pyramid.append(batch_windows[i])
                        
                        batch_images.clear()
                        batch_windows.clear()
                    
                    pbar.update(1)  # Update progress bar
            
            level += 1
            current_size -= size_reduction
        
        # Process any remaining images in batch
        if batch_images:
            batch_tensor = torch.cat(batch_images, dim=0)
            outputs = model(batch_tensor).squeeze(1).tolist()
            
            for i, prediction in enumerate(outputs):
                batch_windows[i]['prediction'] = prediction
                batch_windows[i]['is_face'] = prediction > 0.8
                image_pyramid.append(batch_windows[i])
                pbar.update(1)  # Update progress bar
    
    return image_pyramid



"""

Draw detected faces on an image and save the result.

    This function takes an input image, overlays bounding boxes around detected faces 
    from different pyramid levels, and saves the modified image to the specified output path.

    Parameters:
    -----------
    image_path : str
        Path to the input image file.
    image_pyramid : List[]
        A list of detections from the image pyramid, where each entry contains at least:
        - topx - pixel coordinates of the top x corner
        - topy - pixel coordinates of the top y corner
        - bottomx - pixel coordinates of the bottom x corner
        - bottomy -  - pixel coordinates of the bottom y corner
    output_path : str
        Path where the output image with drawn faces will be saved.

    Returns:
    --------
    None
        The function saves the modified image to `output_path` and does not return a value.

    Example:
    --------
    >>> draw_faces_on_image("input.jpg", image_pyramid, "output.jpg")

"""

def draw_faces_on_image(image_path, image_pyramid, output_path):
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)
    
    for box in image_pyramid:
        if box["is_face"]:
            draw.rectangle(
                [(box["topx"], box["topy"]), (box["bottomx"], box["bottomy"])],
                outline="red",
                width=2
            )
    print("saving image: " + output_path)
    image.save(output_path)


parser = argparse.ArgumentParser(description="Face detection script with drawing and printing options.")

parser = argparse.ArgumentParser(
        prog="train", 
        description="Face detection script with drawing and printing options.",
        usage=" python3 infer_imgp.py -m <model_file> -i <image_file> -bs <image|square> -o <output_name>",
        epilog="This programs uses a multiscale sliding window technique to apply a classification method to object detection. It has two box shapes, one square and another one that has the same aspect ratio"
        )


parser.add_argument('-i', "--input_image", required=True, help="Path to input image")
parser.add_argument('-m', "--model", default="models/face_model.pth", help="Path to input image")
parser.add_argument('-bs', "--box_shape", help="The shape of the box used for detection. - square or image", default="square")
parser.add_argument('-s', "--step", default=10, type=int, help="The scaling step size between pyramid levels (default is 10). A smaller step results in finer granularity in the pyramid")
parser.add_argument('-o', "--output_image_root", default = "output", help="Name root of the output image. Two images will be generated, one without suffix and another with _nms suffix")
parser.add_argument('-e', "--embedding_size", default=512, help="size of the last fully connected layer", type=int)
parser.add_argument('-sr', "--square_size_reduction", default = 50, type=int, help="Size reduction step for square boxes - it will be ignored when image box is selected")
parser.add_argument('-bz', "--min_box_size", default = 128, type=int, help="Minimum box size for sliding window")
parser.add_argument('-lv', "--image_window_levels", default = 4, type=int, help="Number of halving steps for image sliding window - it will be ignored if square box is selected")


args = parser.parse_args()

if not os.path.isfile(args.input_image):
    print(f"Error: Input image '{args.input_image}' does not exist.")
    exit(1)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNNClassifier(embedding_size=args.embedding_size).to(DEVICE) 
model.load_state_dict(torch.load(args.model, weights_only=True, map_location=torch.device('cpu')))  # Load trained model


image_pyramid = []
faceboxes=0

if args.box_shape == 'square':
    image_pyramid = detect_with_square_image_pyramid(model, args.input_image, transform, min_box_size=args.min_box_size, size_reduction=args.square_size_reduction, step=args.step)

elif args.box_shape == 'image':
    detect_with_image_pyramid(model, args.input_image, transform)
    image_pyramid =  detect_with_image_pyramid(model, args.input_image, transform, num_levels=4, step=5)

else: 
    print(f"Box shape: {args.box_shape} wrong")

faceboxes=0

for box in image_pyramid:
    if box["is_face"]:
        print(box)
        faceboxes+=1

print(f"Faceboxes: {faceboxes} from total: {len(image_pyramid)}")

# final_boxes = apply_tv_nms(image_pyramid)
final_boxes = nms_filter(image_pyramid)

draw_faces_on_image(args.input_image, image_pyramid, args.output_image_root+".jpg")
draw_faces_on_image(args.input_image, final_boxes, args.output_image_root+"_nms.jpg")


