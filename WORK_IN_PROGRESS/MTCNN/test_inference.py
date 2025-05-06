import argparse
import os
from src import detect_faces
from PIL import Image, ImageDraw


parser = argparse.ArgumentParser(description="Face detection script with drawing and printing options.")
parser.add_argument('-i', "--input_image", required=True, help="Path to input image")
parser.add_argument('-o', "--output_image", help="Path to save output image")
parser.add_argument("--no-draw-boxes", action="store_false", dest="draw_boxes", help="Draw bounding boxes on the image")
parser.add_argument("--no-draw-landmarks", action="store_false", dest="draw_landmarks",help="Draw landmarks on the image")
parser.add_argument("--print-boxes", action="store_true", help="Print bounding box coordinates")
parser.add_argument("--print-landmarks", action="store_true", help="Print landmark coordinates")

args = parser.parse_args()


if not os.path.isfile(args.input_image):
    print(f"Error: Input image '{args.input_image}' does not exist.")
    exit(1)

output_image = args.output_image if args.output_image else "output.jpg"
if not args.output_image:
    print("No output image provided, using default: output.jpg")

input_image = Image.open(args.input_image)
bounding_boxes, landmarks = detect_faces(input_image)


draw = ImageDraw.Draw(input_image)

if args.draw_boxes:
    for box in bounding_boxes:
        x1, y1, x2, y2, _ = box  # Extract coordinates (ignore confidence score)
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)

if args.draw_landmarks:
    for landmark in landmarks:
        for i in range(5):  # Assuming 5 key points per face (eyes, nose, mouth corners)
            x, y = landmark[i], landmark[i + 5]  # First 5 values are x, next 5 are y
            draw.ellipse([x - 2, y - 2, x + 2, y + 2], fill="blue", outline="blue")  # Draw small circles

input_image.save(output_image)