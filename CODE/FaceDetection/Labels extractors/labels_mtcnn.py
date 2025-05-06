import argparse
import cv2
import sys
import os
import time
import glob
from tqdm import tqdm
import numpy as np
from mtcnn import MTCNN


def resize_and_save_image(input_path, max_edge=2500):
    """Resize image while keeping aspect ratio (max edge ≤ 1000 px), convert PNG to JPG, save, and return image."""
    
    should_save = 0
    # Read image
    image = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
    if image is None:
        print(f"Skipping: {input_path} (cannot read)")
        return None

    # Convert PNG with alpha to JPG (remove transparency)
    if input_path.lower().endswith(".png") and image.shape[-1] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        should_save=1

    h, w = image.shape[:2]

    # Resize only if larger than max_edge
    if max(h, w) > max_edge:
        scale = max_edge / max(h, w)
        new_size = (int(w * scale), int(h * scale))
        image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
        should_save=1

    # Change extension to .jpg
    output_path = os.path.splitext(input_path)[0] + ".jpg"
    
    base, ext = os.path.splitext(input_path)
    if ext.lower() in [".png", ".jpeg"]:
        output_path = base + ".jpg"
        if ext.lower() == ".jpeg":
            os.rename(input_path, output_path)
            print(f"Renamed: {input_path} → {output_path}")
    else:
        output_path = input_path  # Keep the original name for already .jpg files

    # Save image as JPG (overwrite if exists)
    if should_save == 1:
        cv2.imwrite(output_path, image, [cv2.IMWRITE_JPEG_QUALITY, 90])
        print(f"Saved: {output_path}")

    # Return the processed image
    return image



description = 'MTCNN face detection model directory extractor.'

parser = argparse.ArgumentParser(prog='extract_directory_mtcnn.py',
                                 description=description,
                                 usage='extract_directory_mtcnn.py -in <input_directory> -ot <output_directory>',
                                 epilog="This program gets a directory with jpeg images as an argument, runs the MTCNN face detection model on each image and extracts the image fragments and writes them to the output directory")
parser.add_argument("-in", "--input_dir", required=True, help="Directory with input images", type=str)
parser.add_argument("-ot", "--outdir", help="Directory for writing output images", type=str)
parser.add_argument("-img", "--generate_images", type=str, choices=["True", "False"], default="True", help="Set to 'False' to disable saving cropped face images (default: True)")
parser.add_argument("-lt", "--label_type", help="Types of boxes coordinates: bbox, centerbbox, centernorm", type=str, default="centernorm")


start_time = time.time()

args = parser.parse_args()

if args.input_dir:
    CHECK_IDIR = os.path.isdir(args.input_dir)
    if not CHECK_IDIR:
        print('Input directory not found')
        sys.exit(1)
    else:
        print('Working directory:' + args.input_dir)

if args.outdir:
    CHECK_HOUTDIR = os.path.isdir(args.outdir)
    if not CHECK_HOUTDIR:
        print("Creating output directory: " + args.outdir)
        os.makedirs(args.outdir)

# Initialize the MTCNN detector
detector = MTCNN()

image_extensions = ["*.jpg", "*.jpeg", "*.png"]

jpg_files = []
for ext in image_extensions:
    jpg_files.extend(glob.glob(os.path.join(args.input_dir, ext)))

# Get list of .jpg files in input directory
# jpg_files = glob.glob(os.path.join(args.input_dir, "*.jpg"))

# Use tqdm to track progress while iterating through files list
for file in tqdm(jpg_files, desc="Processing images", unit="image"):

    # img = cv2.imread(file)
    # imh, imw, _ = img.shape
    # img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    resized_img = resize_and_save_image(file, max_edge=2500)
    imh, imw, _ = resized_img.shape
    # Detect faces in the image
    results = detector.detect_faces(resized_img)

    faces = []
    for result in results:
        confidence = result['confidence']
        
        if confidence > 0.5:  # Filter out weak detections
            x, y, w, h = result['box']
            faces.append((x, y, w, h))

    if len(faces) > 0:
        # Create a label file for the current image
        label_file_path = os.path.join(args.outdir, os.path.basename(file).replace(".jpg", ".txt"))
        with open(label_file_path, 'w') as label_file:
            for i, (x, y, w, h) in enumerate(faces):
                if args.generate_images == "True":
                    face = resized_img[y:y+h, x:x+w]  # Crop the face
                    filename = os.path.basename(file)  # Get the image filename
                    name, ext = os.path.splitext(filename)
                    output_path = os.path.join(args.outdir, f"{name}_extracted{i}{ext}")

                    cv2.imwrite(output_path, face)

                # Write the bounding box coordinates to the label file (format: x y w h)
                if args.label_type == 'centernorm':
                    label_file.write(f"1 {(x + w / 2) / imw} {(y + h / 2) / imh} {w / imw} {h / imh}\n")
                elif args.label_type == 'centerbbox':
                    label_file.write(f"1 {int(x + w / 2)} {int(y + h / 2)} {w} {h}\n")
                elif args.label_type == 'bbox':
                    label_file.write(f"1 {x} {y} {w} {h}\n")
                elif args.label_type == 'gbox':
                    label_file.write(f"1 {x} {y} {x+w} {y+h}\n")
        
print(f"Labels saved to: {label_file_path}")
