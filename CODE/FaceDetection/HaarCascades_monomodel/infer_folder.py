import argparse
import matplotlib.pyplot as plt
import os
import numpy as np
import cv2
from tqdm import tqdm

def run_inference(haarcascade_face, file_path, output_file_path, hsff, hmnf, hmsf):
    img = cv2.imread(file_path)
    height, width, _ = img.shape
    edge_width = 2 + max(1, int(width / 100))
    min_dim = min(width, height) // hmsf
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = haarcascade_face.detectMultiScale(
                    gray,
                    scaleFactor=hsff,
                    minNeighbors=hmnf,
                    minSize=(min_dim, min_dim)
                )



    if len(faces) > 0:
        for i, (x, y, w, h) in enumerate(faces):
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), thickness=edge_width)

    cv2.imwrite(output_file_path, img)



description = 'Inference program for haar cascades face model that process an entire directory'

parser = argparse.ArgumentParser(prog='infer_folder.py',
                                 description=description,
                                 usage='infer_folder.py -id <input_directory> -od <output_directory> -m <model file>' ,
                                 epilog="This program gets a directory with jpeg images as an argument, runs the model on each image replicates the input folder structure on the output folder. The output folder has to be empty")

parser.add_argument("-id", "--input_directory", required=True, help="Directory with the testing images. Can have subdiorectories", type=str)
parser.add_argument("-od", "--output_directory", required=True, help="Directory with the inference results. It has to be empty. If the given directory does not exists, it will be created. The structure of the input directory (subdirectories, etc) will be recreated", type=str)
parser.add_argument("-m", "--model", help="File that holds the model", type=str)
parser.add_argument("-hsff", "--haarScaleFactor_face", help="Parameter specifying how much the image size is reduced at each image scale for face detection model", type=float, default=1.025)
parser.add_argument("-hmnf", "--haarMinNeighbours_face", help="Parameter specifying how many neighbors each candidate rectangle should have to retain it - face detection", type=int, default=50)
parser.add_argument("-hmsf", "--haarMinSize_face", help="Minimum possible object size. Objects smaller than that are ignored - face detection.", type=int, default=10)



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

haarcascade_face = cv2.CascadeClassifier(args.model)


# Process files with progress bar
with tqdm(total=total_files, desc="Processing files", unit="file") as pbar:
    for file_path in file_list:
        rel_path = os.path.relpath(file_path, input_dir)
        target_path = os.path.join(output_dir, os.path.dirname(rel_path))
        os.makedirs(target_path, exist_ok=True)
        output_file_path = os.path.join(target_path, os.path.basename(file_path))
        
        # Process file (for now, just copying as a placeholder)
        # shutil.copy(file_path, output_file_path)

        run_inference(haarcascade_face, file_path, output_file_path, hsff=args.haarScaleFactor_face , hmnf=args.haarMinNeighbours_face, hmsf=args.haarMinSize_face)
        
        pbar.update(1)

print("Processing complete.")
