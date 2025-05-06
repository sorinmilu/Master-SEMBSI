import argparse
import os
import cv2
from tqdm import tqdm
from mtcnn import MTCNN

def run_inference(detector, file_path, output_file_path):
    
    img = cv2.imread(file_path)
    height, width, _ = img.shape
    faces = detector.detect_faces(img)
    edge_width = 2 + max(1, int(width / 100))
    font_size = 0.5 * (width / 300) 

    for face in faces:
        x, y, w, h = face["box"]
        confidence = face["confidence"] 
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        if confidence is not None:
            text = f"{confidence:.2f}"
            cv2.putText(img, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, font_size, (0, 255, 0), 2)

    cv2.imwrite(output_file_path, img)

description = 'Inference program for MTCNN face detection model that processes an entire directory'

parser = argparse.ArgumentParser(prog='infer_folder.py',
                                 description=description,
                                 usage='infer_folder.py -id <input_directory> -od <output_directory> -mdc <min_detection_confidence>',
                                 epilog="This program gets a directory with jpeg images as an argument, runs the model on each image, and replicates the input folder structure on the output folder. The output folder has to be empty.")

parser.add_argument("-id", "--input_directory", required=True, help="Directory with the testing images. Can have subdirectories.", type=str)
parser.add_argument("-od", "--output_directory", required=True, help="Directory with the inference results. It has to be empty. If the given directory does not exist, it will be created. The structure of the input directory (subdirectories, etc.) will be recreated.", type=str)


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

detector = MTCNN()

# Process files with progress bar
with tqdm(total=total_files, desc="Processing files", unit="file") as pbar:
    for file_path in file_list:
        rel_path = os.path.relpath(file_path, input_dir)
        target_path = os.path.join(output_dir, os.path.dirname(rel_path))
        os.makedirs(target_path, exist_ok=True)
        output_file_path = os.path.join(target_path, os.path.basename(file_path))

        # Run inference
        run_inference(detector, file_path, output_file_path)

        pbar.update(1)

print("Processing complete.")