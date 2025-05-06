import argparse
import matplotlib.pyplot as plt
import os
import numpy as np
import cv2
from tqdm import tqdm
import FaceMultimodel as fmm





def run_inference(file_path, output_file_path, args):
    img = cv2.imread(file_path)
    height, width, _ = img.shape
    edge_width = 2 + max(1, int(width / 100))
    min_dim = min(width, height) // 10
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    report = fmm.detect_frame(gray, args)

    face_content = fmm.check_face_types(report, args)

    print(face_content)

    img = fmm.draw_face_content(img, face_content, args)
    cv2.imwrite(output_file_path, img)


description = 'Inference program for haar cascades face model that process an entire directory'

parser = argparse.ArgumentParser(prog='infer_folder.py',
                                 description=description,
                                 usage='infer_folder.py -id <input_directory> -od <output_directory> -m <model file>' ,
                                 epilog="This program gets a directory with jpeg images as an argument, runs the model on each image replicates the input folder structure on the output folder. The output folder has to be empty")

parser.add_argument("-id", "--input_directory", required=True, help="Directory with the testing images. Can have subdiorectories", type=str)
parser.add_argument("-od", "--output_directory", required=True, help="Directory with the inference results. It has to be empty. If the given directory does not exists, it will be created. The structure of the input directory (subdirectories, etc) will be recreated", type=str)

parser.add_argument("-cxf", "--haarcascade_face", required=True, help="haar cascade xml for face detection", type=str)
parser.add_argument("-cxe", "--haarcascade_eye", required=True, help="haar cascade xml for eye detection", type=str)
parser.add_argument("-cxn", "--haarcascade_nose", required=True, help="haar cascade xml for nose detection", type=str)
parser.add_argument("-cxm", "--haarcascade_mouth", required=True, help="haar cascade xml for mouth detection", type=str)
parser.add_argument("-hsff", "--haarScaleFactor_face", help="Parameter specifying how much the image size is reduced at each image scale for face detection model", type=float, default=1.025)
parser.add_argument("-hsfe", "--haarScaleFactor_eye", help="Parameter specifying how much the image size is reduced at each image scale for eye detection model", type=float, default=1.025)
parser.add_argument("-hsfn", "--haarScaleFactor_nose", help="Parameter specifying how much the image size is reduced at each image scale for nose detection model", type=float, default=1.025)
parser.add_argument("-hsfm", "--haarScaleFactor_mouth", help="Parameter specifying how much the image size is reduced at each image scale for mouth detection model", type=float, default=1.025)

parser.add_argument("-hmnf", "--haarMinNeighbours_face", help="Parameter specifying how many neighbors each candidate rectangle should have to retain it - face detection", type=int, default=25)
parser.add_argument("-hmne", "--haarMinNeighbours_eye", help="Parameter specifying how many neighbors each candidate rectangle should have to retain it - eye detection", type=int, default=50)
parser.add_argument("-hmnn", "--haarMinNeighbours_nose", help="Parameter specifying how many neighbors each candidate rectangle should have to retain it - nose detection", type=int, default=50)
parser.add_argument("-hmnm", "--haarMinNeighbours_mouth", help="Parameter specifying how many neighbors each candidate rectangle should have to retain it - mouth detection", type=int, default=150)

parser.add_argument("-hmsf", "--haarMinSize_face", help="Minimum possible object size. Objects smaller than that are ignored - face detection.", type=int, default=50)
parser.add_argument("-hmse", "--haarMinSize_eye", help="Minimum possible object size. Objects smaller than that are ignored - eye detection.", type=int, default=10)
parser.add_argument("-hmsn", "--haarMinSize_nose", help="Minimum possible object size. Objects smaller than that are ignored - nose detection.", type=int, default=10)
parser.add_argument("-hmsm", "--haarMinSize_mouth", help="Minimum possible object size. Objects smaller than that are ignored - mouth detection.", type=int, default=10)

args = parser.parse_args()

input_dir = args.input_directory
output_dir = args.output_directory


if not args.haarcascade_face:
    print('please provide a cascade.xml model (full path) for face detection')
    parser.print_help(sys.stderr)
    sys.exit(1)

if not args.haarcascade_eye:
    print('please provide a cascade.xml model (full path) for eye detection')
    parser.print_help(sys.stderr)
    sys.exit(1)

if not args.haarcascade_nose:
    print('please provide a cascade.xml model (full path) for nose detection')
    parser.print_help(sys.stderr)
    sys.exit(1)

if not args.haarcascade_mouth:
    print('please provide a cascade.xml model (full path) for mouth detection')
    parser.print_help(sys.stderr)
    sys.exit(1)


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

fmm.init_cascades(args)

# Process files with progress bar
with tqdm(total=total_files, desc="Processing files", unit="file") as pbar:
    for file_path in file_list:
        print(file_path)
        rel_path = os.path.relpath(file_path, input_dir)
        target_path = os.path.join(output_dir, os.path.dirname(rel_path))
        os.makedirs(target_path, exist_ok=True)
        output_file_path = os.path.join(target_path, os.path.basename(file_path))
        
        # Process file (for now, just copying as a placeholder)
        # shutil.copy(file_path, output_file_path)

        run_inference(file_path, output_file_path, args)
        
        pbar.update(1)

print("Processing complete.")
