import argparse
import sys
import os
import cv2
import time
import glob
from tqdm import tqdm

description = 'OpenCV Haar cascade Multimodel directory extractor.'

parser = argparse.ArgumentParser(prog='extract_directory.py',
                                 description=description,
                                 usage='extract_directory.py -in <input_directory> -ot <output_directory> -cxf <xml file with haar model>',
                                 epilog="This program gets a directory with jpeg images as an argument, runs the haar cascade model on each image and writes label files in the output directory")
parser.add_argument("-in", "--input_dir", required=True, help="Directory with input images", type=str)
parser.add_argument("-ot", "--outdir" , help="Directory for writing output images", type=str)
parser.add_argument("-cxf", "--haarcascade_face", required=True, help="haar cascade xml for face detection", type=str)
parser.add_argument("-hsff", "--haarScaleFactor_face", help="Parameter specifying how much the image size is reduced at each image scale for face detection model", type=float, default=1.025)
parser.add_argument("-hmnf", "--haarMinNeighbours_face", help="Parameter specifying how many neighbors each candidate rectangle should have to retain it - face detection", type=int, default=25)
parser.add_argument("-hmsf", "--haarMinSize_face", help="Minimum possible object size. Objects smaller than that are ignored - face detection.", type=int, default=50)
parser.add_argument("-img", "--generate_images", type=str, choices=["True", "False"], default="True", help="Set to 'False' to disable saving cropped face images (default: True)")


start_time = time.time()

args = parser.parse_args()

if args.input_dir:
    CHECK_IDIR = os.path.isdir(args.input_dir)
    if not CHECK_IDIR:
        print('Input directory not found')
        sys.exit(1)
    else:
        print('Working directory:' + args.input_dir)

if not args.haarcascade_face:
    print('Please provide a cascade.xml model (full path) for face detection')
    parser.print_help(sys.stderr)
    sys.exit(1)

if args.outdir:
    CHECK_HOUTDIR = os.path.isdir(args.outdir)
    if not CHECK_HOUTDIR:
        print("Creating output haar directory: " + args.outdir)
        os.makedirs(args.outdir)

haarcascade_face = cv2.CascadeClassifier(args.haarcascade_face)

jpg_files = glob.glob(os.path.join(args.input_dir, "*.jpg"))

for file in tqdm(jpg_files, desc="Processing images", unit="image"):
    img = cv2.imread(file)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = haarcascade_face.detectMultiScale(
                gray,
                scaleFactor=args.haarScaleFactor_face,
                minNeighbors=args.haarMinNeighbours_face,
                minSize=(args.haarMinSize_face, args.haarMinSize_face)
            )
    
    if len(faces) > 0:
        # Create a label file for the current image
        label_file_path = os.path.join(args.outdir, os.path.basename(file).replace(".jpg", ".txt"))
        with open(label_file_path, 'w') as label_file:
            for i, (x, y, w, h) in enumerate(faces):
                if args.generate_images == "True":
                    face = img[y:y+h, x:x+w]  # Crop the face

                    filename = os.path.basename(file)  # Get the image filename
                    name, ext = os.path.splitext(filename)
                    output_path = os.path.join(args.outdir, f"{name}_extracted{i}{ext}")

                    cv2.imwrite(output_path, face)

                label_file.write(f"0 {int(x + w / 2)} {int(y + h / 2)} {w} {h}\n")

                # label_file.write(f"0 {x} {y} {w} {h}\n")
        
print(f"Labels saved to: {label_file_path}")
