import argparse
import os
import cv2
from tqdm import tqdm

def run_inference(file_path, output_file_path, model):
    # Load the Caffe model

    image = cv2.imread(file_path)

    height, width, _ = image.shape
    edge_width = 1 + max(1, int(width / 100))  # Dynamic rectangle thickness
    # Resize model input to match the image size
    model.setInputSize((image.shape[1], image.shape[0]))

    # Detect faces
    _, faces = model.detect(image)

    # Draw bounding boxes around detected faces
    if faces is not None:
        for face in faces:
            x, y, w, h = map(int, face[:4])  # Extract face coordinates
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), edge_width)  # Draw box

    cv2.imwrite(output_file_path, image)



description = 'Inference program for OpenCV Caffe face detection model that processes an entire directory'

parser = argparse.ArgumentParser(prog='infer_folder.py',
                                 description=description,
                                 usage='infer_folder.py -id <input_directory> -od <output_directory> -m <model_file> -c <config_file> -ct <confidence_threshold>',
                                 epilog="This program gets a directory with jpeg images as an argument, runs the model on each image, and replicates the input folder structure on the output folder. The output folder has to be empty.")

parser.add_argument("-id", "--input_directory", required=True, help="Directory with the testing images. Can have subdirectories.", type=str)
parser.add_argument("-od", "--output_directory", required=True, help="Directory with the inference results. It has to be empty. If the given directory does not exist, it will be created. The structure of the input directory (subdirectories, etc.) will be recreated.", type=str)
parser.add_argument("-m", "--model", required=True, help="Path to the Caffe model file (.caffemodel).", type=str)
parser.add_argument("-ct", "--confidence_threshold", help="Minimum confidence score for a detected face to be considered valid. Higher values (e.g., 0.9) make detection stricter (fewer false positives). Lower values (e.g., 0.5) allow more detections but may increase false positives.", type=float, default=0.9)
parser.add_argument("-nms", "--nms_threshold", help="Non-Maximum Suppression (NMS) threshold. Helps remove duplicate bounding boxes for the same face. Lower values (e.g., 0.3) keep only the best face box and remove overlapping detections. Higher values (e.g., 0.7) allow more overlapping detections.", type=float, default=0.5)
parser.add_argument("-tk", "--top_k_candidates", help="The maximum number of face candidates to process before filtering. Usually set to a high number like 5000 so the detector does not miss faces.", type=int, default=5000)


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

face_detector = cv2.FaceDetectorYN_create(args.model, "", (320, 320), args.confidence_threshold, args.nms_threshold, args.top_k_candidates)

# Process files with progress bar
with tqdm(total=total_files, desc="Processing files", unit="file") as pbar:
    for file_path in file_list:
        rel_path = os.path.relpath(file_path, input_dir)
        target_path = os.path.join(output_dir, os.path.dirname(rel_path))
        os.makedirs(target_path, exist_ok=True)
        output_file_path = os.path.join(target_path, os.path.basename(file_path))

        # Run inference
        run_inference(file_path, output_file_path, face_detector)

        pbar.update(1)

print("Processing complete.")