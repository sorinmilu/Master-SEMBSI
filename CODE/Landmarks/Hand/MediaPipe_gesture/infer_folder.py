import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppresses INFO and WARNING logs
os.environ['CUDA_VISIBLE_DEVICES'] = '-1' 
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from PIL import Image
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from tqdm import tqdm
import argparse
import math



def draw_landmarks_on_image(rgb_image, detection_result, show_numbers=True):
    """
    Draws hand landmarks, handedness, and optionally landmark numbers on the image.

    Args:
        rgb_image: The input RGB image.
        detection_result: The detection result containing hand landmarks and handedness.
        show_numbers: Boolean, if True, draw landmark numbers on the image.

    Returns:
        The annotated image with landmarks and optionally landmark numbers drawn.
    """
    hand_landmarks_list = detection_result.hand_landmarks
    handedness_list = detection_result.handedness
    annotated_image = np.copy(rgb_image)
    image_height = rgb_image.shape[0]
    image_width = rgb_image.shape[1]


    font_scale = max(0.5, max(image_width,image_height) / 1500)  # Adjust the divisor to control scaling
    thickness = max(1, int(max(image_width,image_height) / 1000))  # Dynamically adjust thickness
    color = (0, 255, 0)

    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]
        handedness = handedness_list[idx]

        # Draw the hand landmarks.
        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        hand_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
        ])
        solutions.drawing_utils.draw_landmarks(
            annotated_image,
            hand_landmarks_proto,
            solutions.hands.HAND_CONNECTIONS,
            solutions.drawing_styles.get_default_hand_landmarks_style(),
            solutions.drawing_styles.get_default_hand_connections_style()
        )

        # Optionally draw landmark numbers
        if show_numbers:
            height, width, _ = annotated_image.shape
            for i, landmark in enumerate(hand_landmarks):
                x = int(landmark.x * width)
                y = int(landmark.y * height)
                cv2.putText(
                    annotated_image,
                    str(i),  # Landmark index
                    (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,  # Smaller font for numbers
                    (255, 0, 0),  # Blue color for numbers
                    thickness,
                    cv2.LINE_AA
                )

        # Get the top left corner of the detected hand's bounding box.
        height, width, _ = annotated_image.shape
        x_coordinates = [landmark.x for landmark in hand_landmarks]
        y_coordinates = [landmark.y for landmark in hand_landmarks]
        text_x = int(min(x_coordinates) * width)
        text_y = int(min(y_coordinates) * height) - 10

        # Draw handedness (left or right hand) on the image.
        # cv2.putText(
        #     annotated_image,
        #     f"{handedness[0].category_name}",
        #     (text_x, text_y),
        #     cv2.FONT_HERSHEY_DUPLEX,
        #     font_scale,
        #     color,
        #     thickness,
        #     cv2.LINE_AA
        # )

    return annotated_image


def process_image(recognizer, file_path, target_path):
    file_dir = os.path.dirname(file_path)
    file_base, file_ext = os.path.splitext(os.path.basename(file_path))

    outext = ".png"

    mp_image = mp.Image.create_from_file(file_path)
    recognition_result = recognizer.recognize(mp_image)

    original_image = mp_image.numpy_view()

    # Annotated image

    annotated_image_path = os.path.join(target_path, f"{file_base}_0_landmarks{file_ext}")
    annotated_image = draw_landmarks_on_image(original_image, recognition_result)

    hand_landmarks_list = recognition_result.hand_landmarks
    handedness_list = recognition_result.handedness
    gestures_list = recognition_result.gestures

    

    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]
        handedness = handedness_list[idx]

#        Detect the gesture for the current hand

        gestures = gestures_list[idx]

        height, width, _ = annotated_image.shape
        x_coordinates = [landmark.x for landmark in hand_landmarks]
        y_coordinates = [landmark.y for landmark in hand_landmarks]
        text_x = int(min(x_coordinates) * width)
        text_y = int(min(y_coordinates) * height) - 10

        # Draw handedness and gesture on the image
        text = f"{handedness[0].category_name}: {gestures[0].category_name}"  
        font_scale = max(0.5, width / 1500)  # Adjust font scale based on image width
        thickness = max(1, int(width / 500))  # Adjust thickness dynamically
        color = (0, 255, 0)  # Green color for text
        cv2.putText(annotated_image, text, (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX, font_scale, color, thickness, cv2.LINE_AA)


    annotated_image_pil = Image.fromarray(annotated_image)
    annotated_image_pil.save(annotated_image_path)




description = 'Inference program for face landmarks and emotion using Mediapipe library that processes an entire folder and subfolders'

parser = argparse.ArgumentParser(prog='infer_folder.py',
                                 description=description,
                                 usage='infer_item.py -id <input_directory> -od <output_directory> -m <model_path>',
                                 epilog="The model is called pose_landmarker.task and it can be downloaded with: !wget -O face_landmarker_v2_with_blendshapes.task -q https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task")

parser.add_argument("-id", "--input_directory", required=True, help="Directory with the testing images. Can have subdiorectories", type=str)
parser.add_argument("-od", "--output_directory", required=True, help="Directory with the inference results. It has to be empty. If the given directory does not exists, it will be created. The structure of the input directory (subdirectories, etc) will be recreated", type=str)
parser.add_argument("-m", "--model_file", required=True, help="Gesture model file", type=str)

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



file_list = []
valid_extensions = {".jpg", ".jpeg", ".png"}
for root, _, files in os.walk(input_dir):
    for file in files:
        if os.path.splitext(file)[1].lower() in valid_extensions:
            file_list.append(os.path.join(root, file))

total_files = len(file_list)
print(f"Total files to process: {total_files}")

base_options = python.BaseOptions(model_asset_path=args.model_file)
options = vision.GestureRecognizerOptions(
    base_options=base_options,
    num_hands=2,
    min_hand_detection_confidence=0.2,
    min_hand_presence_confidence=0.2)
recognizer = vision.GestureRecognizer.create_from_options(options)


# landmark_image = False
# blendshapes_chart = False
# emotion_image = False
# rotated_image = False

# if args.no_landmark_image:
#     landmark_image = True

# if args.no_blendshapes_chart:
#     blendshapes_chart = True

# if args.no_emotion_image:
#     emotion_image = True  

# if args.no_rotated_image:
#     rotated_image = True


with tqdm(total=total_files, desc="Processing files", unit="file") as pbar:
    for file_path in file_list:
        
        rel_path = os.path.relpath(file_path, input_dir)
        target_path = os.path.join(output_dir, os.path.dirname(rel_path))
        os.makedirs(target_path, exist_ok=True)
        output_file_path = os.path.join(target_path, os.path.basename(file_path))
        
        # Process file (for now, just copying as a placeholder)
        # shutil.copy(file_path, output_file_path)

        process_image(recognizer, file_path, target_path)
        
        pbar.update(1)

print("Processing complete.")

