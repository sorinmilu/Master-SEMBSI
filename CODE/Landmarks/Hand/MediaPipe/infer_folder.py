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




def detect_hand_gesture(hand_landmarks):
    """
    Detects gestures for a single hand based on Mediapipe landmarks.
    Supported gestures: Victory, Open Palm, Thumbs Up, Thumbs Down, Closed Fist,
                        Live Long and Prosper, Rock and Roll, OK Sign.

    Args:
        hand_landmarks: List of Mediapipe hand landmarks.

    Returns:
        A string representing the detected gesture.
    """
    # Define landmark indices for fingers
    INDEX_FINGER_TIP = 8
    INDEX_FINGER_PIP = 6
    MIDDLE_FINGER_TIP = 12
    MIDDLE_FINGER_PIP = 10
    MIDDLE_FINGER_MCP = 9
    RING_FINGER_TIP = 16
    RING_FINGER_PIP = 14
    PINKY_TIP = 20
    PINKY_PIP = 18
    THUMB_TIP = 4
    THUMB_IP = 3
    WRIST = 0

    # Get coordinates for key landmarks
    thumb_tip = hand_landmarks[THUMB_TIP]
    thumb_ip = hand_landmarks[THUMB_IP]
    index_tip = hand_landmarks[INDEX_FINGER_TIP]
    index_pip = hand_landmarks[INDEX_FINGER_PIP]
    middle_tip = hand_landmarks[MIDDLE_FINGER_TIP]
    middle_pip = hand_landmarks[MIDDLE_FINGER_PIP]
    middle_mcp = hand_landmarks[MIDDLE_FINGER_MCP]
    ring_tip = hand_landmarks[RING_FINGER_TIP]
    ring_pip = hand_landmarks[RING_FINGER_PIP]
    pinky_tip = hand_landmarks[PINKY_TIP]
    pinky_pip = hand_landmarks[PINKY_PIP]
    wrist = hand_landmarks[WRIST]

    # Gesture: Victory (Index and Middle fingers extended, others folded)
    if (index_tip.y < index_pip.y and  # Index finger extended
        middle_tip.y < middle_pip.y and  # Middle finger extended
        ring_tip.y > ring_pip.y and  # Ring finger folded
        pinky_tip.y > pinky_pip.y and  # Pinky folded
        thumb_tip.y > thumb_ip.y):  # Thumb folded
        return "Victory"

    # Gesture: Open Palm (All fingers extended)
    if (index_tip.y < index_pip.y and
        middle_tip.y < middle_pip.y and
        ring_tip.y < ring_pip.y and
        pinky_tip.y < pinky_pip.y and
        thumb_tip.x < wrist.x and  # Thumb extended
        thumb_tip.y < thumb_ip.y):  # Thumb not folded
        return "Open Palm"

    # Gesture: Thumbs Up (Thumb extended upward, other fingers folded)
    if (thumb_tip.y < thumb_ip.y and  # Thumb pointing up
        index_tip.y > index_pip.y and  # Index finger folded
        middle_tip.y > middle_pip.y and  # Middle finger folded
        ring_tip.y > ring_pip.y and  # Ring finger folded
        pinky_tip.y > pinky_pip.y):  # Pinky folded
        return "Thumbs Up"

    # Gesture: Thumbs Down (Thumb extended downward, other fingers folded)
    if (thumb_tip.y > thumb_ip.y and  # Thumb pointing down
        index_tip.y > index_pip.y and  # Index finger folded
        middle_tip.y > middle_pip.y and  # Middle finger folded
        ring_tip.y > ring_pip.y and  # Ring finger folded
        pinky_tip.y > pinky_pip.y):  # Pinky folded
        return "Thumbs Down"

    # Gesture: Closed Fist (All fingers folded)
    if (index_tip.y > index_pip.y and
        middle_tip.y > middle_pip.y and
        ring_tip.y > ring_pip.y and
        pinky_tip.y > pinky_pip.y and
        thumb_tip.y > thumb_ip.y):
        return "Closed Fist"

    # Gesture: Live Long and Prosper (Vulcan Sign)
    if (index_tip.y < index_pip.y and  # Index finger extended
        middle_tip.y < middle_pip.y and  # Middle finger extended
        ring_tip.y < ring_pip.y and  # Ring finger extended
        pinky_tip.y < pinky_pip.y and  # Pinky extended
        abs(index_tip.x - middle_tip.x) > abs(middle_tip.x - ring_tip.x)):  # Gap between index and middle
        return "Live Long and Prosper"

    # Gesture: Rock and Roll (Index and Pinky extended, others folded)
    if (index_tip.y < index_pip.y and  # Index finger extended
        middle_tip.y > middle_pip.y and  # Middle finger folded
        ring_tip.y > ring_pip.y and  # Ring finger folded
        pinky_tip.y < pinky_pip.y):  # Pinky extended
        return "Rock and Roll"

    # Gesture: OK Sign (Thumb and Index form a circle, others extended)
    if (abs(thumb_tip.x - index_tip.x) < 0.05 and  # Thumb and Index form a circle
        abs(thumb_tip.y - index_tip.y) < 0.05 and
        middle_tip.y < middle_pip.y and  # Middle finger extended
        ring_tip.y < ring_pip.y and  # Ring finger extended
        pinky_tip.y < pinky_pip.y):  # Pinky extended
        return "OK Sign"


    # hand_size = abs(wrist.y - middle_mcp.y)  # Use Y-distance as a reference
    # z_threshold = hand_size * 0.6 
    # #Gesture: Middle Finger (Middle finger extended, others folded)
    # middle_extended = (middle_tip.y < middle_pip.y and  # Extended upward
    #                    middle_pip.y < middle_mcp.y ) 

    # # Ensure all other fingers are folded (Y + Z checks)
    
    # y_threshold = 0.6
    
    # print(f"1 {index_tip.y - index_pip.y}")
    # print(f" 1.5: {abs(index_tip.y - index_pip.y)} - {abs(index_tip.y - index_pip.y) < y_threshold}")
    # print(f"2 {ring_tip.y - ring_pip.y}")    
    # print(f" 2.5: {abs(ring_tip.y > ring_pip.y)}  - {abs(ring_tip.y > ring_pip.y) < y_threshold}")
    # print(f"3 {pinky_tip.y - pinky_pip.y}")
    # print(f" 3.5: {abs(pinky_tip.y > pinky_pip.y)}  - {abs(pinky_tip.y > pinky_pip.y) < y_threshold}")    
    # print(f"4 {thumb_tip.y - thumb_ip.y}")

    

    # others_folded = (
    #     abs(index_tip.y - index_pip.y) < y_threshold and abs(index_tip.z - index_pip.z) < z_threshold and
    #     abs(ring_tip.y - ring_pip.y) < y_threshold and abs(ring_tip.z - ring_pip.z) < z_threshold and
    #     abs(pinky_tip.y - pinky_pip.y) < y_threshold and abs(pinky_tip.z - pinky_pip.z) < z_threshold
    # )

    # print(f"Middle extended: {middle_extended} others_folded {others_folded}")

    # if middle_extended and others_folded:
    #     return "Middle Finger"


    # Default: No gesture detected
    return "Unknown"




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


def process_image(detector, file_path, target_path):
    file_dir = os.path.dirname(file_path)
    file_base, file_ext = os.path.splitext(os.path.basename(file_path))

    outext = ".png"

    mp_image = mp.Image.create_from_file(file_path)
    detection_result = detector.detect(mp_image)
    original_image = mp_image.numpy_view()

    # Annotated image

    annotated_image_path = os.path.join(target_path, f"{file_base}_0_landmarks{file_ext}")
    annotated_image = draw_landmarks_on_image(original_image, detection_result)

    hand_landmarks_list = detection_result.hand_landmarks
    handedness_list = detection_result.handedness

    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]
        handedness = handedness_list[idx]

        # Detect the gesture for the current hand
        gesture = detect_hand_gesture(hand_landmarks)

        # Get the top left corner of the detected hand's bounding box
        height, width, _ = annotated_image.shape
        x_coordinates = [landmark.x for landmark in hand_landmarks]
        y_coordinates = [landmark.y for landmark in hand_landmarks]
        text_x = int(min(x_coordinates) * width)
        text_y = int(min(y_coordinates) * height) - 10

        # Draw handedness and gesture on the image
        text = f"{handedness[0].category_name}: {gesture}"  
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
parser.add_argument("-m", "--model_file", required=True, help="Input file", type=str)


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
options = vision.HandLandmarkerOptions(base_options=base_options,
                                       num_hands=2
                                    )
detector = vision.HandLandmarker.create_from_options(options)

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
        print(file_path)
        rel_path = os.path.relpath(file_path, input_dir)
        target_path = os.path.join(output_dir, os.path.dirname(rel_path))
        os.makedirs(target_path, exist_ok=True)
        output_file_path = os.path.join(target_path, os.path.basename(file_path))
        
        # Process file (for now, just copying as a placeholder)
        # shutil.copy(file_path, output_file_path)

        process_image(detector, file_path, target_path)
        
        pbar.update(1)

print("Processing complete.")

