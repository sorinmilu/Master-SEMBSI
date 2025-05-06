import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import os
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from PIL import Image
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from tqdm import tqdm
import argparse



def detect_big_nose(landmarks, threshold_nose_width=0.28, threshold_nose_height=0.4):
    # Calculate nose width and height

    # face_width = np.linalg.norm([landmarks[33].x - landmarks[362].x, landmarks[33].y - landmarks[362].y])
    face_width = np.linalg.norm([landmarks[234].x - landmarks[454].x, landmarks[234].y - landmarks[454].y])


    face_height = np.linalg.norm([landmarks[152].x - landmarks[10].x, landmarks[152].y - landmarks[10].y])

    nose_width = np.linalg.norm([
        landmarks[48].x - landmarks[278].x,
        landmarks[48].y - landmarks[278].y
    ])
    nose_height = np.linalg.norm([landmarks[1].x - landmarks[8].x, landmarks[1].y - landmarks[8].y])

    # Proportional analysis
    nose_width_ratio = nose_width / face_width
    nose_height_ratio = nose_height / face_height

    is_big_nose = nose_width_ratio > threshold_nose_width or nose_height_ratio > threshold_nose_height

    print(f"Nose Width Ratio: {nose_width_ratio}, Nose Height Ratio: {nose_height_ratio}")

    if is_big_nose:
        return f"Big: W: {nose_width_ratio:.3f}, Height: {nose_height_ratio:.3f}"
    else:
        return f"Normal: W: {nose_width_ratio:.3f}, Height: {nose_height_ratio:.3f}"

def detect_big_mouth(landmarks, threshold_mouth_width=0.4):
    """
    Detects if the mouth is unusually big based on proportions.
    """
    face_width = np.linalg.norm([landmarks[234].x - landmarks[454].x, landmarks[234].y - landmarks[454].y])


    # Calculate mouth width (distance between the corners of the mouth)
    mouth_width = np.linalg.norm([landmarks[61].x - landmarks[291].x, landmarks[61].y - landmarks[291].y])

    # Proportional analysis (mouth width relative to face width)
    mouth_width_ratio = mouth_width / face_width

    print(f"Mouth Width Ratio: {mouth_width_ratio}")

    is_big_mouth = mouth_width_ratio > threshold_mouth_width
    if is_big_mouth:
        return f"Big Mouth W: {mouth_width_ratio:.4f}"
    else:
        return f"Normal Mouth W: {mouth_width_ratio:.4f}"



def detect_eye_separation(landmarks, threshold_eye_separation=0.55):
    """
    Detects if the eyes are too far apart based on the distance between the eyes.
    """
    face_width = np.linalg.norm([landmarks[234].x - landmarks[454].x, landmarks[234].y - landmarks[454].y])


    # Calculate distance between outer eyes
    left_eye_outer = np.array([landmarks[33].x, landmarks[33].y])
    right_eye_outer = np.array([landmarks[362].x, landmarks[362].y])
    eye_distance = np.linalg.norm(left_eye_outer - right_eye_outer)

    # Proportional analysis
    eye_distance_ratio = eye_distance / face_width

    is_strangely_apart = eye_distance_ratio > threshold_eye_separation
    return is_strangely_apart

def is_mouth_open(landmarks, threshold=0.02):
    # Extract upper and lower lip landmarks
    upper_lip = np.array([landmarks[13].y])
    lower_lip = np.array([landmarks[14].y])

    # Compute the mouth opening distance
    mouth_opening = np.abs(upper_lip - lower_lip)

    # Normalize by face height (distance between nose and chin)
    nose = np.array([landmarks[1].y])  # Nose tip
    chin = np.array([landmarks[152].y])  # Chin
    face_height = np.abs(nose - chin)

    # Compute openness ratio
    openness_ratio = mouth_opening / face_height

    if openness_ratio > threshold:
        return("Mouth is Open")
    else:
        return("Mouth is Closed")


def detect_smile(blendshapes):
    smile_left = next(shape.score for shape in blendshapes if shape.category_name == "mouthSmileLeft")
    smile_right = next(shape.score for shape in blendshapes if shape.category_name == "mouthSmileRight")

    if (smile_left + smile_right) / 2 > 0.5:
        return "Person is Smiling"
    else:
        return "Neutral or Frowning"

def detect_gaze(landmarks):
    """
    Detect where the person is looking based on iris position.
    """
    # Get eye landmarks
    left_eye_outer = np.array([landmarks[33].x, landmarks[33].y])  # Outer left corner
    left_eye_inner = np.array([landmarks[133].x, landmarks[133].y])  # Inner left corner
    left_iris = np.array([landmarks[468].x, landmarks[468].y])  # Left iris center

    right_eye_outer = np.array([landmarks[362].x, landmarks[362].y])  # Outer right corner
    right_eye_inner = np.array([landmarks[263].x, landmarks[263].y])  # Inner right corner
    right_iris = np.array([landmarks[473].x, landmarks[473].y])  # Right iris center

    # Compute eye width
    left_eye_width = np.linalg.norm(left_eye_outer - left_eye_inner)
    right_eye_width = np.linalg.norm(right_eye_outer - right_eye_inner)

    # Compute iris position ratio within the eye
    left_ratio = np.linalg.norm(left_iris - left_eye_inner) / left_eye_width
    right_ratio = np.linalg.norm(right_iris - right_eye_inner) / right_eye_width

    # Thresholds to determine gaze direction
    threshold_left = 0.35
    threshold_right = 0.65

    if left_ratio < threshold_left and right_ratio < threshold_left:
        return "Looking Left"
    elif left_ratio > threshold_right and right_ratio > threshold_right:
        return "Looking Right"
    else:
        return "Looking Forward"


def detect_surprise(blendshapes, threshold=0.5):
    """
    Detects if a person looks surprised based on MediaPipe blendshapes.
    
    Parameters:
        blendshapes (list): The list of detected blendshapes from MediaPipe.
        threshold (float): The minimum score to consider an expression as surprise.
        
    Returns:
        bool: True if surprise is detected, False otherwise.
    """
    # Extract relevant blendshapes
    surprise_factors = {
        "browInnerUp": 0,
        "browOuterUpLeft": 0,
        "browOuterUpRight": 0,
        "eyeWideLeft": 0,
        "eyeWideRight": 0,
        "jawOpen": 0
    }
    
    # Fill the dictionary with actual blendshape scores
    for shape in blendshapes:
        if shape.category_name in surprise_factors:
            surprise_factors[shape.category_name] = shape.score

    # Compute average intensity of surprise-related features
    surprise_intensity = (
        surprise_factors["browInnerUp"] +
        surprise_factors["browOuterUpLeft"] +
        surprise_factors["browOuterUpRight"] +
        surprise_factors["eyeWideLeft"] +
        surprise_factors["eyeWideRight"] +
        surprise_factors["jawOpen"]
    ) / len(surprise_factors)

    # Return True if intensity exceeds the threshold
    if surprise_intensity > threshold:
        return "Surprise detected!"
    else:
        return "No surprise detected."

def detect_smile(blendshapes, threshold=0.5):
    """
    Detects if a person is smiling based on MediaPipe blendshapes.
    
    Parameters:
        blendshapes (list): The list of detected blendshapes from MediaPipe.
        threshold (float): The minimum score to consider an expression as a smile.
        
    Returns:
        bool: True if a smile is detected, False otherwise.
    """
    # Extract smile blendshapes
    smile_left = 0
    smile_right = 0

    for shape in blendshapes:
        if shape.category_name == "mouthSmileLeft":
            smile_left = shape.score
        elif shape.category_name == "mouthSmileRight":
            smile_right = shape.score

    # Compute average smile intensity
    smile_intensity = (smile_left + smile_right) / 2

    # Return True if smiling, otherwise False
    if smile_intensity > threshold:
        return "Smiling detected!"
    else:
        "No smile detected."


def detect_emotion(blendshapes):
    # Convert list to dictionary for easy access
    blendshape_dict = {item.category_name: item.score for item in blendshapes}

    # Define thresholds
    smile = (blendshape_dict.get("mouthSmileLeft", 0) + blendshape_dict.get("mouthSmileRight", 0)) / 2
    frown = (blendshape_dict.get("mouthFrownLeft", 0) + blendshape_dict.get("mouthFrownRight", 0)) / 2
    brow_down = (blendshape_dict.get("browDownLeft", 0) + blendshape_dict.get("browDownRight", 0)) / 2
    brow_up = blendshape_dict.get("browInnerUp", 0)
    jaw_open = blendshape_dict.get("jawOpen", 0)
    eyes_wide = (blendshape_dict.get("eyeWideLeft", 0) + blendshape_dict.get("eyeWideRight", 0)) / 2

    emotion = "Neutral"
    # Emotion detection logic
    if smile > 0.5:
        return  "Happy"
    elif frown > 0.3 and brow_up < 0.2:
        return  "Sad"
    elif brow_down > 0.5 and jaw_open < 0.2:
        return  "Angry"
    elif jaw_open > 0.6 and eyes_wide > 0.5:
        return  "Surprised"
   


def rotate_image(image_np, facial_matrix):
    rotation_matrix = facial_matrix[:3, :3]

    # If it's a pure rotation, we can just transpose it to get the inverse rotation
    inverse_rotation = rotation_matrix
    # Convert to 2D affine transformation by removing the Z-axis (third row and column)
    affine_matrix = inverse_rotation[:2, :2]  # Take only the first two rows/cols for 2D rotation


    # Get image center for rotation
    h, w = image_np.shape[:2]
    center = (w // 2, h // 2)

    # Compute the warp affine transformation matrix
    warp_matrix = np.hstack([affine_matrix, np.array([[0], [0]])])  # Add translation (set to 0 initially)

    # Rotate the image
    rotated_image = cv2.warpAffine(image_np, warp_matrix, (w, h))
    return rotated_image


def draw_landmarks_on_image(rgb_image, detection_result):
    image_width = rgb_image.shape[1]

    # Dynamically calculate font scale based on image width
    font_scale = max(0.3, image_width / 6000)
    print (font_scale)
    # font_scale = 0.3
    face_landmarks_list = detection_result.face_landmarks
    annotated_image = np.copy(rgb_image)

    # Loop through the detected faces to visualize.
    for idx in range(len(face_landmarks_list)):
        face_landmarks = face_landmarks_list[idx]

        # Draw the face landmarks.
        face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        face_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in face_landmarks
        ])

        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles
            .get_default_face_mesh_tesselation_style())
        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles
            .get_default_face_mesh_contours_style())
        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_IRISES,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp.solutions.drawing_styles
                .get_default_face_mesh_iris_connections_style())
        
        for i, landmark in enumerate(face_landmarks):
            x = int(landmark.x * rgb_image.shape[1])
            y = int(landmark.y * rgb_image.shape[0])
            cv2.putText(
                annotated_image,
                str(i),  # Landmark index
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,  
                (12, 12, 12),  
                1,  
                cv2.LINE_AA
            )

    return annotated_image

def plot_face_blendshapes_bar_graph(face_blendshapes, output_path=None):
    """
    Plots a bar graph of face blendshapes with scores greater than 0.5 and optionally saves the plot.

    Args:
        face_blendshapes: List of blendshapes with category names and scores.
        output_path: Path to save the plot image. If None, the plot is not saved.

    Returns:
        A PIL image of the plot for further processing or saving outside the function.
    """
    # Filter blendshapes to include only those with a score > 0.5
    face_blendshapes = face_blendshapes[0]
    filtered_blendshapes = [
        blendshape for blendshape in face_blendshapes if blendshape.score > 0.3
    ]

    # Extract the filtered blendshapes' category names and scores
    face_blendshapes_names = [blendshape.category_name for blendshape in filtered_blendshapes]
    face_blendshapes_scores = [blendshape.score for blendshape in filtered_blendshapes]

    # Check if there are any blendshapes to plot
    if not face_blendshapes_names:
        # print("No blendshapes with a score greater than 0.5.")
        return None

    # The blendshapes are ordered in decreasing score value
    face_blendshapes_ranks = range(len(face_blendshapes_names))

    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 12))
    bar = ax.barh(face_blendshapes_ranks, face_blendshapes_scores, label=[str(x) for x in face_blendshapes_ranks])
    ax.set_yticks(face_blendshapes_ranks, face_blendshapes_names)
    ax.invert_yaxis()

    # Label each bar with values
    for score, patch in zip(face_blendshapes_scores, bar.patches):
        plt.text(patch.get_x() + patch.get_width(), patch.get_y(), f"{score:.4f}", va="top")

    ax.set_xlabel('Score')
    ax.set_title("Face Blendshapes (Score > 0.5)")
    plt.tight_layout()

    # Save the plot to a file if an output path is provided
    if output_path:
        plt.savefig(output_path, format="png")
        # print(f"Plot saved to {output_path}")

    # Convert the plot to a PIL image
    from io import BytesIO
    buf = BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    plot_image = Image.open(buf)

    # Close the plot to free memory
    plt.close(fig)

    return plot_image

def draw_emotions_image(image,blendshapes,landmarks):
    image_width = image.shape[1]
    emotions_image = np.array(image, copy=True)

    # Dynamically calculate font scale based on image width
    font_scale = max(0.5, image_width / 1500)  # Adjust the divisor to control scaling
    thickness = max(1, int(image_width / 500))  # Dynamically adjust thickness
    (position_x, position_y) = (50,50)
    color=(0, 255, 0)
    line_spacing = int(font_scale * 30)    

    smile = detect_smile(blendshapes)
    gaze = detect_gaze(landmarks)
    mouth_open = is_mouth_open(landmarks)
    surprise = detect_surprise(blendshapes)
    emotion = detect_emotion(blendshapes)
    big_nose = detect_big_nose(landmarks)
    big_mouth = detect_big_mouth(landmarks)
    eye_separation = detect_eye_separation(landmarks)

    text = f"Smile: {smile}\nGaze: {gaze}\nMouth open: {mouth_open}\nSurprise: {surprise}\nEmotion: {emotion}\nBig nose: {big_nose}\nBig mouth: {big_mouth}\nEye separation: {eye_separation}"

    # Use OpenCV to add text directly to the original image
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_color = (235, 215, 107) 
    shadow_color = (0, 0, 0)
    # Split the text into lines and overlay each line
    for i, line in enumerate(text.split("\n")):
        y = position_y + i * line_spacing
        cv2.putText(emotions_image, line, (position_x + 2, y + 2), font, font_scale, shadow_color, thickness, cv2.LINE_AA)

        cv2.putText(emotions_image, line, (position_x, y), font, font_scale, text_color, thickness, cv2.LINE_AA)

    return emotions_image

def process_image(detector, file_path, target_path, landmark_image, blendshapes_chart, emotion_images, rotated_image):
    file_dir = os.path.dirname(file_path)
    file_base, file_ext = os.path.splitext(os.path.basename(file_path))

    outext = ".png"

    mp_image = mp.Image.create_from_file(file_path)
    detection_result = detector.detect(mp_image)
    original_image = mp_image.numpy_view()

    # Annotated image
    if landmark_image:
        annotated_image_path = os.path.join(target_path, f"{file_base}_0_landmarks{file_ext}")
        annotated_image = draw_landmarks_on_image(original_image, detection_result)
        annotated_image_pil = Image.fromarray(annotated_image)
        annotated_image_pil.save(annotated_image_path)


    if blendshapes_chart:
        plot_image = plot_face_blendshapes_bar_graph(detection_result.face_blendshapes, os.path.join(target_path, f"{file_base}_1_blendshapes{outext}"))    
        if plot_image:
            plot_image.save(os.path.join(target_path, f"{file_base}_1_blendshapes{outext}"))

    if emotion_images:
        emotions_image = draw_emotions_image(original_image, detection_result.face_blendshapes[0], detection_result.face_landmarks[0])
        emotion_image_pil = Image.fromarray(emotions_image)
        emotion_image_pil.save(os.path.join(target_path, f"{file_base}_2_emotions{outext}"))

    if rotated_image:
        annotated_image_path = os.path.join(target_path, f"{file_base}_0_rotated{file_ext}")
        rotated_image = rotate_image(original_image, detection_result.facial_transformation_matrixes[0])
        rotated_image_pil = Image.fromarray(rotated_image)
        rotated_image_pil.save(annotated_image_path)




description = 'Inference program for face landmarks and emotion using Mediapipe library that processes an entire folder and subfolders'

parser = argparse.ArgumentParser(prog='infer_folder.py',
                                 description=description,
                                 usage='infer_item.py -id <input_directory> -od <output_directory> -m <model_path>',
                                 epilog="The model is called pose_landmarker.task and it can be downloaded with: !wget -O face_landmarker_v2_with_blendshapes.task -q https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task")

parser.add_argument("-id", "--input_directory", required=True, help="Directory with the testing images. Can have subdiorectories", type=str)
parser.add_argument("-od", "--output_directory", required=True, help="Directory with the inference results. It has to be empty. If the given directory does not exists, it will be created. The structure of the input directory (subdirectories, etc) will be recreated", type=str)
parser.add_argument("-m", "--model_file", required=True, help="Input file", type=str)
parser.add_argument("-no_lm", "--no_landmark_image", help="Do not generate landmarks image", action="store_false")
parser.add_argument("-no_bs", "--no_blendshapes_chart", help="Do not generate chart with relevant blendshapes", action="store_false")
parser.add_argument("-no_em", "--no_emotion_image", help="Do not generate image with emotions and metrics", action="store_false")
parser.add_argument("-no_rot", "--no_rotated_image", help="Do not generate rotated image", action="store_false")



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
options = vision.FaceLandmarkerOptions(base_options=base_options,
                                       output_face_blendshapes=True,
                                       output_facial_transformation_matrixes=True,
                                       num_faces=1)
detector = vision.FaceLandmarker.create_from_options(options)

landmark_image = False
blendshapes_chart = False
emotion_image = False
rotated_image = False

if args.no_landmark_image:
    landmark_image = True

if args.no_blendshapes_chart:
    blendshapes_chart = True

if args.no_emotion_image:
    emotion_image = True  

if args.no_rotated_image:
    rotated_image = True


with tqdm(total=total_files, desc="Processing files", unit="file") as pbar:
    for file_path in file_list:
     
        rel_path = os.path.relpath(file_path, input_dir)
        target_path = os.path.join(output_dir, os.path.dirname(rel_path))
        os.makedirs(target_path, exist_ok=True)
        output_file_path = os.path.join(target_path, os.path.basename(file_path))
        
        # Process file (for now, just copying as a placeholder)
        # shutil.copy(file_path, output_file_path)

        process_image(detector, file_path, target_path, landmark_image, blendshapes_chart, emotion_image, rotated_image)
        
        pbar.update(1)

print("Processing complete.")

