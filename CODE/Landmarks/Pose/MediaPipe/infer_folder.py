import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import argparse
import os
from tqdm import tqdm
from mediapipe.python.solutions.drawing_utils import DrawingSpec



def detect_activity(pose_landmarks):
    """
    Detect if the person is sitting, sitting with hands raised, or standing based on hip, knee, wrist, shoulder, and foot positions.

    Args:
        pose_landmarks: List of pose landmarks detected by Mediapipe.

    Returns:
        A string indicating the activity: "Sitting", "Sitting Hands", or "Standing".
    """
    # Get the landmarks for hips, knees, shoulders, wrists, and feet
    left_hip = pose_landmarks[23]  # Left hip landmark
    right_hip = pose_landmarks[24]  # Right hip landmark
    left_knee = pose_landmarks[25]  # Left knee landmark
    right_knee = pose_landmarks[26]  # Right knee landmark
    left_shoulder = pose_landmarks[11]  # Left shoulder landmark
    right_shoulder = pose_landmarks[12]  # Right shoulder landmark
    left_wrist = pose_landmarks[15]  # Left wrist landmark
    right_wrist = pose_landmarks[16]  # Right wrist landmark
    left_foot = pose_landmarks[31]  # Left foot landmark
    right_foot = pose_landmarks[32]  # Right foot landmark

    # Calculate the average y-coordinates for key landmarks
    avg_hip_y = (left_hip.y + right_hip.y) / 2
    avg_knee_y = (left_knee.y + right_knee.y) / 2
    avg_shoulder_y = (left_shoulder.y + right_shoulder.y) / 2
    avg_wrist_y = (left_wrist.y + right_wrist.y) / 2
    avg_foot_y = (left_foot.y + right_foot.y) / 2

    # Check if the person is sitting (hips are lower than knees and shoulders are above hips)
    if avg_foot_y < avg_shoulder_y :
        return "Upside Down"

    # Check if the person is sitting (hips are lower than knees and shoulders are above hips)
    if avg_hip_y > avg_knee_y and avg_shoulder_y < avg_hip_y:
        # Check if both wrists are above the shoulders
        if left_wrist.y < avg_shoulder_y and right_wrist.y < avg_shoulder_y:
            return "Sitting Hands"  # Sitting with hands raised
        else:
            return "Sitting"  # Sitting without hands raised

    # Default to standing if hips are not lower than knees
    return "Standing"

def detect_hands_position(pose_landmarks):
    """
    Detect if hands are raised or lowered based on wrist and shoulder positions.

    Args:
        pose_landmarks: List of pose landmarks detected by Mediapipe.

    Returns:
        A string indicating the position of the hands: "Hands Raised", "Hands Lowered", or "Unknown".
    """
    # Get the landmarks for shoulders and wrists
    left_shoulder = pose_landmarks[11]  # Left shoulder landmark
    right_shoulder = pose_landmarks[12]  # Right shoulder landmark
    left_wrist = pose_landmarks[15]  # Left wrist landmark
    right_wrist = pose_landmarks[16]  # Right wrist landmark

    # Calculate the average y-coordinate of the shoulders
    avg_shoulder_y = (left_shoulder.y + right_shoulder.y) / 2

    # Check if both wrists are above the shoulders
    if left_wrist.y < avg_shoulder_y and right_wrist.y < avg_shoulder_y:
        return "Hands Raised"
    # Check if both wrists are below the shoulders
    elif left_wrist.y > avg_shoulder_y and right_wrist.y > avg_shoulder_y:
        return "Hands Lowered"
    else:
        return "Unknown"

def detect_lying_down(pose_landmarks):
    """
    Detect if a person is lying down based on the relative vertical positions of key landmarks.

    Args:
        pose_landmarks: List of pose landmarks detected by Mediapipe.

    Returns:
        A string indicating whether the person is "Lying Down" or "Not Lying Down".
    """
    # Get the landmarks for shoulders, hips, and feet
    left_shoulder = pose_landmarks[11]  # Left shoulder landmark
    right_shoulder = pose_landmarks[12]  # Right shoulder landmark
    left_hip = pose_landmarks[23]  # Left hip landmark
    right_hip = pose_landmarks[24]  # Right hip landmark
    left_foot = pose_landmarks[31]  # Left foot landmark
    right_foot = pose_landmarks[32]  # Right foot landmark

    # Calculate the average y-coordinates for shoulders, hips, and feet
    avg_shoulder_y = (left_shoulder.y + right_shoulder.y) / 2
    avg_hip_y = (left_hip.y + right_hip.y) / 2
    avg_foot_y = (left_foot.y + right_foot.y) / 2

    # Check if the shoulders, hips, and feet are roughly aligned horizontally
    if abs(avg_shoulder_y - avg_hip_y) < 0.1 and abs(avg_hip_y - avg_foot_y) < 0.1:
        return "Lying Down"
    else:
        return "Not Lying Down"


def draw_landmarks_on_image(rgb_image, pose_landmarks_list):
    annotated_image = np.copy(rgb_image)

    image_width = rgb_image.shape[1]

    # Dynamically calculate thickness and circle radius based on image width
    thickness = max(1, image_width // 500)  # Adjust the divisor to control scaling
    circle_radius = max(1, image_width // 200)  # Adjust the divisor to control scaling

    # Define custom styles for landmarks and connections
    landmark_style = DrawingSpec(color=(0, 255, 0), thickness=thickness, circle_radius=circle_radius)
    connection_style = DrawingSpec(color=(255, 0, 0), thickness=thickness)


    # Loop through the detected poses to visualize.
    for idx in range(len(pose_landmarks_list)):
        pose_landmarks = pose_landmarks_list[idx]

        sitting = detect_activity(pose_landmarks)
        hands_position = detect_hands_position(pose_landmarks)


    # Draw the pose landmarks.
        pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        pose_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
        ])
    
        solutions.drawing_utils.draw_landmarks(
        annotated_image,
        pose_landmarks_proto,
        solutions.pose.POSE_CONNECTIONS,
        landmark_drawing_spec=landmark_style,
        connection_drawing_spec=connection_style)
    return annotated_image



def detect_pose(rgb_image, pose_landmarks_list,segmentation_masks):
    # Create a blank RGBA image for overlaying text

    overlay_no_alpha = np.zeros_like(rgb_image, dtype=np.uint8)
    alpha_channel = np.full((overlay_no_alpha.shape[0], overlay_no_alpha.shape[1], 1), 255, dtype=np.uint8)
    overlay = np.concatenate((overlay_no_alpha, alpha_channel), axis=-1)
    pose_image = np.array(rgb_image, copy=True)

    image_width = rgb_image.shape[1]

    # Dynamically calculate font scale based on image width
    font_scale = max(0.5, image_width / 1500)  # Adjust the divisor to control scaling
    font_thickness = max(1, int(image_width / 500))  # Dynamically adjust thickness
    line_spacing = int(font_scale * 30)

    for idx in range(len(pose_landmarks_list)):
        pose_landmarks = pose_landmarks_list[idx]

        # Detect activity, hands position, and lying status for each person
        sitting = detect_activity(pose_landmarks)
        hands_position = detect_hands_position(pose_landmarks)
        lying_down = detect_lying_down(pose_landmarks)

        # Get the segmentation mask for the current person
        segmentation_mask = segmentation_masks[idx].numpy_view()

        # Find the bounding box of the segmentation mask
        mask_indices = np.where(segmentation_mask > 0.5)  # Threshold the mask
        if mask_indices[0].size == 0 or mask_indices[1].size == 0:
            print(f"No valid mask region found for person {idx + 1}.")
            continue

        # Calculate the bounding box of the mask
        min_y, max_y = np.min(mask_indices[0]), np.max(mask_indices[0])
        min_x, max_x = np.min(mask_indices[1]), np.max(mask_indices[1])

        # Define the position for the text (centered within the mask region)
        text_x = min_x + (max_x - min_x) // 2
        text_y = min_y + (max_y - min_y) // 2

        # Prepare the text to overlay
        text = f"Sitting: {sitting}\nHands: {hands_position}\nLying: {lying_down}"

        # Use OpenCV to add text directly to the original image
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_color = (255, 255, 0) 

        # Split the text into lines and overlay each line
        for i, line in enumerate(text.split("\n")):
            line_y = text_y + i * line_spacing
            cv2.putText(pose_image, line, (text_x, line_y), font, font_scale, text_color, font_thickness, cv2.LINE_AA)

    return pose_image    

def process_background(rgb_image, segmentation_masks):
    if segmentation_masks is None or len(segmentation_masks) == 0:
        print("No segmentation masks found. Skipping background processing.")
        return None, None  # Return empty results if no masks are available

    colors = [
        (255, 0, 0),   # Red for person 1
        (0, 255, 0),   # Green for person 2
        (0, 0, 255),   # Blue for person 3
        (255, 255, 0), # Yellow for person 4
        (255, 0, 255), # Magenta for person 5
        (0, 255, 255)  # Cyan for person 6
    ]

    combined_mask = np.zeros((*segmentation_masks[0].numpy_view().shape, 3), dtype=np.uint8)
    combined_background = np.zeros((*segmentation_masks[0].numpy_view().shape, 4), dtype=np.uint8)

    for idx, segmentation_mask in enumerate(segmentation_masks):
        # Ensure we don't exceed the number of predefined colors
        color = colors[idx % len(colors)]

        # Scale the mask to 255 and convert to uint8
        mask = (segmentation_mask.numpy_view() * 255).astype(np.uint8)

        # Create a colored mask for visualization
        colored_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
        for i in range(3):  # Apply the color to each channel
            colored_mask[:, :, i] = mask * (color[i] / 255)

        # Add the colored mask to the combined mask
        combined_mask = cv2.addWeighted(combined_mask, 1.0, colored_mask, 0.5, 0)

        # Extract the background for this person
        alpha_channel = mask  # Use the mask as the alpha channel
        person_background = np.zeros_like(combined_background, dtype=np.uint8)

        # Preserve the original RGB values where the mask is non-zero
        person_background[:, :, :3] = np.where(mask[:, :, np.newaxis] > 0, rgb_image, 0)
        person_background[:, :, 3] = alpha_channel  # Set the alpha channel

        # Blend the extracted background for this person into the combined background
        combined_background = cv2.addWeighted(combined_background, 1.0, person_background, 1.0, 0)

    return combined_mask, combined_background


def process_image(detector, file_path, target_path, landmark_image, pose_image, mask_images):
    # STEP 3: Load the input image.

    file_dir = os.path.dirname(file_path)
    file_base, file_ext = os.path.splitext(os.path.basename(file_path))

    outext = ".png"
    # Generate alternative file names with suffixes

    mp_image = mp.Image.create_from_file(file_path)

    # run the mode and extract the data
    detection_result = detector.detect(mp_image)
    pose_landmarks_list = detection_result.pose_landmarks
    segmentation_masks = detection_result.segmentation_masks


    # Convert the original image to a NumPy array
    original_image = mp_image.numpy_view()

    # Annotated image
    if landmark_image:
        annotated_image_path = os.path.join(target_path, f"{file_base}_0_landmarks{file_ext}")
        annotated_image = draw_landmarks_on_image(original_image, pose_landmarks_list)
        annotated_image_pil = Image.fromarray(annotated_image)
        annotated_image_pil.save(annotated_image_path)

    # Pose detection phase
    if pose_image:
        final_pose_image_path = os.path.join(target_path, f"{file_base}_1_pose{outext}")
        pose_detection_image = detect_pose(original_image, pose_landmarks_list, segmentation_masks)
        pose_detection_pil = Image.fromarray(pose_detection_image.astype(np.uint8))
        pose_detection_pil.save(final_pose_image_path)

    #Background extraction and mask creation
    
    if mask_images:
        segmentation_image_path = os.path.join(target_path, f"{file_base}_2_mask{outext}")
        combined_background_path = os.path.join(target_path, f"{file_base}_3_extracted{outext}")
        combined_mask, combined_background = process_background(original_image, segmentation_masks)
        if combined_mask is not None and combined_background is not None:
            combined_mask_pil = Image.fromarray(combined_mask.astype(np.uint8))
            combined_mask_pil.save(segmentation_image_path)

            # Save the extracted background with transparency
            combined_background_pil = Image.fromarray(combined_background.astype(np.uint8), mode="RGBA")
            combined_background_pil.save(combined_background_path)
    





description = 'Inference program for pose detection using Mediapipe library that processes an entire folder and subfolders'

parser = argparse.ArgumentParser(prog='infer_folder.py',
                                 description=description,
                                 usage='infer_item.py -id <input_directory> -od <output_directory> -m <model_path>',
                                 epilog="The model is called pose_landmarker.task and it can be downloaded with: wget -O pose_landmarker.task -q https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task")

parser.add_argument("-id", "--input_directory", required=True, help="Directory with the testing images. Can have subdiorectories", type=str)
parser.add_argument("-od", "--output_directory", required=True, help="Directory with the inference results. It has to be empty. If the given directory does not exists, it will be created. The structure of the input directory (subdirectories, etc) will be recreated", type=str)
parser.add_argument("-m", "--model_file", required=True, help="Input file", type=str)
parser.add_argument("-no_lm", "--no_landmark_image", help="Do not generate landmarks image", action="store_false")
parser.add_argument("-no_ps", "--no_pose_image", help="Do not generate pose image", action="store_false")
parser.add_argument("-no_mk", "--no_mask_images", help="Do not generate mask and background extracted image", action="store_false")

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
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    output_segmentation_masks=True,
    num_poses=10)
detector = vision.PoseLandmarker.create_from_options(options)

landmark_image = False
pose_image = False
mask_images = False

if args.no_landmark_image:
    landmark_image = True

if args.no_pose_image:
    pose_image = True

if args.no_mask_images:
    mask_images = True  

with tqdm(total=total_files, desc="Processing files", unit="file") as pbar:
    for file_path in file_list:
     
        rel_path = os.path.relpath(file_path, input_dir)
        target_path = os.path.join(output_dir, os.path.dirname(rel_path))
        os.makedirs(target_path, exist_ok=True)
        output_file_path = os.path.join(target_path, os.path.basename(file_path))
        
        # Process file (for now, just copying as a placeholder)
        # shutil.copy(file_path, output_file_path)

        process_image(detector, file_path, target_path, landmark_image, pose_image, mask_images)
        
        pbar.update(1)

print("Processing complete.")