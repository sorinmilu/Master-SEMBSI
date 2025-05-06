import cv2
import mediapipe as mp
import numpy as np
import sys

# Initialize Mediapipe
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def process_image(image_path):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Unable to load image.")
        return

    # Convert the image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Initialize Mediapipe Pose
    with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose:
        # Process the image to extract landmarks
        results = pose.process(image_rgb)

        # Check if landmarks are detected
        if not results.pose_landmarks:
            print("No landmarks detected.")
            return

        # Draw landmarks on the image
        annotated_image = image.copy()
        mp_drawing.draw_landmarks(
            annotated_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Save the annotated image
        output_path = "annotated_image.jpg"
        cv2.imwrite(output_path, annotated_image)
        print(f"Annotated image saved as {output_path}")

        # Print available landmarks and calculate parameters
        landmarks = results.pose_landmarks.landmark
        print("Landmarks detected:")
        for i, landmark in enumerate(landmarks):
            print(f"Landmark {i}: (x={landmark.x}, y={landmark.y}, z={landmark.z}, visibility={landmark.visibility})")

        # Example: Calculate distances between landmarks
        def calculate_distance(lm1, lm2):
            return np.sqrt((lm1.x - lm2.x)**2 + (lm1.y - lm2.y)**2 + (lm1.z - lm2.z)**2)

        print("\nExample Parameters:")
        if len(landmarks) > 11:  # Ensure there are enough landmarks
            left_shoulder = landmarks[11]
            right_shoulder = landmarks[12]
            distance = calculate_distance(left_shoulder, right_shoulder)
            print(f"Distance between shoulders: {distance:.4f}")

            left_hip = landmarks[23]
            right_hip = landmarks[24]
            hip_distance = calculate_distance(left_hip, right_hip)
            print(f"Distance between hips: {hip_distance:.4f}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py <image_path>")
    else:
        process_image(sys.argv[1])