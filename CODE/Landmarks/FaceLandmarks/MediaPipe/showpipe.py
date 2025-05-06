import cv2
import mediapipe as mp
import numpy as np

# Function to calculate the Euclidean distance
def calculate_distance(point1, point2):
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

# Function to calculate the area of a polygon using landmarks
def calculate_area(landmarks, indices, width, height):
    points = [(landmarks[i][0] * width, landmarks[i][1] * height) for i in indices]
    contour = np.array(points, dtype=np.int32)
    return cv2.contourArea(contour)

# Initialize Mediapipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, refine_landmarks=True)

# Load input image
input_image_path = "../data/face_square.jpg"  # Replace with your input PNG file
image = cv2.imread(input_image_path)
rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Process the image to detect face landmarks
results = face_mesh.process(rgb_image)

# Check if landmarks are detected
if results.multi_face_landmarks:
    height, width, _ = image.shape
    annotated_image = image.copy()
    measurement_image = image.copy()
    
    for face_landmarks in results.multi_face_landmarks:
        # Convert normalized landmarks to pixel coordinates
        landmarks = [(lm.x, lm.y) for lm in face_landmarks.landmark]
        
        # Draw landmarks with red dots and numbers
        for i, (x, y) in enumerate(landmarks):
            x_px, y_px = int(x * width), int(y * height)
            cv2.circle(annotated_image, (x_px, y_px), 2, (0, 0, 255), -1)
            cv2.putText(annotated_image, str(i), (x_px, y_px - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
        
        # Calculate measurements
        interocular_distance = calculate_distance(landmarks[33], landmarks[133])
        nose_length = calculate_distance(landmarks[1], landmarks[152])
        nose_width = calculate_distance(landmarks[49], landmarks[279])
        mouth_width = calculate_distance(landmarks[61], landmarks[291])
        face_width = calculate_distance(landmarks[178], landmarks[454])
        face_height = calculate_distance(landmarks[10], landmarks[152])
        upper_lip_nose_distance = calculate_distance(landmarks[13], landmarks[1])
        
        # Draw measurement lines
        def draw_line(p1, p2, color):
            p1_px = (int(p1[0] * width), int(p1[1] * height))
            p2_px = (int(p2[0] * width), int(p2[1] * height))
            cv2.line(measurement_image, p1_px, p2_px, color, 1)
        
        draw_line(landmarks[33], landmarks[133], (255, 0, 0))  # Interocular distance
        draw_line(landmarks[1], landmarks[152], (0, 255, 0))  # Nose length
        draw_line(landmarks[49], landmarks[279], (0, 0, 255))  # Nose width
        draw_line(landmarks[61], landmarks[291], (255, 255, 0))  # Mouth width
        draw_line(landmarks[178], landmarks[454], (255, 0, 255))  # Face width
        draw_line(landmarks[10], landmarks[152], (0, 255, 255))  # Face height
        
        # Calculate areas
        left_eye_indices = [33, 133, 160, 144, 163, 33]
        right_eye_indices = [362, 263, 249, 466, 467, 362]
        mouth_indices = [61, 185, 40, 39, 37, 0, 61]
        
        left_eye_area = calculate_area(landmarks, left_eye_indices, width, height)
        right_eye_area = calculate_area(landmarks, right_eye_indices, width, height)
        mouth_area = calculate_area(landmarks, mouth_indices, width, height)
        
        # Add text for measurements
        cv2.putText(measurement_image, f"Interocular: {interocular_distance:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(measurement_image, f"Nose Length: {nose_length:.2f}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(measurement_image, f"Nose Width: {nose_width:.2f}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(measurement_image, f"Mouth Width: {mouth_width:.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(measurement_image, f"Face Width: {face_width:.2f}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(measurement_image, f"Face Height: {face_height:.2f}", (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(measurement_image, f"Left Eye Area: {left_eye_area:.2f}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(measurement_image, f"Right Eye Area: {right_eye_area:.2f}", (10, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(measurement_image, f"Mouth Area: {mouth_area:.2f}", (10, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(measurement_image, f"Upper Lip-Nose Distance: {upper_lip_nose_distance:.2f}", (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    # Save output images
    cv2.imwrite("annotated_landmarks.png", annotated_image)
    cv2.imwrite("measurements_overlay.png", measurement_image)
    print("Output saved as 'annotated_landmarks.png' and 'measurements_overlay.png'")
else:
    print("No face landmarks detected!")
