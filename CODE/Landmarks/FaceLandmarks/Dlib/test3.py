import dlib
import cv2
import numpy as np

# Load face detector and predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Load image
img = cv2.imread("faces.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect faces
faces = detector(gray)

# Loop through detected faces
for face in faces:
    landmarks = predictor(gray, face)

    # Get landmarks as tuples
    points = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(68)]

    # Draw lines connecting the points for the eyes, nose, and mouth
    # Draw lines for left eye (36-41)
    cv2.polylines(img, [np.array(points[36:42], dtype=np.int32)], isClosed=True, color=(0, 255, 0), thickness=2)

    # Draw lines for right eye (42-47)
    cv2.polylines(img, [np.array(points[42:48], dtype=np.int32)], isClosed=True, color=(0, 255, 0), thickness=2)

    # Draw lines for the nose contour (27-35)
    cv2.polylines(img, [np.array(points[27:36], dtype=np.int32)], isClosed=True, color=(0, 255, 0), thickness=2)

    # Draw lines for mouth contour (48-67)
    cv2.polylines(img, [np.array(points[48:68], dtype=np.int32)], isClosed=True, color=(0, 255, 0), thickness=2)

# Save the image with contours drawn
cv2.imwrite("face_with_landmarks_and_contours.jpg", img)

