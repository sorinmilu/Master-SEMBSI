import dlib
import cv2
import numpy as np
from PIL import Image


# Load face detector and landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Load image
img = dlib.load_rgb_image("face_square.jpg")
faces = detector(img)

# Convert to OpenCV format (BGR)
img_cv2 = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

# Process detected faces
for face in faces:
    landmarks = predictor(img, face)
    points = np.array([(p.x, p.y) for p in landmarks.parts()])

    # Draw landmarks
    for (x, y) in points:
        cv2.circle(img_cv2, (x, y), 2, (0, 255, 0), -1)  # Green dots

# Convert back to PIL and display
img_pil = Image.fromarray(cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB))
img_pil.save("face_with_landmarks.jpg")  # Save the image

print("Image saved as 'face_with_landmarks.jpg'")
