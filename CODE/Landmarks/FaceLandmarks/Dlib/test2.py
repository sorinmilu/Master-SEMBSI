import dlib
import cv2

# Load face detector and predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Load image
img = cv2.imread("face_square.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect faces
faces = detector(gray)

# Loop through detected faces
for face in faces:
    landmarks = predictor(gray, face)
    
    # Extract landmark points for eyes, nose, and mouth
    # Left eye: 36 to 41
    for i in range(36, 42):
        x, y = landmarks.part(i).x, landmarks.part(i).y
        cv2.circle(img, (x, y), 2, (0, 255, 0), -1)  # Green dots
    
    # Right eye: 42 to 47
    for i in range(42, 48):
        x, y = landmarks.part(i).x, landmarks.part(i).y
        cv2.circle(img, (x, y), 2, (0, 255, 0), -1)  # Green dots
    
    # Nose: 27 to 35
    for i in range(27, 36):
        x, y = landmarks.part(i).x, landmarks.part(i).y
        cv2.circle(img, (x, y), 2, (0, 255, 0), -1)  # Green dots
    
    # Mouth: 48 to 67
    for i in range(48, 68):
        x, y = landmarks.part(i).x, landmarks.part(i).y
        cv2.circle(img, (x, y), 2, (0, 255, 0), -1)  # Green dots

# Show image with facial features
cv2.imwrite("face_with_landmarks2.jpg", img)  # This will save the image to disk
