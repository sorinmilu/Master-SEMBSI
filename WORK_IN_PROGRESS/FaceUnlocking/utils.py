import cv2
import mediapipe as mp
import json
import numpy as np

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
mp_face_detection = mp.solutions.face_detection
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.2, min_tracking_confidence=0.5)
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.2)

# Storage file
DATA_FILE = "data/faces.json"

def extract_embedding(image):
    """ Extract 468 facial landmark points as a unique identifier """
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)

    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0]
        embedding = np.array([[p.x, p.y, p.z] for p in face_landmarks.landmark]).flatten()
        return embedding.tolist()
    return None

def draw_face_landmarks(frame):
    """ Detect and draw face landmarks (points and lines) on the frame """
    # Convert the frame to RGB for processing with mediapipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb_frame)

    # Check if landmarks are detected in the frame
    if result.multi_face_landmarks:
        for face_landmarks in result.multi_face_landmarks:
            # Draw the landmarks on the face (points)
            for landmark in face_landmarks.landmark:
                x, y = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

            # Draw the face mesh connections (lines)
            for connection in mp_face_mesh.FACEMESH_TESSELATION:
                start_idx, end_idx = connection
                start = face_landmarks.landmark[start_idx]
                end = face_landmarks.landmark[end_idx]
                start_point = (int(start.x * frame.shape[1]), int(start.y * frame.shape[0]))
                end_point = (int(end.x * frame.shape[1]), int(end.y * frame.shape[0]))
                cv2.line(frame, start_point, end_point, (0, 255, 255), 1)

    return frame


def save_face(name, embeddings):
    """ Save the user's embeddings to a JSON file """
    import os
    folder_path = os.path.join("data", name)
    os.makedirs(folder_path, exist_ok=True)

    data = {
        "name": name,
        "embeddings": embeddings
    }

    # Save embeddings to a JSON file
    with open(os.path.join(folder_path, f"{name}_embeddings.json"), "w") as json_file:
        import json
        json.dump(data, json_file, indent=4)
    print(f"Embeddings saved successfully for {name}!")

    

def load_faces():
    """ Load registered faces from JSON file """
    try:
        with open(DATA_FILE, "r") as f:
            faces = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        faces = {}
    return faces

def recognize_face(embedding):
    """ Compare embedding with stored faces """
    faces = load_faces()
    if not faces:
        return "Unknown"

    best_match = "Unknown"
    best_score = 0.5  # Minimum similarity threshold

    for name, stored_embedding in faces.items():
        stored_embedding = np.array(stored_embedding)
        score = np.dot(embedding, stored_embedding) / (np.linalg.norm(embedding) * np.linalg.norm(stored_embedding))
        if score > best_score:
            best_score = score
            best_match = name

    return best_match


