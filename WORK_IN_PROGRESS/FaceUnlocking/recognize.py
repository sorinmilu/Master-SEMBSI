import cv2
import numpy as np
import os
from utils import extract_embedding, draw_face_landmarks
import json
import time

# Function to load saved embeddings from JSON file
def load_embeddings(name):
    folder_path = os.path.join("data", name)
    try:
        with open(os.path.join(folder_path, f"{name}_embeddings.json"), "r") as json_file:
            data = json.load(json_file)
            return np.array(data["embeddings"])
    except FileNotFoundError:
        print(f"No data found for {name}.")
        return None

# Function to compare embeddings (Euclidean distance)
def compare_embeddings(known_embedding, test_embedding):
    distance = np.linalg.norm(known_embedding - test_embedding)
    return distance

# Function to recognize the face
def recognize_face(frame, known_embeddings, known_names):
    embedding = extract_embedding(frame)
    if embedding is not None:
        # Compare with known embeddings
        distances = [compare_embeddings(known_embedding, embedding) for known_embedding in known_embeddings]
        
        # Find the closest match
        min_distance_idx = np.argmin(distances)
        min_distance = distances[min_distance_idx]

        # If distance is below a threshold, consider it a match
        if min_distance < 0.6:  # Adjust threshold as needed
            name = known_names[min_distance_idx]
            cv2.putText(frame, f"Hello, {name}!", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        else:
            cv2.putText(frame, "No match found", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
    
    return frame

# Main function to run the recognition process
def main():
    # Load the known embeddings and names
    known_names = [name for name in os.listdir("data") if os.path.isdir(os.path.join("data", name))]
    known_embeddings = []

    # Load embeddings for all known names
    for name in known_names:
        embeddings = load_embeddings(name)
        if embeddings is not None:
            known_embeddings.append(np.mean(embeddings, axis=0))  # Use the average embedding for each person

    # Initialize the camera
    cap = cv2.VideoCapture(1)  # Change to your correct camera index if necessary
    print("Press 'q' to quit the recognition.")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        
        # Draw face landmarks
        frame_with_landmarks = draw_face_landmarks(frame)

        # Recognize face and overlay text on frame
        frame_with_recognition = recognize_face(frame_with_landmarks, known_embeddings, known_names)

        # Display the frame
        cv2.imshow("Face Recognition", frame_with_recognition)

        # Check if the user pressed 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
