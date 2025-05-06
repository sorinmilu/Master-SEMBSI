import cv2
import os
import json
import numpy as np
import time
from utils import extract_embedding, save_face, draw_face_landmarks

# Step 1: Capture the user's name before starting the camera
name = input("Enter your name: ")

# Step 2: Create a folder to save data for the user
folder_path = os.path.join("data", name)
os.makedirs(folder_path, exist_ok=True)

# Step 3: Initialize the camera feed
cap = cv2.VideoCapture(1)  # Change to your correct camera index if necessary

# Step 4: Wait for 5 seconds before starting face capture
print("Please look at the camera. Starting capture in 5 seconds...")
time.sleep(5)  # Wait for 5 seconds

# Step 5: Initialize variables for capturing frames and embeddings
count = 0
embeddings = []

# Step 6: Start capturing faces and drawing the face landmarks
while count < 5:
    ret, frame = cap.read()
    if not ret:
        continue

    # Step 7: Draw face landmarks on the frame using the utility function
    frame_with_landmarks = draw_face_landmarks(frame)

    # Extract the embedding for the current frame
    embedding = extract_embedding(frame)
    if embedding is not None:
        embeddings.append(embedding)

        # Draw the embedding as text (first 5 elements) on the image
        embedding_text = f"Embedding: {embedding[:5]}"  # Display only first 5 values
        cv2.putText(frame_with_landmarks, embedding_text, (20, 30 + count * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2, cv2.LINE_AA)

        # Save the frame with the full overlay (landmarks, lines, and embedding text)
        cv2.imwrite(os.path.join(folder_path, f"captured_face_{count + 1}.jpg"), frame_with_landmarks)
        count += 1
        print(f"Captured {count}/5 samples.")

        # Add a 1-second delay after each successful capture
        time.sleep(1)

    # Display the camera feed with face landmarks
    cv2.imshow("Face Registration - Press 'q' to Quit", frame_with_landmarks)

    # Check if the user pressed 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Step 8: Release camera and close the window
cap.release()
cv2.destroyAllWindows()

# Step 9: Save the embeddings to JSON in the user's folder
if embeddings:
    embeddings = [embedding.tolist() if isinstance(embedding, np.ndarray) else embedding for embedding in embeddings]
    
    save_face(name, embeddings)
    print(f"Face registered successfully as {name}!")
