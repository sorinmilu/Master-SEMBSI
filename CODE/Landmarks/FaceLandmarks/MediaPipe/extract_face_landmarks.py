import cv2
import mediapipe as mp
import argparse
import os

def draw_landmarks_with_numbers_and_mesh(image_path, output_path="landmarks_numbered_mesh.jpg"):
    """
    Loads an image, detects face landmarks using MediaPipe, and draws the
    landmarks with their numbers and the face mesh on the image.

    Args:
        image_path (str): Path to the input image file.
        output_path (str): Path to save the output image.
    """
    mp_face_mesh = mp.solutions.face_mesh
    mp_drawing = mp.solutions.drawing_utils
    drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as face_mesh:

        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not open or find the image at {image_path}")
            return

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_image)

        if results.multi_face_landmarks:
            # Convert the image to grayscale
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # Convert grayscale back to BGR so we can draw colored landmarks
            gray_image_bgr = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)

            for face_landmarks in results.multi_face_landmarks:
                h, w, _ = image.shape

                # Draw mesh on the grayscale image
                mp_drawing.draw_landmarks(
                    image=gray_image_bgr,
                    landmark_list=face_landmarks,
                    connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=drawing_spec,
                    connection_drawing_spec=mp_drawing.DrawingSpec(thickness=2, color=(0, 255, 255)))

                # Draw landmarks with numbers
                for i, landmark in enumerate(face_landmarks.landmark):
                    x, y = int(landmark.x * w), int(landmark.y * h)
                    cv2.circle(gray_image_bgr, (x, y), 3, (0, 255, 0), -1)
                    cv2.putText(gray_image_bgr, str(i), (x + 5, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)

            # Save the final image
            cv2.imwrite(output_path, gray_image_bgr)
            print(f"Landmarks with numbers and mesh saved to {output_path} (grayscale background with colored landmarks)")
        else:
            print("No face detected in the image.")

parser = argparse.ArgumentParser(prog='infer_dir.py',
                                 description="Scriptul extrage si afiseaza punctele de interes ale fetei folosind MediaPipe",
                                 usage='infer_dir.py -m <model file> -df <images directory>',
                                 epilog="This program gets a directory with jpeg images as an argument, runs the model on each image and prints the class of fingerprints found in the images")


parser.add_argument('-f', "--input_file", help="Imaginea de intrare")
parser.add_argument('-o', "--output_file", default="landmarks_numbered_mesh.jpg" ,help="Imaginea de intrare")

args = parser.parse_args()

if not os.path.exists(args.input_file):
    raise FileNotFoundError(f"Input file '{args.input_file}' does not exist.")


draw_landmarks_with_numbers_and_mesh(args.input_file, args.output_file)