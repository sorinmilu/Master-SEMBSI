import os
import cv2
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from scipy.spatial import procrustes

# Init MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True)

def process_image(image_path, max_size=512):
    img = cv2.imread(image_path)
    if img is None:
        return None

    # Resize if too big
    h, w = img.shape[:2]
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)))

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)
    if not results.multi_face_landmarks:
        return None

    landmarks = results.multi_face_landmarks[0].landmark
    coords = np.array([[lm.x, lm.y] for lm in landmarks])
    return coords

def align_landmarks(landmarks, reference=None):
    """
    Aliniază trăsăturile faciale la un sistem de coordonate standard.

    Args:
        landmarks (np.ndarray): Landmark-urile ca array (N, 2).
        reference (np.ndarray): Landmark-urile template pentru aliniere (opțional).

    Returns:
        np.ndarray: Landmark-urile aliniate.
        np.ndarray: Centroidul (x, y).
    """
    centroid = np.mean(landmarks, axis=0)
    centered = landmarks - centroid
    scale = np.linalg.norm(centered)
    if scale == 0:
        return centered, centroid
    normalized = centered / scale

    if reference is not None:
        # Procrustes alignment (returns aligned landmark set as 'm2')
        _, m2, _ = procrustes(reference, normalized)
        return m2, centroid

    return normalized, centroid

def extract_landmarks(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = face_mesh.process(rgb_img)
    if not results.multi_face_landmarks:
        raise ValueError(f"No face landmarks found in {image_path}")

    landmarks = results.multi_face_landmarks[0].landmark
    landmarks_array = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
    return landmarks_array

def build_landmark_dict(image_dir, use_reference_alignment=True):
    landmark_dict = {}
    reference_template = None

    for fname in os.listdir(image_dir):
        path = os.path.join(image_dir, fname)
        try:
            landmarks = extract_landmarks(path)
            landmarks_2d = landmarks[:, :2]

            if use_reference_alignment and reference_template is not None:
                aligned, centroid = align_landmarks(landmarks_2d, reference_template)
            else:
                aligned, centroid = align_landmarks(landmarks_2d)

            if reference_template is None and use_reference_alignment:
                reference_template = aligned  # Save for future alignment

            landmark_dict[fname] = {
                'centroid': centroid.tolist(),
                'landmarks': aligned.tolist()
            }
        except Exception as e:
            print(f"Error processing {fname}: {e}")

    return landmark_dict

def plot_heatmap(landmark_dict, output_path="landmark_heatmap.png"):
    all_points = []
    for entry in landmark_dict.values():
        aligned_landmarks = np.array(entry["landmarks"])
        all_points.extend(aligned_landmarks)

    all_points = np.array(all_points)
    x = all_points[:, 0]
    y = all_points[:, 1]

    plt.figure(figsize=(8, 8))
    sns.kdeplot(x=x, y=y, cmap="viridis", fill=True, thresh=0, levels=100)
    plt.title("Landmark Heatmap")
    plt.axis("equal")
    plt.gca().invert_yaxis()  # Flip Y axis to match image coordinates
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved heatmap to: {output_path}")

# --- Usage ---


description = 'Program that extracts the landmarks from a set of images (using mediapipe library) and tries to inspect their clustering properties. The program generates t-SNE and UMAP charts of the embeddings.'

parser = argparse.ArgumentParser(prog='infer_folder.py',
                                 description=description,
                                 usage='infer_folder.py -id <input_directory> -os <tsne_chart_output_name> -ou <umap_chart_output_name>',
                                 formatter_class=argparse.RawTextHelpFormatter )

parser.add_argument("-id", "--input_directory", required=True, help="Directory with the testing images.", type=str)
parser.add_argument("-os", "--output_heatmap", default="landmark_heatmap.png", required=False, help="Output name for the heatmap", type=str)

args = parser.parse_args()

input_dir = args.input_directory


# Usage

landmark_data = build_landmark_dict(input_dir, use_reference_alignment=True)
plot_heatmap(landmark_data, output_path=args.output_heatmap)


