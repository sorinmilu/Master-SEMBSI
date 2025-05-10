import argparse
from torchvision import transforms
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import os
import numpy as np
from torchvision.ops import nms
from tqdm import tqdm
import cv2
from sklearn.preprocessing import normalize
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import umap
from sklearn.decomposition import PCA
import textwrap
import mediapipe as mp
import numpy as np

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def do_tsne(embedding_db, output_file, embedding_size):
    all_embeddings = []
    all_labels = []

    for label, vectors in embedding_db.items():
        for vec in vectors:
            all_embeddings.append(vec)
            all_labels.append(label)

    # Convert to NumPy array
    all_embeddings = np.array(all_embeddings)  # shape: (N, D)


    normalized_embeddings = normalize(all_embeddings, norm='l2')
    n_samples, n_features = normalized_embeddings.shape

    pca_dimensions = 0    

    if embedding_size >= 32 and embedding_size <= 64:    
        pca_dimensions = min(n_samples, n_features, embedding_size, 40)
    elif embedding_size > 64 and embedding_size <= 128:
        pca_dimensions = min(n_samples, n_features, 64)
    elif embedding_size > 128 and embedding_size <= 256:
        pca_dimensions = min(n_samples, n_features, 100)
    elif embedding_size > 256 and embedding_size <= 512:
        pca_dimensions = min(n_samples, n_features, 150)


    if pca_dimensions > 0:
        pca = PCA(n_components=pca_dimensions)
        reduced_vectors = pca.fit_transform(normalized_embeddings)
    else:
        reduced_vectors = normalized_embeddings
    
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    reduced = tsne.fit_transform(normalized_embeddings)  # shape: (N, 2)
    unique_labels = list(set(all_labels))
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    label_indices = [label_to_idx[label] for label in all_labels]

    custom_palette = [
        "#000000",  # Black
        "#808080",  # Dark Gray
        "#D3D3D3",  # Light Gray
        "#FF0000",  # Red
        "#00FF00",  # Green
        "#0000FF",  # Blue
        "#FFFF00",  # Yellow
        "#FF00FF",  # Magenta
        "#00FFFF",  # Cyan
        "#800000",  # Maroon
        "#808000",  # Olive
        "#008000",  # Dark Green
        "#800080",  # Purple
        "#008080",  # Teal
        "#000080",  # Navy
        "#FFA500",  # Orange
        "#A52A2A",  # Brown
        "#8A2BE2",  # Blue Violet
        "#5F9EA0",  # Cadet Blue
        "#7FFF00",  # Chartreuse
    ]

    unique_labels = list(set(all_labels))
    num_labels = len(unique_labels)

    # Ensure the palette has enough colors for all labels
    if num_labels > len(custom_palette):
        raise ValueError(f"Number of labels ({num_labels}) exceeds the number of available colors in the custom palette ({len(custom_palette)}).")

    # Map labels to colors
    palette = custom_palette[:num_labels]

    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=reduced[:, 0], y=reduced[:, 1], hue=all_labels, palette=palette, s=60)
    plt.title("t-SNE projection of embeddings by label")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f"{output_file}_{embedding_size}.png", dpi=300) 

def do_umap(embedding_db, output_file, embedding_size):
    all_embeddings = []
    all_labels = []

    for label, vectors in embedding_db.items():
        for vec in vectors:
            all_embeddings.append(vec)
            all_labels.append(label)

    # Convert to NumPy array
    all_embeddings = np.array(all_embeddings)  
    normalized_embeddings = normalize(all_embeddings, norm='l2')

    n_samples, n_features = normalized_embeddings.shape

    pca_dimensions = 0    


    if embedding_size >= 32 and embedding_size <= 64:    
        pca_dimensions = min(n_samples, n_features, embedding_size, 40)
    elif embedding_size > 64 and embedding_size <= 128:
        pca_dimensions = min(n_samples, n_features, 64)
    elif embedding_size > 128 and embedding_size <= 256:
        pca_dimensions = min(n_samples, n_features, 100)
    elif embedding_size > 256 and embedding_size <= 512:
        pca_dimensions = min(n_samples, n_features, 150)



    if pca_dimensions > 0:
        pca = PCA(n_components=pca_dimensions)
        reduced_vectors = pca.fit_transform(normalized_embeddings)
    else:
        reduced_vectors = normalized_embeddings

    umap_model = umap.UMAP(n_components=2, random_state=42)
    umap_result = umap_model.fit_transform(all_embeddings)


    custom_palette = [
        "#000000",  # Black
        "#808080",  # Dark Gray
        "#D3D3D3",  # Light Gray
        "#FF0000",  # Red
        "#00FF00",  # Green
        "#0000FF",  # Blue
        "#FFFF00",  # Yellow
        "#FF00FF",  # Magenta
        "#00FFFF",  # Cyan
        "#800000",  # Maroon
        "#808000",  # Olive
        "#008000",  # Dark Green
        "#800080",  # Purple
        "#008080",  # Teal
        "#000080",  # Navy
        "#FFA500",  # Orange
        "#A52A2A",  # Brown
        "#8A2BE2",  # Blue Violet
        "#5F9EA0",  # Cadet Blue
        "#7FFF00",  # Chartreuse
    ]

    unique_labels = list(set(all_labels))
    num_labels = len(unique_labels)

    # Ensure the palette has enough colors for all labels
    if num_labels > len(custom_palette):
        raise ValueError(f"Number of labels ({num_labels}) exceeds the number of available colors in the custom palette ({len(custom_palette)}).")

    # Map labels to colors
    palette = custom_palette[:num_labels]


    # Plot UMAP
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=umap_result[:, 0], y=umap_result[:, 1], hue=all_labels, palette=palette, s=60)
    plt.title("UMAP projection of embeddings by label")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f"{output_file}_{embedding_size}.png", dpi=300) 



def predict(image_path):
    """
    Processes the image and returns the aligned face mesh landmarks.

    Args:
        image_path (str): Path to the input image.

    Returns:
        list: A vector containing the aligned face mesh landmarks (x, y, z).
    """
    # Open the image
    original_image = cv2.imread(image_path)
    if original_image is None:
        raise ValueError(f"Could not read image: {image_path}")

    # Convert the image to RGB (MediaPipe expects RGB images)
    rgb_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    # Process the image with MediaPipe Face Mesh
    results = face_mesh.process(rgb_image)

    # Check if any face landmarks were detected
    if not results.multi_face_landmarks:
        raise ValueError(f"No face landmarks detected in image: {image_path}")

    # Extract the landmarks of the first detected face
    landmarks = results.multi_face_landmarks[0].landmark

    # Convert landmarks to a NumPy array
    landmarks_array = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])

    # Align the landmarks to a standard coordinate system
    aligned_landmarks = align_landmarks(landmarks_array)

    # Flatten the aligned landmarks into a 1D vector and return
    return aligned_landmarks.flatten().tolist()

def align_landmarks(landmarks):
    """
    Aligns the face landmarks to a standard coordinate system.

    Args:
        landmarks (np.ndarray): The face landmarks as a NumPy array of shape (N, 3).

    Returns:
        np.ndarray: The aligned landmarks.
    """
    # Compute the centroid of the landmarks
    centroid = np.mean(landmarks, axis=0)

    # Center the landmarks around the origin
    centered_landmarks = landmarks - centroid

    # Perform Procrustes alignment (optional: align to a template if needed)
    # Here, we normalize the landmarks to unit scale
    scale = np.linalg.norm(centered_landmarks)
    normalized_landmarks = centered_landmarks / scale

    return normalized_landmarks


description = 'Program that extracts the landmarks from a set of images (using mediapipe library) and tries to inspect their clustering properties. The program generates t-SNE and UMAP charts of the embeddings.'

epilog=r""" 
The structure of the directory has to be very precise, it has to have one level of subdirectories and within those subdirectories the images. 
The images have to be extracted faces (not casual images) 
    \../../DATA/face/clustering/short/ 
    ├── face1 
    ├── face2 
    └── face3 
"""
epilog_text = textwrap.dedent(epilog).strip()

parser = argparse.ArgumentParser(prog='infer_folder.py',
                                 description=description,
                                 usage='infer_folder.py -id <input_directory> -os <tsne_chart_output_name> -ou <umap_chart_output_name>',
                                 epilog=epilog_text,
                                 formatter_class=argparse.RawTextHelpFormatter )

parser.add_argument("-id", "--input_directory", required=True, help="Directory with the testing images.", type=str)
parser.add_argument("-os", "--output_tsne", default='output_tsne', required=False, help="Name of the t-sne chart. No extension (png will be added, together with the size of the embeddings vector)", type=str)
parser.add_argument("-ou", "--output_umap", default='output_umap', required=False, help="Name of the umap chart. No extension (png will be added, together with the size of the embeddings vector)", type=str)

args = parser.parse_args()

input_dir = args.input_directory

# Check if input directory exists
if not os.path.exists(input_dir):
    raise FileNotFoundError(f"Input directory '{input_dir}' does not exist.")

# Count total files
file_list = []
valid_extensions = {".jpg", ".jpeg", ".png"}
for root, _, files in os.walk(input_dir):
    for file in files:
        if os.path.splitext(file)[1].lower() in valid_extensions:
            file_list.append(os.path.join(root, file))

total_files = len(file_list)
print(f"Total files to process: {total_files}")

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)


embedding_db = {}

# Process files with progress bar
with tqdm(total=total_files, desc="Processing files", unit="file") as pbar:
    for file_path in file_list:
        label = os.path.basename(os.path.dirname(file_path))     
        embeddings = predict(file_path)
        if label not in embedding_db:
            embedding_db[label] = []  # Initialize an empty list for the label if it doesn't exist

        embedding_db[label].append(embeddings) 
        pbar.update(1)

print("Data collection complete")


do_tsne(embedding_db, args.output_tsne, 460)
do_umap(embedding_db, args.output_umap, 460)







print("Processing complete.")
