import argparse
import cv2
from fingerprint_enhancer import enhance_fingerprint
import fingerprint_feature_extractor
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List
from scipy.spatial import cKDTree
from math import radians, degrees, cos, sin, atan2
from sklearn.linear_model import RANSACRegressor
from scipy.signal import find_peaks
from pycpd import RigidRegistration
from scipy.spatial.distance import cdist
import os


def align_labeled_point_sets(A_T, A_F, B_T, B_F):
    """
    Aligns point set B (with T and F labeled subgroups) to point set A.

    Args:
        A_T, A_F: list of (x, y) tuples for A's T and F subgroups
        B_T, B_F: list of (x, y) tuples for B's T and F subgroups

    Returns:
        aligned_B_T, aligned_B_F: transformed versions of B_T and B_F
    """
    # Combine A and B into full arrays
    A_combined = np.array(A_T + A_F, dtype=np.float64)
    B_combined = np.array(B_T + B_F, dtype=np.float64)

    # Keep track of label boundaries
    len_T = len(B_T)

    # Align B_combined to A_combined
    reg = RigidRegistration(X=A_combined, Y=B_combined)
    TY, _ = reg.register()

    # Split the aligned result
    aligned_B_T = [tuple(pt) for pt in TY[:len_T]]
    aligned_B_F = [tuple(pt) for pt in TY[len_T:]]

    return aligned_B_T, aligned_B_F

def nearest_neighbor_distances(A, B_aligned, bin_method="fd"):
    A = np.array(A)
    B_aligned = np.array(B_aligned)

    tree = cKDTree(A)
    distances, _ = tree.query(B_aligned)

    bins = optimal_bins(distances, method=bin_method)

    stats = {
        "mean": np.mean(distances),
        "median": np.median(distances),
        "std": np.std(distances),
        "min": np.min(distances),
        "max": np.max(distances),
        "num_points": len(distances),
        "within_1_unit": np.sum(distances < 1.0),
        "within_5_units": np.sum(distances < 5.0)
    }

    return distances, stats

def find_threshold(distances):
    bins = optimal_bins(distances, method="fd")
    hist, bin_edges = np.histogram(distances, bins, density=True)

    peaks, _ = find_peaks(hist)

    if len(peaks) == 0:
        print("No peaks found in histogram.")
        return None

    # First peak = smallest nonzero mode in distances
    first_peak_index = peaks[0]
    peak_value = hist[first_peak_index]

    half_max = peak_value / 2

    # Try finding where histogram falls below half max to the right of the peak
    right_half = np.where(hist[first_peak_index:] <= half_max)[0]

    if right_half.size > 0:
        right_idx = right_half[0] + first_peak_index
        reason = "drop below half-max"
    else:
        # Try to find a local minimum (valley) after the peak
        valleys, _ = find_peaks(-hist[first_peak_index:])
        if valleys.size > 0:
            right_idx = valleys[0] + first_peak_index
            reason = "first valley after peak"
        else:
            # Final fallback: fixed bin offset
            fallback_offset = 5
            right_idx = min(first_peak_index + fallback_offset, len(bin_edges) - 2)
            reason = "fixed offset fallback"

    threshold_x = bin_edges[right_idx]
    return threshold_x


def optimal_bins(data, method="fd"):
    n = len(data)
    data = np.asarray(data)
    range_ = np.max(data) - np.min(data)
    
    if method == "sturges":
        return int(np.ceil(np.log2(n) + 1))
    elif method == "rice":
        return int(np.ceil(2 * n ** (1/3)))
    elif method == "scott":
        bin_width = 3.5 * np.std(data) / (n ** (1/3))
    elif method == "fd":
        q75, q25 = np.percentile(data, [75 ,25])
        iqr = q75 - q25
        bin_width = 2 * iqr / (n ** (1/3))
    else:
        raise ValueError("Unknown method for bin selection.")

    return int(np.ceil(range_ / bin_width)) if bin_width > 0 else 10

def pair_points(A, B, threshold):
    """
    Pairs points from A and B based on a distance threshold.
    
    Parameters:
    - A (np.ndarray): 2D array or list of points (x, y) in the first set.
    - B (np.ndarray): 2D array or list of points (x, y) in the second set.
    - threshold (float): The maximum distance allowed for a pair to be considered.
    
    Returns:
    - paired_A (list): List of points from A that are paired with points from B.
    - paired_B (list): List of points from B that are paired with points from A.
    """
    
    # Convert to numpy arrays if A and B are lists
    A = np.array(A)
    B = np.array(B)
    
    # Compute all pairwise distances between points in A and B
    distances = cdist(A, B)
    
    # Find the points that can be paired based on the threshold
    pairs = np.where(distances <= threshold)
    
    # Extract the paired points from A and B
    paired_A = A[pairs[0]]
    paired_B = B[pairs[1]]
    
    return paired_A, paired_B

def compare_minutiae(db_terms, db_bifs, terminations, bifurcations):
    
    aligned_terminations, aligned_bifurcations = align_labeled_point_sets(db_terms, db_bifs, terminations, bifurcations)

    original_reference_points = np.vstack([db_terms, db_bifs])
    aligned_tested_points = np.vstack([aligned_terminations, aligned_bifurcations])

    distances, stats = nearest_neighbor_distances(original_reference_points, aligned_tested_points, bin_method="fd")

    threshold = find_threshold(distances)

    mse_val = None
    smape_val = None

    if threshold is not None:
        paired_A, paired_B = pair_points(original_reference_points, aligned_tested_points, threshold)
        mse_val = mse(paired_A, paired_B)
        smape_val = smape(paired_A, paired_B)
        return mse_val, smape_val
    else:
        return None, None

def get_best_matches(image, minutiae_db, spurious_threshold):
    enhanced_image = enhance_fingerprint(image)
    enhanced_image = (enhanced_image * 255).astype(np.uint8)
    terminations, bifurcations = fingerprint_feature_extractor.extract_minutiae_features(enhanced_image, spuriousMinutiaeThresh=spurious_threshold, invertImage=False, showResult=False, saveResult=False)

    terminations_coords = [(f.locX, f.locY) for f in terminations]
    bifurcations_coords = [(f.locX, f.locY) for f in bifurcations]

    scores = []
    

    for label, entry in minutiae_db.items():
        # Accumulate terminations if 'T' exists
        db_terms = []  # To accumulate all terminations
        db_bifs = []   # To accumulate all bifurcations

        if 'T' in entry:
            db_terms.extend(entry['T'])  # Directly extend with the list of terminations

        # Accumulate bifurcations if 'B' exists
        if 'B' in entry:
            db_bifs.extend(entry['B'])  # Directly extend with the list of bifurcations

        mse, smape = compare_minutiae(db_terms, db_bifs, terminations_coords, bifurcations_coords)
        if mse is None or smape is None:
            continue
        scores.append((label, mse, smape))
        top_10_scores = sorted(scores, key=lambda x: x[2])[:10]

    return top_10_scores

def mse(set_a, set_b):
    # Assuming set_a and set_b are lists of matched points (x, y)
    differences = np.array(set_a) - np.array(set_b)
    squared_diff = np.sum(differences ** 2, axis=1)
    mse_value = np.mean(squared_diff)
    return mse_value

def smape(set_a, set_b):
    set_a, set_b = np.array(set_a), np.array(set_b)
    diff = np.abs(set_a - set_b)
    denominator = (np.abs(set_a) + np.abs(set_b)) / 2.0
    return np.mean(2 * diff / denominator) * 100

def load_database(csv_path: str) -> Dict[str, Tuple[List[Tuple[int, int, float]], List[Tuple[int, int, Tuple[float, float, float]]]]]:
    """
    Loads a CSV database and returns a dictionary grouped by image_id.
    The dictionary contains tuples of (terminations, bifurcations) for each fingerprint.
    """
    # Load the CSV into a pandas DataFrame
    df = pd.read_csv(csv_path)
    
    # Group by 'image_id' to process each image's minutiae separately
    grouped = df.groupby('image_id')
    
    database = {}
    
    # Loop over each grouped image_id
    for image_id, group in grouped:
        terminations = []
        bifurcations = []
        
        # Process each row in the group (terminations and bifurcations)
        for _, row in group.iterrows():
            x, y = row['x'], row['y']
            
            if row['type'] == 'T':  # Termination
                theta = row['theta1']  # Orientation angle for termination
                terminations.append((x, y))
            elif row['type'] == 'B':  # Bifurcation
                thetas = row['theta1'], row['theta2'], row['theta3']  # 3 angles for bifurcation
                bifurcations.append((x, y))
        
        # Store the terminations and bifurcations for this image_id
        database[image_id] = {'T' : terminations, 'B': bifurcations}
    
    return database

description = 'Programul extrage lista de micro-structuri din imaginea inițială și o compară cu baza de date a amprentelor digitale în format CSV.'

parser = argparse.ArgumentParser(prog='search_database.py',
                                 description=description,
                                 usage='search_database.py -f <fisierul de intrare> -df <fisierul CSV - baza de date> -sp <spurious threshold>',
                                 formatter_class=argparse.RawTextHelpFormatter )

parser.add_argument("-f", "--input_file", required=True, help="Fișierul de intrare", type=str)
parser.add_argument("-db", "--database_file", default='minutiae_db.csv', required=False, help="Numele bazei de date generat de create_database.py", type=str)
parser.add_argument("-sp", "--spurious_threshold", default=10, required=False, help="Filtreaza caracteristicile astfel incat sa fie eliminate cele care sunt mai apropiate de acest parametru", type=int)


args = parser.parse_args()


if os.path.isfile(args.input_file):
    print(f"Loading image: {args.input_file}")
    img = cv2.imread(args.input_file, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error: Could not load image from {args.input_file}")
else: 
    raise FileNotFoundError(f"Input file '{args.input_file}' does not exist.")    

if os.path.isfile(args.database_file):
    minutiae_db =  load_database(args.database_file)   
else: 
    raise FileNotFoundError(f"Database file '{args.database_file}' does not exist.")    

top10 = get_best_matches(img, minutiae_db, args.spurious_threshold)

for label, mse, smape in top10:
    print(f"Label: {label}, MSE: {mse}, SMAPE: {smape}")

# print(top10)
