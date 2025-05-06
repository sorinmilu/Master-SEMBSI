import argparse
import cv2
import os
from fingerprint_enhancer import enhance_fingerprint
import fingerprint_feature_extractor
import numpy as np
from scipy.spatial import cKDTree
import math
from pycpd import RigidRegistration
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import find_peaks
from scipy.spatial.distance import cdist



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


def find_threshold(distances):
    # Step 1: Generate the histogram
    print('-----------------')
    print(distances.shape)
    print('-----------------')
    print(f"Min: {np.min(distances)}")
    print(f"Max: {np.max(distances)}")


    bins = optimal_bins(distances, method="fd")
    if bins > 30:
        bins = 30

    print(bins)

    hist, bin_edges = np.histogram(distances, bins, density=True)
    

    # Step 3: Find the peaks in the cumulative distribution
    peaks, _ = find_peaks(hist)
    
    if len(peaks) > 0:
        # Step 4: Get the first peak
        first_peak_index = peaks[0]
        peak_value = hist[first_peak_index]
        
        # Step 5: Find the point where the peak reaches half height, farther from the origin
        half_max = peak_value / 2
        
        # Search to the right of the first peak for the point where the cumulative histogram drops below half_max
        right_idx = np.where(hist[first_peak_index:] <= half_max)[0][0] + first_peak_index
        
        # Step 6: Return the x-value corresponding to the point where the peak reaches half height
        threshold_x = bin_edges[right_idx]


        return threshold_x
    else:
        print("No peaks found in the cumulative histogram.")
        return None


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

def nearest_neighbor_distances(A, B_aligned, bin_method="fd"):
    A = np.array(A)
    B_aligned = np.array(B_aligned)

    tree = cKDTree(A)
    distances, _ = tree.query(B_aligned)

    bins = optimal_bins(distances, method=bin_method)
    if bins > 30:
        bins = 30

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


def draw_fingerprint_features(image, FeaturesTerminations, FeaturesBifurcations):

    color_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # Draw termination minutiae (red circles)
    for curr_minutiae in FeaturesTerminations:
        row, col = int(curr_minutiae.locY), int(curr_minutiae.locX)  # OpenCV uses (x, y) as (col, row)
        cv2.circle(color_image, (row, col), 6, (0, 0, 255), 2)  # Red circle

    # Draw bifurcation minutiae (blue circles)
    for curr_minutiae in FeaturesBifurcations:
        row, col = int(curr_minutiae.locY), int(curr_minutiae.locX)
        cv2.circle(color_image, (row, col), 6, (255, 0, 0), 2)  # Blue circle

    return color_image


def draw_fingerprint_features_from_lists(image, terminations_coordinates, bifurcation_coordinates):

    color_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # Draw termination minutiae (red circles)
    for (col, row) in terminations_coordinates:
        cv2.circle(color_image, (int(row), int(col)), 6, (0, 0, 255), 2)  # Red circle

    # Draw bifurcation minutiae (blue circles)
    for (col, row) in bifurcation_coordinates:
        cv2.circle(color_image, (int(row), int(col)), 6, (255, 0, 0), 2)  # Blue circle

    return color_image

def draw_fingerprint_features_from_lists_no_background(image, terminations_coordinates, bifurcation_coordinates):

    color_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # Draw termination minutiae (red circles)
    for (col, row) in terminations_coordinates:
        cv2.circle(color_image, (int(row), int(col)), 6, (0, 0, 255), 2)  # Red circle

    # Draw bifurcation minutiae (blue circles)
    for (col, row) in bifurcation_coordinates:
        cv2.circle(color_image, (int(row), int(col)), 6, (255, 0, 0), 2)  # Blue circle

    return color_image

def draw_termination_lines(image, FeaturesTerminations, line_length=10, color=(0, 0, 255), thickness=1):
    """
    Draws small lines on the image at the (x, y) locations with the angle from the list in the third element.

    Args:
        image (np.ndarray): The OpenCV image on which to draw.
        terminations (list): List of tuples, where each tuple contains (x, y, [angle]).
        line_length (int): Length of the line to draw.
        color (tuple): Color of the line in BGR format.
        thickness (int): Thickness of the line.

    Returns:
        np.ndarray: The image with lines drawn.
    """

    color_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    for curr_minutiae in FeaturesTerminations:
        row, col = int(curr_minutiae.locY), int(curr_minutiae.locX)  # OpenCV uses (x, y) as (col, ro
        orient = curr_minutiae.Orientation  

        if len(orient) == 1:  # Ensure the third element is a list with one angle
            angle = orient[0]
            # Convert angle to radians
            angle_rad = math.radians(angle)

            # Calculate the end point of the line
            row_end = int(row + line_length * math.cos(angle_rad))
            col_end = int(col - line_length * math.sin(angle_rad))  # Subtract because OpenCV's origin is top-left

            # Draw the line
            cv2.line(color_image, (int(row), int(col)), (row_end, col_end), color, thickness)

    return color_image


def create_side_by_side_image(img1, img2, label1, label2):
    """
    Creates an OpenCV image showing two images side by side with labels centered above them.

    Args:
        img1 (np.ndarray): First grayscale image.
        img2 (np.ndarray): Second grayscale image.
        label1 (str): Label for the first image (e.g., file name without path).
        label2 (str): Label for the second image (e.g., file name without path).

    Returns:
        np.ndarray: Combined image with labels and spacing.
    """
    # Ensure both images are grayscale
    if len(img1.shape) == 2:
        img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    if len(img2.shape) == 2:
        img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)

    # Get dimensions of the images
    h1, w1, _ = img1.shape
    h2, w2, _ = img2.shape

    # Determine the height of the combined image (max height of both images + space for labels)
    label_space = 50  # Space for labels above the images
    combined_height = max(h1, h2) + label_space

    # Create a blank canvas for the combined image
    combined_width = w1 + w2 + 50  # Add 50 pixels of space between the images
    combined_image = np.ones((combined_height, combined_width, 3), dtype=np.uint8) * 255  # White background

    # Place the first image on the canvas
    combined_image[label_space:label_space + h1, :w1] = img1

    # Place the second image on the canvas
    combined_image[label_space:label_space + h2, w1 + 50:w1 + 50 + w2] = img2

    # Add labels above the images
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_thickness = 2

    # Calculate text size and position for the first label
    text_size1 = cv2.getTextSize(label1, font, font_scale, font_thickness)[0]
    text_x1 = (w1 - text_size1[0]) // 2
    text_y1 = (label_space - text_size1[1]) // 2 + text_size1[1]
    cv2.putText(combined_image, label1, (text_x1, text_y1), font, font_scale, (0, 0, 0), font_thickness)

    # Calculate text size and position for the second label
    text_size2 = cv2.getTextSize(label2, font, font_scale, font_thickness)[0]
    text_x2 = w1 + 50 + (w2 - text_size2[0]) // 2
    text_y2 = (label_space - text_size2[1]) // 2 + text_size2[1]
    cv2.putText(combined_image, label2, (text_x2, text_y2), font, font_scale, (0, 0, 0), font_thickness)

    return combined_image


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

def compare_fingerprint_images(reference_image, tested_image, save_images=False, images_prefix='output'):

    # Enhancing images

    if save_images:
        original_images = create_side_by_side_image(reference_image, tested_image, os.path.basename(args.first_input_file), os.path.basename(args.second_input_file))
        cv2.imwrite(f"{images_prefix}_original.png", original_images)

    reference_enhanced_image = enhance_fingerprint(reference_image)
    reference_enhanced_image = (reference_enhanced_image * 255).astype(np.uint8)

    tested_enhanced_image = enhance_fingerprint(tested_image)
    tested_enhanced_image = (tested_enhanced_image * 255).astype(np.uint8)

    if save_images:
        original_images = create_side_by_side_image(reference_enhanced_image, tested_enhanced_image, os.path.basename(args.first_input_file), os.path.basename(args.second_input_file))
        cv2.imwrite(f"{images_prefix}_enhanced.png", original_images)


    reference_terminations, reference_bifurcations = fingerprint_feature_extractor.extract_minutiae_features(reference_enhanced_image, spuriousMinutiaeThresh=10, invertImage=False, showResult=False, saveResult=False)

    reference_terminations_coords = [(f.locX, f.locY) for f in reference_terminations]
    reference_bifurcations_coords = [(f.locX, f.locY) for f in reference_bifurcations]

    tested_terminations, tested_bifurcations = fingerprint_feature_extractor.extract_minutiae_features(tested_enhanced_image, spuriousMinutiaeThresh=10, invertImage=False, showResult=False, saveResult=False)

    tested_terminations_coords = [(f.locX, f.locY) for f in tested_terminations]
    tested_bifurcations_coords = [(f.locX, f.locY) for f in tested_bifurcations]

    if save_images:
        reference_skeleton_image = cv2.ximgproc.thinning(reference_enhanced_image)
        tested_skeleton_image = cv2.ximgproc.thinning(tested_enhanced_image)

        skeleton_images = create_side_by_side_image(reference_skeleton_image, tested_skeleton_image, os.path.basename(args.first_input_file), os.path.basename(args.second_input_file))
        cv2.imwrite(f"{images_prefix}_skeleton.png", skeleton_images)

        reference_features_image = draw_fingerprint_features(reference_skeleton_image, reference_terminations, reference_bifurcations)
        tested_features_image = draw_fingerprint_features(tested_skeleton_image, tested_terminations, tested_bifurcations)

        feature_images = create_side_by_side_image(reference_features_image, tested_features_image, os.path.basename(args.first_input_file), os.path.basename(args.second_input_file))

        cv2.imwrite(f"{images_prefix}_features.png", feature_images)

        int_aligned_terminations = [(int(x), int(y)) for (x, y) in tested_terminations_coords]
        int_aligned_bifurcations = [(int(x), int(y)) for (x, y) in tested_bifurcations_coords]

        post_alignment_image = draw_fingerprint_features_from_lists(reference_skeleton_image, int_aligned_terminations, int_aligned_bifurcations)
        post_alignment_image = draw_fingerprint_features_from_lists_no_background(reference_skeleton_image, int_aligned_terminations, int_aligned_bifurcations)
        post_alignment_results = create_side_by_side_image(reference_features_image, post_alignment_image, os.path.basename(args.first_input_file), os.path.basename(args.second_input_file))
        cv2.imwrite(f"{images_prefix}_post_alignment.png", post_alignment_results)

    aligned_terminations, aligned_bifurcations = align_labeled_point_sets(reference_terminations_coords, reference_bifurcations_coords, tested_terminations_coords, tested_bifurcations_coords)

    int_aligned_terminations = [(int(x), int(y)) for (x, y) in aligned_terminations]
    int_aligned_bifurcations = [(int(x), int(y)) for (x, y) in aligned_bifurcations]

    original_reference_points = np.vstack([reference_terminations_coords, reference_bifurcations_coords])
    aligned_tested_points = np.vstack([aligned_terminations, aligned_bifurcations])


    distances, stats = nearest_neighbor_distances(original_reference_points, aligned_tested_points, bin_method="fd")

    threshold = find_threshold(distances)
    if threshold is not None:
        paired_A, paired_B = pair_points(original_reference_points, aligned_tested_points, threshold)
        mse_val = mse(paired_A, paired_B)
        smape_val = smape(paired_A, paired_B)

        if save_images:
            bins = optimal_bins(distances, method="fd")
            if bins > 30:
                bins = 30
            plt.figure(figsize=(8, 4))
            sns.histplot(distances, bins=bins, kde=True, color='teal')
            plt.axvline(x=threshold, color='g', linestyle='--', label="Threshold at Half Height")
            plt.title(f"Distribution of Nearest Neighbor Distances (bins: {bins}, method: fd)")
            plt.xlabel("Distance to nearest point in A")
            plt.ylabel("Number of B points")
            plt.grid(True)
            plt.tight_layout()

            # Save the plot automatically to a predefined path
            save_path = f"{images_prefix}_distance_distribution.png"
            plt.savefig(save_path)
            plt.close()


        return mse_val, smape_val
    else:
        print("No valid threshold found.")
        return None, None

    


description = 'Programul extrage caracteristicile locale din două imagini de amprente și le compară. Programul produce o serie de imagini care ilustrează rezultatele.'

parser = argparse.ArgumentParser(prog='compare_fp.py',
                                 description=description,
                                 usage='compare_fp.py -fa <first_image> -fb <second_image>',
                                 formatter_class=argparse.RawTextHelpFormatter )

parser.add_argument("-fa", "--first_input_file", required=True, help="Input image file", type=str)
parser.add_argument("-fb", "--second_input_file", required=True, help="Input image file", type=str)


args = parser.parse_args()

if os.path.isfile(args.first_input_file):
    print(f"Loading first image: {args.first_input_file}")
    img1 = cv2.imread(args.first_input_file, cv2.IMREAD_GRAYSCALE)
    if img1 is None:
        print(f"Error: Could not load image from {args.first_input_file}")
else:
    raise FileNotFoundError(f"First input file '{args.first_input_file}' does not exist.")


if os.path.isfile(args.first_input_file):
    print(f"Loading second image: {args.second_input_file}")
    img2 = cv2.imread(args.second_input_file, cv2.IMREAD_GRAYSCALE)
    if img2 is None:
        print(f"Error: Could not load image from {args.second_input_file}")    
else:
    raise FileNotFoundError(f"Second input file '{args.second_input_file}' does not exist.")



mse, smape = compare_fingerprint_images(img1, img2, save_images=True, images_prefix='output')

print(f"MSE: {mse}")
print(f"SMAPE: {smape}")


