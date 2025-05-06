import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import argparse
import math
from scipy.ndimage import rotate
from scipy.signal import find_peaks
from fingerprint_enhancer import enhance_fingerprint
import fingerprint_feature_extractor
from tqdm import tqdm
import textwrap
import csv

def extract_minutiae(image_path, spurious_threshold):


    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error: Could not load image from {image_path}")
        return
    
    fp_enhanced_image = enhance_fingerprint(img)
    fp_enhanced_image = (fp_enhanced_image * 255).astype(np.uint8)

    FeaturesTerminations, FeaturesBifurcations = fingerprint_feature_extractor.extract_minutiae_features(fp_enhanced_image, spuriousMinutiaeThresh=spurious_threshold, invertImage=False, showResult=False, saveResult=False)
    minutiae = {
        'Terminations': FeaturesTerminations,   
        'Bifurcations': FeaturesBifurcations   
    }
    return minutiae


def write_minutiae_to_csv(minutiae_db, output_csv):

    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['image_id', 'x', 'y', 'theta1', 'theta2', 'theta3', 'type'])

        print(len(minutiae_db))

        with tqdm(total=len(minutiae_db), desc="Exporting minutiae", unit="fp") as pbar:
            for fp in minutiae_db:
                pbar.set_description(f"Writing: {fp}")     

                for minutiae in minutiae_db[fp]:

                    # Write terminations
                    for idx, curr_minutiae in enumerate(minutiae['Terminations']):
                        x = int(curr_minutiae.locX)
                        y = int(curr_minutiae.locY)
                        thetas = curr_minutiae.Orientation[0]
                        mtype = 'T'
                        writer.writerow([fp, x, y, thetas, '', '', mtype])

                    # Write bifurcations
                    for idx, curr_minutiae in enumerate(minutiae['Bifurcations']):
                        x = int(curr_minutiae.locX)
                        y = int(curr_minutiae.locY)
                        thetas = curr_minutiae.Orientation  # assumed to be iterable of 3 angles
                        thetas = list(thetas) + [''] * (3 - len(thetas))  # pad if fewer than 3
                        mtype = 'B'
                        writer.writerow([fp, x, y] + thetas[:3] + [mtype])
                pbar.update(1)


description = 'Programul extrage liste de micro-structuri din fiecare imagine a unei amprente digitale din directorul de intrare si le salveaza intr-un fisier CSV.'

epilog=r""" 
Fisierele sunt citite unul cate unul, caracteristicile sunt extrase si inregistrate intr-un fisier te tip csv

Fiecare amprenta genereaza doua tipuri de caracteristici: terminatii (T) si bifurcatii (B). Pentru fiecare caracteristica se inregistreaza coordonatele (x,y) si unghiul de orientare theta.:
In cazul terminatiilor exista un singur unghi, in cazul bifurcatiilor se inregistreaza trei, cate unul pentru fiecare segment al bifurcatiei. 

101_1.tif,536,180,153.434948822922,,,T
101_1.tif,259,185,45.0,-180.0,-90.0,B

"""
epilog_text = textwrap.dedent(epilog).strip()

parser = argparse.ArgumentParser(prog='create_database.py',
                                 description=description,
                                 usage='python create_database.py -id <input_directory> -ou <output csv file>',
                                 epilog=epilog_text,
                                 formatter_class=argparse.RawTextHelpFormatter )

parser.add_argument("-id", "--input_directory", required=True, help="Directorul in care se gasesc imaginile cu amprente. Nu trebuie sa aiba subdirectoare", type=str)
parser.add_argument("-ou", "--output_file", default='minutiae_db.csv', required=False, help="Numele fisierului csv in care se vor scrie rezultatele", type=str)
parser.add_argument("-sp", "--spurious_threshold", default=10, required=False, help="Filtreaza caracteristicile astfel incat sa fie eliminate cele care sunt mai apropiate de acest parametru", type=int)


args = parser.parse_args()

input_dir = args.input_directory

# Check if input directory exists
if not os.path.exists(input_dir):
    raise FileNotFoundError(f"Input directory '{input_dir}' does not exist.")

# Count total files
file_list = []
valid_extensions = {".jpg", ".jpeg", ".png", ".tif"}
for root, _, files in os.walk(input_dir):
    for file in files:
        if os.path.splitext(file)[1].lower() in valid_extensions:
            file_list.append(os.path.join(root, file))

total_files = len(file_list)
print(f"Total files to process: {total_files}")


minutiae_db = {}

# Process files with progress bar
with tqdm(total=total_files, desc="Processing files", unit="file") as pbar:
    for file_path in file_list:
        pbar.set_description(f"Processing: {file_path}") 
        label = os.path.basename(file_path)
        try:
            minutiae = extract_minutiae(file_path, args.spurious_threshold)
        except Exception as e:
            minutiae = None

        if minutiae is not None:
            if label not in minutiae_db:
                minutiae_db[label] = []  # Initialize an empty list for the label if it doesn't exist

            minutiae_db[label].append(minutiae) 
        pbar.update(1)

print("Data collection complete")

write_minutiae_to_csv(minutiae_db, args.output_file)

