import argparse
import cv2
from fingerprint_enhancer import enhance_fingerprint
import fingerprint_feature_extractor
import numpy as np

description = 'Programul încarcă o imagine și extrage micro-structurile din aceasta. Produce o listă a acestora (terminații și bifurcații)'

parser = argparse.ArgumentParser(prog='extract_image.py.py',
                                 description=description,
                                 usage='extract_image.py -f <input_file> -sp <spurious threeshold>',
                                 formatter_class=argparse.RawTextHelpFormatter )

parser.add_argument("-f", "--input_file", required=True, help="Input image file", type=str)
parser.add_argument("-sp", "--spurious_threshold", default=10, required=False, help="Filtreaza caracteristicile astfel incat sa fie eliminate cele care sunt mai apropiate de acest parametru", type=int)


args = parser.parse_args()

print(f"Loading image: {args.input_file}")
img = cv2.imread(args.input_file, cv2.IMREAD_GRAYSCALE)
if img is None:
    print(f"Error: Could not load image from {args.input_file}")
    
    
fp_enhanced_image = enhance_fingerprint(img)
fp_enhanced_image = (fp_enhanced_image * 255).astype(np.uint8)

FeaturesTerminations, FeaturesBifurcations = fingerprint_feature_extractor.extract_minutiae_features(fp_enhanced_image, spuriousMinutiaeThresh=args.spurious_threshold, invertImage=False, showResult=False, saveResult=False)


for idx, curr_minutiae in enumerate(FeaturesTerminations):
    row, col, Orientation, Type = curr_minutiae.locX, curr_minutiae.locY, curr_minutiae.Orientation, curr_minutiae.Type
    print(f"Termination {idx}: ({row}, {col} {Orientation} {Type})")    
    
for idx, curr_minutiae in enumerate(FeaturesBifurcations):
    row, col, Orientation, Type = curr_minutiae.locX, curr_minutiae.locY, curr_minutiae.Orientation, curr_minutiae.Type
    print(f"Bifurcation {idx}: ({row}, {col} {Orientation} {Type})")    




