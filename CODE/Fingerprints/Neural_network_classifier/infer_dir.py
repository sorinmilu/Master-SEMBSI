import os
import torch
import argparse
from PIL import Image
from torchvision import transforms
from network import FingerprintClassifier

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

def classify_images_in_directory(test_dir, model_path="fingerprint_classifier.pth"):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    label_map = {
        0: 'class1_Arc',
        1: 'class2_Whorl',
        2: 'class3_Loop'
    }

    model = FingerprintClassifier().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    # Collect image paths
    image_files = [
        os.path.join(test_dir, f)
        for f in os.listdir(test_dir)
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif'))
    ]

    if not image_files:
        print("No images found in the directory.")
        return

    # Run classification
    for img_path in sorted(image_files):
        image = Image.open(img_path).convert('L')
        image = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(image)
            predicted_class = output.argmax(dim=1).item()
            print(f"{os.path.basename(img_path)} => {label_map[predicted_class]}")




parser = argparse.ArgumentParser(prog='infer_dir.py',
                                 description="Classification script for a neural network fingerprints classifier. It divides the input images into three classes: Arc, Whorl and Loop",
                                 usage='infer_dir.py -m <model file> -df <images directory>',
                                 epilog="This program gets a directory with jpeg images as an argument, runs the model on each image and prints the class of fingerprints found in the images")


parser.add_argument('-m', "--model_file", default="models/face_model.pth", help="Path to input image")
parser.add_argument('-id', "--input_directory", default="test", help="Path to test data")

args = parser.parse_args()

if not os.path.exists(args.data_folder):
    raise FileNotFoundError(f"Input directory '{args.data_folder}' does not exist.")

if not os.path.exists(args.model):
    raise FileNotFoundError(f"Input directory '{args.model}' does not exist.")

classify_images_in_directory(args.data_folder, model_path=args.model)
