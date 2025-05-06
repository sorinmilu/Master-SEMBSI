import torch
import argparse
import torchvision.transforms as transforms
from PIL import Image
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from cnn_class import CNNClassifier

# Define transformations (must match those used in training)
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def print_penultimate_fc_layer(model, test_images, fc_layer_size):
    """
    Prints the values of the penultimate fully connected layer during inference.

    Args:
        model: The trained PyTorch model.
        test_images: A batch of input images (tensor).
        fc_layer_size: The size of the penultimate fully connected layer.
    """
    penultimate_layer_output = None

    # Define a forward hook to capture the output of the penultimate layer
    def hook(module, input, output):
        nonlocal penultimate_layer_output
        penultimate_layer_output = output

    # Register the hook on the penultimate fully connected layer
    for layer in model.modules():
        if isinstance(layer, torch.nn.Linear) and layer.out_features == fc_layer_size:
            layer.register_forward_hook(hook)
            break
    else:
        print(f"Fully connected layer with size {fc_layer_size} not found in the model.")
        return

    # Perform inference
    with torch.no_grad():
        _ = model(test_images)  # Forward pass to trigger the hook

    # Print the captured output
    if penultimate_layer_output is not None:
        print(f"Values of the penultimate fully connected layer (size {fc_layer_size}):")
        print(penultimate_layer_output.cpu().numpy())
    else:
        print("Failed to capture output for the penultimate fully connected layer.")

def load_images_from_folder(folder, prefix):
    images = []
    labels = []
    filenames = []
    
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        
        # Ensure it's an image file
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            img = Image.open(img_path)
            
            # Check if the image has at least 3 channels (RGB)
            if img.mode != "RGB":
                print(f"Skipping {filename}: Image has {len(img.getbands())} channels (expected at least 3).")
                continue
            
            img = img.convert("RGB")  # Convert to grayscale
            img = transform(img)
            # img = transform(img).unsqueeze(0)             
            images.append(img)
            labels.append(1 if prefix in filename.lower() else 0)  # 1 for faces, 0 for non-faces
            filenames.append(filename)  # Keep track of filenames
    
    return images, labels, filenames



description = 'Inference program for CNN classifier'

parser = argparse.ArgumentParser(prog='infer_dir.py',
                                 description=description,
                                 usage='infer_dir.py -m <model file> -df <images directory>',
                                 epilog="This program gets a directory with jpeg images as an argument, runs the model on each image and prints false positives, false negatives and detection statistics")


parser = argparse.ArgumentParser(description="Face detection script with drawing and printing options.")
parser.add_argument('-m', "--model", default="models/face_model.pth", help="Path to input image")
parser.add_argument('-p', "--positive_prefix", default="face", help="the part found in the names of the positive files")
parser.add_argument('-e', "--embedding_size", default=512, help="size of the last fully connected layer", type=int)
parser.add_argument('-df', "--data_folder", default="test", help="Path to test data")

args = parser.parse_args()


# Load test data

test_images, test_labels, test_filenames = load_images_from_folder(args.data_folder, args.positive_prefix)

# Convert lists to tensors
test_images = torch.stack(test_images)
test_labels = torch.tensor(test_labels, dtype=torch.float32)

# Load trained model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CNNClassifier(embedding_size=args.embedding_size).to(DEVICE)  # Initialize the model
model.load_state_dict(torch.load(args.model, weights_only=True, map_location=torch.device('cpu'))) 
model.eval()  # Set model to evaluation mode

# Perform inference
with torch.no_grad():
    outputs = model(test_images)
    predictions = torch.sigmoid(outputs).squeeze() 
    predicted_labels = (predictions > 0.6).float() 
    
    # for i, (filename, prediction, label) in enumerate(zip(test_filenames, predictions, predicted_labels)):
    #     # Print the image name and predicted probability
    #     print(f"{filename}: Predicted Probability = {prediction.item():.4f} label {label}")


false_positives = []
false_negatives = []

for i in range(len(test_labels)):
    if predicted_labels[i].item() == 1 and test_labels[i].item() == 0:
        # False positive: predicted 1, actual 0
        false_positives.append((test_filenames[i], outputs[i].item()))  # Store filename and prediction
    elif predicted_labels[i].item() == 0 and test_labels[i].item() == 1:
        # False negative: predicted 0, actual 1
         false_negatives.append((test_filenames[i], outputs[i].item()))  # Store filename and prediction

print("---false positives---")
for filename, prediction in false_positives:
    print(f"{filename}: Predicted Probability = {prediction:.4f}")

print("---false negatives---")
for filename, prediction in false_negatives:
    print(f"{filename}: Predicted Probability = {prediction:.4f}")

# Convert tensors to numpy arrays for evaluation
test_labels_np = test_labels.numpy()
predicted_labels_np = predicted_labels.numpy()

# Compute performance metrics
accuracy = accuracy_score(test_labels_np, predicted_labels_np)
precision = precision_score(test_labels_np, predicted_labels_np)
recall = recall_score(test_labels_np, predicted_labels_np)
f1 = f1_score(test_labels_np, predicted_labels_np)

# Print evaluation results
print("\nModel Evaluation:")
print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")
