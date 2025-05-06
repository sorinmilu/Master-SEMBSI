import os
import cv2

def delete_grayscale_images(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        # Check if it's a file
        if not os.path.isfile(file_path):
            continue

        # Read the image
        image = cv2.imread(file_path)
        print(len(image.shape))
        # If the image has only one channel, delete it
        if image is not None and len(image.shape) == 2:  # Grayscale images have only 2 dimensions
            os.remove(file_path)
            print(f"Deleted: {file_path}")

# Example usage
delete_grayscale_images("data_500_2")
