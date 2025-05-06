import os
from collections import defaultdict
from PIL import Image, ImageDraw, ImageFont
import argparse


description = 'Accumulates photos from similar directories and puts them on the same image'

epilog_text = f"""
    This program assumes that there are a set of directories with identical structure that are the result of successive inference operations
    such as models with various parameters. The scope of this program is to take all the pictures that are the result of inference operations on the same input and align them
    in the same images to be compared. 

    The directory structure has to be identical for all the subdirectories that will match the prefix. 
    
    For example, assuming that the prefix is results_images

├── results_images_07
    ├── 3+_faces
    ├── closeups
    ├── one_face
├── results_images_14
    ├── 3+_faces
    ├── closeups
    ├── one_face
├── results_images_21
    ├── 3+_faces
    ├── closeups
    ├── one_face
"""

parser = argparse.ArgumentParser(prog='accumulate_result_images.py',
                                 description=description,
                                 usage='accumulate_result_images.py -pf <directory prefix> -od <output_directory>' ,
                                 epilog="This program gets a directory with jpeg images as an argument, runs the model on each image replicates the input folder structure on the output folder. The output folder has to be empty")

parser.add_argument("-pf", "--prefix", required=True, help="Prefix of the directories", type=str)
parser.add_argument("-od", "--output_directory", required=False, help="Directory for the output combined images", type=str, default="output")

args = parser.parse_args()

# Define the base directory pattern
base_prefix = args.prefix

# Dictionary to store file paths
file_dict = defaultdict(list)

filter_substrings = ['grid']
allowed_extensions = {".png", ".jpg", ".jpeg"}

output_dir = args.output_directory

os.makedirs(output_dir, exist_ok=True)

# Get all directories matching the pattern
for root, dirs, files in os.walk("."):
    print(root)
    print(os.path.basename(root))
    if os.path.basename(root).startswith(base_prefix):
        print("ccc")
        for subdir in dirs:
            subdir_path = os.path.join(root, subdir)
            for file in os.listdir(subdir_path):
                file_path = os.path.join(subdir_path, file)
                if os.path.isfile(file_path) and os.path.splitext(file)[1].lower() in allowed_extensions:
                    file_dict[file].append(file_path)

# Print the result
for file_name, paths in file_dict.items():

    if any(substring in file_name for substring in filter_substrings):
        paths.sort() 
        images = [Image.open(path) for path in paths]
        max_height = max(img.height for img in images if img.width <= 1000)
        total_width = sum(img.width for img in images if img.width <= 1000)
        
        final_image = Image.new("RGB", (total_width, max_height + 50), "white")
        draw = ImageDraw.Draw(final_image)
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", 30)
        except:
            print('font error')    
            font = ImageFont.load_default()
        
        x_offset = 0
        for img, path in zip(images, paths):
            if img.width <= 1000:
                final_image.paste(img, (x_offset, 0))
                text_width, _ = draw.textbbox((0, 0), os.path.dirname(path), font=font)[2:4]
                text_x = x_offset + (img.width - text_width) // 2
                draw.text((text_x, max_height + 5), os.path.dirname(path), fill="black", font=font)
                x_offset += img.width

        
        output_path = os.path.join(output_dir, f"combined_{file_name}.jpg")
        final_image.save(output_path)
        print(f"Saved combined image as {output_path}")
