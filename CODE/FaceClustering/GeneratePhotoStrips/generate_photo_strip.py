

import textwrap
import argparse
import os
import sys
import random
from PIL import Image, ImageDraw, ImageFont



def generate_contact_sheets(base_dir, output_dir, img_width=100, images_per_strip=10):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for subdir in sorted(os.listdir(base_dir)):
        subdir_path = os.path.join(base_dir, subdir)
        if not os.path.isdir(subdir_path):
            continue

        # Get image files
        image_files = [
            os.path.join(subdir_path, f) for f in os.listdir(subdir_path)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))
        ]

        if not image_files:
            continue

        # Choose 9 images to leave space for label thumbnail
        selected_images = random.sample(image_files, min(len(image_files), images_per_strip - 1))
        resized_images = []

        # Create label thumbnail (first slot)
        img_height = img_width  # Square label thumbnail
        label_img = Image.new('RGB', (img_width, img_height), color='white')
        draw = ImageDraw.Draw(label_img)
        try:
            font = ImageFont.truetype("arial.ttf", 14)
        except IOError:
            font = ImageFont.load_default()

        # Wrap text if it's too long (optional)
        label_lines = [subdir[i:i+10] for i in range(0, len(subdir), 10)]  # crude wrap
        y_offset = (img_height - len(label_lines) * 14) // 2
        for line in label_lines:
            bbox = draw.textbbox((0, 0), line, font=font)
            text_width = bbox[2] - bbox[0]
            draw.text(((img_width - text_width) / 2, y_offset), line, fill='black', font=font)
            y_offset += 14

        resized_images.append(label_img)

        for img_path in selected_images:
            img = Image.open(img_path).convert('RGB')
            aspect_ratio = img.height / img.width
            img_resized = img.resize((img_width, int(img_width * aspect_ratio)))
            # pad to match label height if needed
            padded_img = Image.new('RGB', (img_width, img_height), color='white')
            y_offset = (img_height - img_resized.height) // 2
            padded_img.paste(img_resized, (0, y_offset))
            resized_images.append(padded_img)

        # Concatenate thumbnails into one row
        total_width = img_width * len(resized_images)
        final_strip = Image.new('RGB', (total_width, img_height), color='white')
        x_offset = 0
        for thumb in resized_images:
            final_strip.paste(thumb, (x_offset, 0))
            x_offset += img_width

        output_path = os.path.join(output_dir, f"{subdir}.jpg")
        final_strip.save(output_path)
        print(f"Saved: {output_path}")


description = 'Program that generates photo strips from a directory with subdirectories with images. ' \

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
                                 usage='infer_folder.py -id <input_directory> -m <model file> -e <embedding size>',
                                 epilog=epilog_text,
                                 formatter_class=argparse.RawTextHelpFormatter )

parser.add_argument("-id", "--input_directory", required=True, help="Directory with the testing images. Can have subdiorectories", type=str)
parser.add_argument("-od", "--output_directory", required=True, help="Directory where the result images will be saved", type=str)
parser.add_argument("-is", "--image_size", default=100, required=False, help="Image width for each thumbnail", type=str)
parser.add_argument('-ni', "--number_images", default=10, help="Number of images in a strip", type=int)


args = parser.parse_args()

input_dir = args.input_directory


generate_contact_sheets(input_dir, "output_images", img_width=100)