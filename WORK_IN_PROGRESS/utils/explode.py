from PIL import Image
import argparse
import os
from pathlib import Path


def explode_image(path, n=4, spacing=10, output="exploded.jpg"):
    img = Image.open(path)
    w, h = img.size
    tile_w, tile_h = w // n, h // n

    # New canvas size with spacing between tiles
    new_w = tile_w * n + spacing * (n - 1)
    new_h = tile_h * n + spacing * (n - 1)
    exploded = Image.new('RGB', (new_w, new_h), color='white')

    # Paste tiles into new canvas with spacing
    for row in range(n):
        for col in range(n):
            left = col * tile_w
            upper = row * tile_h
            box = (left, upper, left + tile_w, upper + tile_h)
            tile = img.crop(box)

            new_left = col * (tile_w + spacing)
            new_upper = row * (tile_h + spacing)
            exploded.paste(tile, (new_left, new_upper))

    exploded.save(output)
    print(f"Saved to {output}")



description='Takes a JPEG image, break it into an g x g grid (like tiles), then "explode" it by spacing the tiles out to form a larger canvas with gaps between them. The result will look like the image was "blown apart" into pieces.'

parser = argparse.ArgumentParser(prog='explode_image.py',description=description)

parser.add_argument("-f", "--file", required=True, help="Image to be exploded", type=str)
parser.add_argument("-g", "--grid_size", default=7, help="Grid size", type=int)
parser.add_argument("-s", "--spacing", default = 20, help="Grid spacing after explosion", type=int)

args = parser.parse_args()

if not os.path.isfile(args.file):
    print(f"Error: Input image '{args.file}' does not exist.")
    exit(1)

file_name, file_ext = os.path.splitext(os.path.basename(args.file))  # Split filename and extension
output = f"{file_name}_exploded{file_ext}"  # Add the suffix and keep the or


explode_image(args.file, n=args.grid_size, spacing=args.spacing, output=output)
