import json
import matplotlib.pyplot as plt
import os
import argparse
import sys
import textwrap

# Load JSON data

def format_duration(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours}h {minutes}m {seconds}s"

# this fragment is used in the help message
json_fragment = {
    "training_parameters": {
        "learning_rate": 0.0001,
        "epochs": 100,
        "batch_size": 32,
        "model_name_root": "grid_model"
    },
    "epochs": [
        {
            "epoch": 0,
            "batches": 135,
            "duration": 71.89708399772644,
            "avg_total_loss": 291.0997297498915
        },
		{
            "epoch": 1,
            "batches": 135,
            "duration": 71.89708399772644,
            "avg_total_loss": 291.0997297498915
        }
	]
}

formatted_json = json.dumps(json_fragment, indent=4)
epilog_text = f"""
    The json has to have two sections: one called \"training_parameters\" with keys-values (no arrays) and another sections called \"epoches\"
    Example JSON configuration:

    {formatted_json}
"""


description = 'This program will plot a loss chart from a training json'

parser = argparse.ArgumentParser(prog='plotter.py',
                                 description=description,
                                 usage='plotter.py -f <input json file>',
                                 epilog=epilog_text,
                                 formatter_class=argparse.RawTextHelpFormatter )
parser.add_argument("-f", "--file", required=True, help="The json file path", type=str)
parser.add_argument("-o", "--output_image", help="The output image name (if not given then the image will be saved in the same directory as the json file with the same name) ", type=str)


args = parser.parse_args()

if args.file:
    CHECK_FILE = os.path.isfile(args.file)
    if not CHECK_FILE:
        print('Input file not found')
        sys.exit(1)

if not args.output_image:
    img_filename = os.path.splitext(args.file)[0] + ".png"
else:
    img_filename = args.output_image


with open(args.file, "r") as f:
    data = json.load(f)

# Extract training parameters
training_params = data["training_parameters"]

# Extract epochs and losses
epochs = [entry["epoch"] for entry in data["epochs"]]
avg_losses = [entry["avg_total_loss"] for entry in data["epochs"]]
total_training_time = sum(entry["duration"] for entry in data["epochs"])
min_loss_entry = min(data["epochs"], key=lambda x: x["avg_total_loss"])
last_epoch_entry = data["epochs"][-1]

training_params["smallest_loss"] = f"minimum loss: {min_loss_entry['avg_total_loss']:.5f} (epoch:  {min_loss_entry['epoch']})"
training_params["last_epoch_loss"] = f"{last_epoch_entry["avg_total_loss"]:.5f}"
training_params["total_training_time"] = format_duration(total_training_time)


# Prepare training parameters for display
param_text = "\n".join([f"{key}: {value}" for key, value in training_params.items()])
param_text += f"\nTotal Training Time: {total_training_time:.2f} sec"

# Plot
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(epochs, avg_losses, marker='o', markersize=3, linestyle='-', linewidth=0.8, color='b', label='Avg Loss')

ax.set_xlabel("Epoch")
ax.set_ylabel("Avg Total Loss")
ax.set_title("Training Loss Over Epochs")
ax.legend()
ax.grid(True)

# Add training parameters as a separate column
plt.figtext(0.95, 0.5, param_text, fontsize=10, verticalalignment='center', bbox=dict(facecolor='lightgrey', alpha=0.5))

plt.savefig(img_filename, bbox_inches="tight", dpi=300)

print(f"Chart saved as {img_filename}")