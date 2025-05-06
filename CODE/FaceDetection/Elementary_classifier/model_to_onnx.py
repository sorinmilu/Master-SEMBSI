from torchinfo import summary
import torch
import torch.nn as nn
import argparse
from cnn_class import CNNClassifier
from pathlib import Path
import os

description = 'Information about a pytorch Neural Network model'

parser = argparse.ArgumentParser(prog='model_info.py',
                                 description=description,
                                 usage='model_info.py -e <embedding_size>',
                                 epilog="This program initializes a model class and prints the summary (layers and sizes) of that model.")


parser = argparse.ArgumentParser(description="Face detection script with drawing and printing options.")
parser.add_argument('-e', "--embedding_size", default=512, help="size of the last fully connected layer", type=int)
parser.add_argument('-m', "--model", required=True, help="Path to input image")

args = parser.parse_args()

if not os.path.isfile(args.model):
    print(f"Error: Input image '{args.mode}' does not exist.")
    exit(1)

file_path = Path(args.model)
file_no_ext = file_path.with_suffix('')
onnx_path = file_no_ext.with_suffix(".onnx")

print(onnx_path)

model = CNNClassifier(embedding_size=args.embedding_size)  # Replace with your model


model.load_state_dict(torch.load(args.model, weights_only=True, map_location='cpu'))

# 3. Set to evaluation mode
model.eval()

# 4. Create dummy input (must match your training input size)
dummy_input = torch.randn(1, 3, 128, 128)


torch.onnx.export(
    model,                     # your model
    dummy_input,               # dummy input with correct shape
    onnx_path,                 # filename to export
    export_params=True,        # include weights
    opset_version=11,          # ONNX version (Zetane supports 10+)
    do_constant_folding=True,  # optimization
    input_names=['input'],     # input name
    output_names=['output'],   # output name
    dynamic_axes={             # for batch size flexibility (optional)
        'input': {0: 'batch_size'},
        'output': {0: 'batch_size'}
    }
)