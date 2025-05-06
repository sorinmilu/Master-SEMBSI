from torchinfo import summary
import torch
import torch.nn as nn
import argparse
from cnn_class import CNNClassifier
import torch.profiler


description = 'Information about a pytorch Neural Network model'

parser = argparse.ArgumentParser(prog='model_info.py',
                                 description=description,
                                 usage='model_info.py -e <embedding_size>',
                                 epilog="This program initializes a model class and prints the summary (layers and sizes) of that model.")


parser = argparse.ArgumentParser(description="Face detection script with drawing and printing options.")
parser.add_argument('-e', "--embedding_size", default=512, help="size of the last fully connected layer", type=int)

args = parser.parse_args()

model = CNNClassifier(embedding_size=args.embedding_size)  # Replace with your model

summary(model, input_size=(1, 3, 128, 128))

# for name, param in model.named_parameters():
#     print(name, param.shape, "requires_grad=", param.requires_grad)

# total_params = sum(p.numel() for p in model.parameters())
# print(f"Total parameters: {total_params}")

# trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
# print(f"Trainable parameters: {trainable_params}")

# with torch.profiler.profile(
#     activities=[torch.profiler.ProfilerActivity.CPU],
#     record_shapes=True,
#     with_stack=True
# ) as prof:
#     dummy_input = torch.randn(1, 3, 128, 128)
#     model(dummy_input)

# print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

