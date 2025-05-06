from torchinfo import summary
import torch
import torch.nn as nn
from gridmodel import GridCNNModel, GridSimpleCNNModel



# Define your model class here
#model = GridCNNModel(griddim=7, image_size=224)  # Replace with your model
model = GridSimpleCNNModel(griddim=7, image_size=224)  # Replace with your model

summary(model, input_size=(1, 3, 224, 224))

