from torchinfo import summary
import torch
import torch.nn as nn
from cnn_yolo_face import YOLOFaceCNN



# Define your model class here
#model = GridCNNModel(griddim=7, image_size=224)  # Replace with your model
model = YOLOFaceCNN(grid_size=7)  # Replace with your model

summary(model, input_size=(1, 3, 224, 224))

