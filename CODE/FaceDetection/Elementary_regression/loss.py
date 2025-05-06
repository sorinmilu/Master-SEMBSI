import torch
import torch.nn as nn
import math

class YOLOFaceLoss(nn.Module):
    def __init__(self, lambda_coord=2, lambda_noobj=0.02):
        super(YOLOFaceLoss, self).__init__()
        self.lambda_coord = lambda_coord  # Weight for face presence
        self.lambda_noobj = lambda_noobj  # Weight for no-face cells
        self.mse = nn.MSELoss(reduction='sum')  # Mean Squared Error loss (sum for all grid cells)
        self.bce = nn.BCELoss(reduction='sum')  # Binary Cross-Entropy Loss

    def forward(self, pred, target):
        """
        pred: (batch, grid_size, grid_size, 5) - Model predictions
        target: (batch, grid_size, grid_size, 5) - Ground truth (x, y, w, h, confidence)
        """
        # Masks for face and no-face cells
        obj_mask = target[..., 0] > 0  # Cells where a face exists (probability is in the first position)
        no_obj_mask = target[..., 0] == 0  # Cells without a face (probability is in the first position)

        # Coordinate Loss (x, y, w, h)
        coord_loss = 0.0
        if obj_mask.any():  # Only apply loss if there are faces
            coord_loss = self.lambda_coord * self.mse(pred[obj_mask][..., 1:5], target[obj_mask][..., 1:5])

        # Confidence Score Loss (Object cells)
        obj_loss = 0.0
        if obj_mask.any():  # Only apply loss for cells containing faces
            obj_loss = self.bce(pred[obj_mask][..., 0], target[obj_mask][..., 0])

        # Confidence Score Loss (No object cells)
        no_obj_loss = 0.0
        if no_obj_mask.any():  # Only apply loss for cells containing no faces
            no_obj_loss = self.lambda_noobj * self.bce(pred[no_obj_mask][..., 0], target[no_obj_mask][..., 0])

        # Check for NaN values in loss terms
        if math.isnan(coord_loss) or math.isnan(obj_loss) or math.isnan(no_obj_loss):
            print(f"NaN detected in loss! coord_loss: {coord_loss}, obj_loss: {obj_loss}, no_obj_loss: {no_obj_loss}")

        # Return the total loss
        # print(f"coord_loss: {coord_loss}  obj_loss: {obj_loss} no_obj_loss: {no_obj_loss}")
        total_loss = coord_loss + obj_loss + no_obj_loss

        return total_loss, coord_loss, obj_loss, no_obj_loss