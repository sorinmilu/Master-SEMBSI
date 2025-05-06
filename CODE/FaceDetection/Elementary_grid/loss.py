import torch
import torch.nn as nn

class GridLoss(nn.Module):
    def __init__(self, lambda_obj=1.0, lambda_noobj=0.5):
        super(GridLoss, self).__init__()
        self.lambda_obj = lambda_obj
        self.lambda_noobj = lambda_noobj
        self.bce_loss = nn.BCELoss(reduction='none')

    def forward(self, pred, target):
        # Calculate object and non-object cells
        obj_mask = target == 1
        noobj_mask = target == 0

        # Calculate the objectness loss
        obj_loss = self.bce_loss(pred[obj_mask], target[obj_mask])
        
        # Calculate the non-objectness loss
        noobj_loss = self.bce_loss(pred[noobj_mask], target[noobj_mask])

        # Weighted sum of objectness and non-objectness losses
        obj_loss = obj_loss.sum()
        noobj_loss = noobj_loss.sum()

        total_loss = self.lambda_obj * obj_loss + self.lambda_noobj * noobj_loss
        return total_loss
