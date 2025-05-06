import torch
import torch.nn as nn

def bbox_iou(box1, box2):
    """
    Compute IoU (Intersection over Union) between two sets of bounding boxes.
    box1, box2: tensors of shape (..., 4) with (x, y, w, h)
    """
    # Convert (x, y, w, h) to (x1, y1, x2, y2)
    box1_x1 = box1[..., 0] - box1[..., 2] / 2
    box1_y1 = box1[..., 1] - box1[..., 3] / 2
    box1_x2 = box1[..., 0] + box1[..., 2] / 2
    box1_y2 = box1[..., 1] + box1[..., 3] / 2

    box2_x1 = box2[..., 0] - box2[..., 2] / 2
    box2_y1 = box2[..., 1] - box2[..., 3] / 2
    box2_x2 = box2[..., 0] + box2[..., 2] / 2
    box2_y2 = box2[..., 1] + box2[..., 3] / 2

    # Compute intersection area
    inter_x1 = torch.max(box1_x1, box2_x1)
    inter_y1 = torch.max(box1_y1, box2_y1)
    inter_x2 = torch.min(box1_x2, box2_x2)
    inter_y2 = torch.min(box1_y2, box2_y2)

    inter_area = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)

    # Compute union area
    box1_area = (box1_x2 - box1_x1) * (box1_y2 - box1_y1)
    box2_area = (box2_x2 - box2_x1) * (box2_y2 - box2_y1)
    union_area = box1_area + box2_area - inter_area

    # Compute IoU
    iou = inter_area / union_area.clamp(min=1e-6)
    return iou

class YOLOFaceLoss(nn.Module):
    def __init__(self, lambda_coord=5, lambda_noobj=0.5):
        super(YOLOFaceLoss, self).__init__()
        self.lambda_coord = lambda_coord  # Weight for bounding box accuracy
        self.lambda_noobj = lambda_noobj  # Weight for no-object loss
        self.bce = nn.BCELoss(reduction='sum')  # Binary Cross-Entropy Loss

    def forward(self, pred, target):
        """
        pred: (batch, grid_size, grid_size, 5) - Model predictions
        target: (batch, grid_size, grid_size, 5) - Ground truth (x, y, w, h, confidence)
        """
        # Masks for face and no-face cells
        obj_mask = target[..., 0] > 0  # Cells where a face exists
        no_obj_mask = target[..., 0] == 0  # Cells without a face

        # IoU-based Bounding Box Loss
        iou_loss = 0.0
        # if obj_mask.any():  # Only compute IoU for cells with faces
        #     iou = bbox_iou(pred[obj_mask][..., 1:5], target[obj_mask][..., 1:5])
        #     iou_loss = self.lambda_coord * (1 - iou).sum()  # 1 - IoU as loss

        if obj_mask.any():
            iou = bbox_iou(pred[obj_mask][..., 1:5], target[obj_mask][..., 1:5])
            iou_loss = self.lambda_coord * (1 - iou).mean()  # Use mean instead of sum
            # Update target confidence score with IoU for object cells
            target[obj_mask][..., 0] = iou


        # Confidence Score Loss (Object cells)
        obj_loss = 0.0
        if obj_mask.any():  
            obj_loss = self.bce(pred[obj_mask][..., 0], target[obj_mask][..., 0])

        # Confidence Score Loss (No object cells)
        no_obj_loss = 0.0
        if no_obj_mask.any():  
            no_obj_loss = self.lambda_noobj * self.bce(pred[no_obj_mask][..., 0], target[no_obj_mask][..., 0]) / no_obj_mask.sum()

        # Total loss
        total_loss = iou_loss + obj_loss + no_obj_loss
        print(f"IOU: {iou_loss} OBJ: {obj_loss} No_obj:{no_obj_loss}")
        return total_loss, iou_loss, obj_loss, no_obj_loss

class YOLOv1Loss(nn.Module):
    def __init__(self, S=7, lambda_coord=5, lambda_noobj=0.5):
        super(YOLOv1Loss, self).__init__()
        self.S = S
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj
        self.mse = nn.MSELoss(reduction="sum")  # Mean Squared Error

    def calculate_iou(self, box1, box2):
        """
        Calculate IoU between two sets of boxes.
        box1: (batch, S, S, 4) -> [x, y, w, h]
        box2: (batch, S, S, 4) -> [x, y, w, h]
        """
        # Convert (x, y, w, h) to (x1, y1, x2, y2)
        box1_x1 = box1[..., 0] - box1[..., 2] / 2
        box1_y1 = box1[..., 1] - box1[..., 3] / 2
        box1_x2 = box1[..., 0] + box1[..., 2] / 2
        box1_y2 = box1[..., 1] + box1[..., 3] / 2

        box2_x1 = box2[..., 0] - box2[..., 2] / 2
        box2_y1 = box2[..., 1] - box2[..., 3] / 2
        box2_x2 = box2[..., 0] + box2[..., 2] / 2
        box2_y2 = box2[..., 1] + box2[..., 3] / 2

        # Calculate intersection
        inter_x1 = torch.max(box1_x1, box2_x1)
        inter_y1 = torch.max(box1_y1, box2_y1)
        inter_x2 = torch.min(box1_x2, box2_x2)
        inter_y2 = torch.min(box1_y2, box2_y2)

        inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)

        # Calculate union
        box1_area = (box1_x2 - box1_x1) * (box1_y2 - box1_y1)
        box2_area = (box2_x2 - box2_x1) * (box2_y2 - box2_y1)
        union_area = box1_area + box2_area - inter_area

        # Avoid division by zero
        iou = inter_area / (union_area + 1e-6)
        return iou

    def forward(self, pred, target):
        """
        pred: (batch, S, S, 5) - Model predictions
        target: (batch, S, S, 5) - Ground truth
        """
        
        obj_mask = target[..., 4] > 0  # Object exists in the cell
        no_obj_mask = target[..., 4] == 0  # No object in the cell

        # Expand masks to match the last dimension of pred/target
        # obj_mask = obj_mask.unsqueeze(-1)  # Shape: [batch, S, S, 1]
        # no_obj_mask = no_obj_mask.unsqueeze(-1)  # Shape: [batch, S, S, 1]

        # Localization loss (x, y, w, h)
        # Use obj_mask to index pred and target

        loc_loss = self.lambda_coord * self.mse(pred[obj_mask][..., :2], target[obj_mask][..., :2])  # (x, y)
        loc_loss += self.lambda_coord * self.mse(pred[obj_mask][..., 2:4], target[obj_mask][..., 2:4])  # (w, h)

        # IoU for confidence loss
        iou = self.calculate_iou(pred[..., :4], target[..., :4])  # IoU between predicted and target boxes
        conf_loss_obj = self.mse(iou[obj_mask.squeeze(-1)], pred[obj_mask][..., 4])  # For cells with objects
        conf_loss_noobj = self.lambda_noobj * self.mse(pred[no_obj_mask][..., 4], target[no_obj_mask][..., 4])  # For cells without objects

        # Total loss
        total_loss = loc_loss + conf_loss_obj + conf_loss_noobj
        return total_loss, loc_loss, conf_loss_obj, conf_loss_noobj