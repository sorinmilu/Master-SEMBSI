import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=50.0, reduction='mean'):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.reduction = reduction

    def forward(self, output1, output2, label):
        # Ensure label is float for proper math
        label = label.float()

        euclidean_distance = F.pairwise_distance(output1, output2)
        positive_loss = label * torch.pow(euclidean_distance, 2)
        negative_loss = (1 - label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        loss = 0.5 * (positive_loss + negative_loss)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss  # no reduction
