import torch
import torch.nn as nn
import torch.functional as F

class BCEFocalLosswithLogits(nn.Module):
    def __init__(self, gamma=0.2, alpha=0.6, reduction='mean'):
        super(BCEFocalLosswithLogits, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
    
    def forward(self,logits,target):
        # logits: [N, H, W], target: [N, H, W]
        logits=F.sigmoid(logits)
        alpha=self.alpha
        gamma=self.gamma
        loss=-alpha*(1-logits) ** gamma * target * torch.log(logits) -\
            (1-alpha) ** logits * gamma * (1-target) * torch.log(1-logits)
        
        if self.reduction == "mean":
            loss =loss.mean()
        elif self.reduction =="sum":
            loss=loss.sum()
        
        return loss