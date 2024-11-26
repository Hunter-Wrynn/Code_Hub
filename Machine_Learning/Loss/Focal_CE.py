# https://github.com/clcarwin/focal_loss_pytorch/blob/master/focalloss.py

import torch
import torch.nn as nn
import torch.functional as F

class CrossEntropyFocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=0.2, reduction='mean'):
        super(CrossEntropyFocalLoss, self).__init__()
        self.reduction = reduction
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self,logits,target):
        # logits: [N, C, H, W], target: [N, H, W]
        # loss = sum(-y_i * log(c_i))
        if logits.dim()>2:
            logits=logits.view(logits.size(0),logits.size(1),-1) # [N, C, HW]
            logits=logits.transpose(1,2) # [N, HW, C]
            logits=logits.contiguous.view(-1,logits.size(2)) # [NHW, C]
        
        target=target.view(-1,1) # [NHW，1]
        
        logits=F.softmax(logits,1) #归一化
        logits=logits.gather(1,target).view(-1)  # [NHW]
        
        log_gt = torch.log(logits)
        
        if self.alpha is not None:
            # alpha: [C]
            alpha=self.alpha.gather(0,target.view(-1)) # [NHW]
            log_gt=log_gt*alpha
        
        loss=-1*(1-logits) ** self.gamma * log_gt
        
        if self.reduction=='mean':
            loss=loss.mean()
        elif self.reduction=='sum':
            loss=loss.sum()
        
        return loss
            
            