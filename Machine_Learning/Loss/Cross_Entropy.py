import torch
import torch.nn as nn
import torch.functional as F

class CrossEntropyLoss(torch.nn.Module):
    def __init__(self, reduction='mean'):
        super(CrossEntropyLoss, self).__init__()
        self.reduction = reduction
    
    def forward(self,logits,target):
        # logits: [N, C, H, W], target: [N, H, W]
        # loss = sum(-ya_i * log(c_i))
        if logits.dim()>2:
            logits=logits.view(logits.size(0), logits.size(1), -1)   # [N, C, HW]
            logits=logits.transpose(1,2) # [N, HW, C]
            logits=logits.contiguous.view(-1,logits.size(2))  # [NHW, C]
        
        target=target.view(-1,1)  # [NHW，1]
        
        logits=F.log_softmax(logits,1) #归一化
        logits=logits.gather(1,target)  # [NHW, 1]
        
        loss=-1*logits
        
        if self.reduction=='mean':
            loss=loss.mean()
        elif self.reduction=='sum':
            loss=loss.sum()
        
        return loss        
        
        
        
        
            
            