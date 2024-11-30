import torch
import torch.nn as nn

class BCELosswithlogits(nn.Module):
    def __init__(self,pos_weight=1,reduction='mean'):
        super(BCELosswithlogits,self).__init__()
        self.pos_weight=pos_weight
        self.reduction=reduction
    
    def forward(self,logits,target):
        logits=F.sigmoid(logits)
        loss=-self.pos_weight*target*torch.log(logits)-(1-target)*torch.log(1-logits)
        
        if self.reduction=='mean':
            loss=loss.mean()     
        elif self.reduction=='sum':
            loss=loss.sum()    
        return loss
