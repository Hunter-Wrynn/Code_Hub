import torch.nn as nn
import torch.nn.attention
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F

    
class Lora(nn.Module):
    def __init__(self, in_features, out_features, lora_alpha, lora_dropout):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.weight = nn.Parameter(torch.zeros(out_features, in_features))
        self.lora_A = nn.Linear(in_features, out_features)
        
class LoraLinear(nn.Module):
    def __init__(self,in_features,out_features,merge,rank=16,lora_alpha=16,drouout_rate=0.5):
        super().__init__(LoraLinear,self)
        self.in_features=in_features
        self.out_features=out_features
        self.merge=merge
        self.rank=rank
        self.lora_alpha=lora_alpha
        self.droupout_rate=drouout_rate
        self.linear=nn.Linear(in_features,out_features)
        if rank>0:
            self.lora_b=nn.Parameter(torch.zeros(out_features,rank))
            self.lora_a=nn.Parameter(torch.zeros(rank,in_features))
            self.scale=self.lora_alpha/self.rank    
            self.linear.weight.requires_grad=False
        if self.droupout_rate>0:
            self.droupout=nn.Dropout(self.droupout_rate)
        else:
            self.droupout=nn.Identity()
        
        self.initial_weights()
    def initial_weights(self):
        nn.init.zeros_(self.lora_b)


    def forward(self,x):
        if self.rank>0 and self.merge:
            output=F.linear(x,self.linear.weight+self.lora_b@self.lora_a*self.scale,self.linear.bias)
            output=self.droupout(output)
             
            return output
         
