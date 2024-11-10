
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositinalWiseFeedForward(nn.Module):
    def __init__(self,d_model,hidden,drouput=0.1):
        super(PositinalWiseFeedForward,self).__init__()
        self.d_model = d_model
        self.hidden = hidden
        
        self.fc1=nn.Linear(d_model,hidden)
        self.fc2=nn.Linear(hidden,d_model)
        self.droupout=nn.Dropout(drouput)

    def forward(self,x):
        x=self.fc1(x)
        x=F.relu(x)
        x=self.droupout(x)
        x=self.fc2(x)
        return x