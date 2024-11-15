import torch
import torch.nn as nn

def batch_norm(x,eps=1e-5):
    alpha=nn.Parameter(torch.ones(x.shape[1]))
    beta=nn.Parameter(torch.zeros(x.shape[1]))
    mean=torch.mean(x,dim=(0,2,3),keepdim=True)
    var=torch.var(x,dim=(0,2,3),keepdim=True,unbiased=False)
    output=(x-mean)/torch.sqrt(var+eps)
    output=output*alpha.view(1,-1,1,1)+beta.view(1,-1,1,1)
    
    return output



class BatchNorm(nn.Module):
    def __init__(self,num_features,eps=1-5,momentum=0.1):
        super(BatchNorm, self).__init__()
        
        self.gamma=nn.Parameter(torch.ones(num_features))
        self.beta=nn.Parameter(torch.zeros(num_features))
        
        self.running_mean=torch.zeros(num_features)
        self.running_var=torch.ones(num_features)
        
        self.eps=eps
        self.momentum=momentum
        
    def forward(self, x):
        batch_mean=x.mean(dim=0,keepdim=True)
        batch_var=x.var(dim=0,keepdim=True,unbias=False)
        
        x_normalized=(x-batch_mean)/torch.sqrt(batch_var+self.eps)
        
        out=self.gamma*x_normalized+self.beta
        
        if self.training:
            self.running_mean=(1-self.momentum)*self.running_mean+self.momentum*self.batch_mean
            self.running_var=(1-self.momentum)*self.running_var+self.momentum*self.batch_var
        
        else:
            x_normalized=(x-self.running_mean)*torch.sqrt(self.running_var+self.eps)
            out=self.gamma*x_normalized+self.beta
            
        return out

class SimpleNet(nn.Module):
    def __iniy__(self):
        super(SimpleNet, self).__init__()
        self.conv1=nn.Conv2d(3,64,kernel_size=3,padding=1)
        self.bn1=BatchNorm(64)
        self.relu=nn.ReLU()
    
    def forward(self, x):
        x=self.conv1(x)
        x=self.bn1(x)
        x=self.relu(x)
    
        return x


def main():
    model=SimpleNet()
    input_tensor=torch.randn(16,3,64,64)
    output=model(input_tensor)
    print(output.shape)

        
        