import torch
import torch.nn as nn
import torch.nn.functional as F 

class BasicExpert(nn.Module):
    def __init__(self,feature_in,feature_out):
        super().__init__()
        self.fc=nn.Linear(feature_in,feature_out)
        
    def forward(self,x):
        return self.fc(x)

class BasicMOE(nn.Module):
    def __init__(self,feature_in,feature_out,num_experts):
        super().__init__()
        self.gate=nn.Linear(feature_in,num_experts)
        
        self.experts=nn.ModuleList(
            BasicExpert(
                feature_in,feature_out
            ) for _ in range (num_experts)
        )
    
    def forward(self,x):
        
        expert_weight=self.gate(x)
        expert_out_list= [
            expert(x) for expert in self.experts
        ]
        
        expert_outputs= [
            expert_out.unsqueeze(1)
            for expert_out in expert_out_list
        ]
        # (b,1,f_o)
        expert_output = torch.cat(
            expert_outputs,
            dim=1
        )
        # (b,num,f_o)
        expert_weights = F.softmax(expert_weight,dim=1)
        
        expert_weights = expert_weights.unsqueeze(1)
        # (b,1,f_o)
        output = expert_weights @ expert_output
        return output.squeeze(1)

def main():
    x = torch.rand(4, 512)  # Batch size 4, input size 512
    basic_moe = BasicMOE(512, 128, 4)  # Initialize the model
    output = basic_moe(x)  # Pass input through the model
    print(output.shape)  # Print the output shape

if __name__ == "__main__":
    main()