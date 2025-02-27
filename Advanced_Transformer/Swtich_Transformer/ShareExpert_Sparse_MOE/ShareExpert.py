from ..Sparse_MOE.Sparse_MOE import SparseMOE
from ..Sparse_MOE.Sparse_MOE import MOEConfig
from ..Basic_MOE.Basic_MOE import BasicExpert

import torch
import torch.nn as nn
import torch.nn.functional as F

class ShareExpertMOE(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.moe_model = SparseMOE(config)
        self.shared_experts = nn.ModuleList(
            [
                BasicExpert(
                    config.hidden_dim, config.hidden_dim
                ) for _ in range(config.shared_experts_number)
            ]
        )

    def forward(self, x):
        # x shape 是 (b, s, hidden_dim)
        # 首先过 moe 模型
        sparse_moe_out, router_logits = self.moe_model(x)
        
        # 针对的还是 x 的每一个 
        # 然后过 shared experts
        shared_experts_out = [
            expert(x) for expert in self.shared_experts
        ] # 每一个 expert 的输出 shape 是 (b, s, hidden_dim)
        
        shared_experts_out = torch.stack(
            shared_experts_out, dim=0
        ).sum(dim=0)
        
        # 把 sparse_moe_out 和 shared_experts_out 加起来
        return sparse_moe_out + shared_experts_out, router_logits


def test_share_expert_moe():
    x = torch.rand(2, 4, 16)
    config = MOEConfig(16, 2, 2)
    share_expert_moe = ShareExpertMOE(config)
    out = share_expert_moe(x)
    print(out[0].shape, out[1].shape)


test_share_expert_moe()