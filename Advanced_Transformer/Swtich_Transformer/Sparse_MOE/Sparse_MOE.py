# 和basic的区别是，MOE选择topK个专家，对topK个专家的输出加权求和，并且把输入样本变成了大模型的真实的输入
# shape(batch, seq_len, hidden_dim)
from ..Basic_MOE.Basic_MOE import BasicExpert
from ..Basic_MOE.Basic_MOE import BasicMOE
import torch
import torch.nn as nn
import torch.nn.functional as F

class MOEConfig:
    def __init__(self,
                 hidden_dim,
                 expert_number,
                 top_k,
                 shared_experts_number=2
            ):
        self.hidden_dim=hidden_dim
        self.expert_number=expert_number
        self.top_k=top_k
        self.shared_experts_number=shared_experts_number

class MOERouter(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.gate=nn.Linear(config.hidden_dim,config.expert_number)
        # 但是后面只会选top_K个专家
        
        self.expert_number=config.expert_number
        self.top_k=config.top_k
    
    def forward(self,x):
        router_logits=self.gate(x) # shape (batch*seq_len,expert_number)
        
        router_probs=F.softmax(router_logits,dim=1,dtype=torch.float)
        router_weights, selected_expert_indices= torch.topk(
            router_probs,
            self.top_k,
            dim=1
        )
        
        router_weights =router_weights/router_weights.sum(
            dim=-1,
            keepdim=True
        )
        router_weights=router_weights.to(x.dtype)
        
        expert_mask=F.one_hot(
            selected_expert_indices,
            num_classes=self.expert_number
        ) # shape (b*s, top_k, expert_number)
        
        expert_mask =expert_mask.permute(2,1,0)
        
        return router_logits, router_weights, selected_expert_indices, expert_mask
        # router_logits (b*s, num)
        # (b*s, topk)
        # (b*s, topk)
        # (num, topk, b*s)

class SparseMOE(nn.Module):
    # 稀疏 MOE 模型，这里每一个 token 都会过 topk 个专家，得到对应token 的 hidden_embeddings
    def __init__(self, config):
        super().__init__()

        self.hidden_dim = config.hidden_dim

        self.expert_number = config.expert_number
        self.top_k = config.top_k

        self.experts = nn.ModuleList(
            [
                BasicExpert(self.hidden_dim, self.hidden_dim) for _ in range(self.expert_number)
            ]
        )

        self.router = MOERouter(self.hidden_dim, self.expert_number, self.top_k)
    
    def forward(self, x):
        # x shape is (b, s, hidden_dim)
        batch_size, seq_len, hidden_dim = x.size()

        # 合并前两个维度，因为不是 Sample 维度了，而是 token 维度
        hidden_states = x.view(-1, hidden_dim) # shape is(b * s, hidden_dim)

        router_logits, router_weights, selected_experts_indices, expert_mask = self.router(hidden_states)
        # 其中 selected_experts_indices shape 是 (b * s, top_k)
        # 其中 expert_mask shape 是 (expert_number, top_k, b * s)
        
        final_hidden_states = torch.zeros(
            (batch_size * seq_len, hidden_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device
        )

        for expert_idx in range(self.expert_number):
            expert_layer = self.experts[expert_idx]
            # expert_mask[expert_idx] shape 是 (top_k, b * s)
            idx, top_x = torch.where(expert_mask[expert_idx]) 
            # idx 和 top_x 都是一维 tensor
            # idx 的值是 0 或 1, 表示这个 token 是作为当前专家的 top1 还是 top2
            # top_x 的值是 token 在 batch*seq_len 中的位置索引
            # 例如对于 batch_size=2, seq_len=4 的输入:
            # top_x 的值范围是 0-7, 表示在展平后的 8 个 token 中的位置
            # idx 的值是 0/1, 表示这个 token 把当前专家作为其 top1/top2 专家

            # hidden_states 的 shape 是 (b * s, hidden_dim)
            # 需要取到 top_x 对应的 hidden_states
            current_state = hidden_states.unsqueeze(
                0
            )[:, top_x, :].reshape(-1, hidden_dim) # （selected_token_number, hidden_dim）

            # router_weight 的 shape 是 (b * s, top_k)
            current_hidden_states = expert_layer(
                current_state
            ) * router_weights[top_x, idx].unsqueeze(-1)  # （selected_token_number, 1） 这里有广播

            # 把当前专家的输出加到 final_hidden_states 中
            # 方式1 的写法性能更好，并且方式1容易出现
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
            # 方式2
            # final_hidden_states[top_x] += current_hidden_states.to(hidden_states.dtype)
            # 方式1 的写法性能更差，并且方式1容易出现错误，+= 操作在处理重复索引时需要多次读写内存，可能会导致竞争条件

        # 把 final_hidden_states 还原到原来的 shape
        final_hidden_states = final_hidden_states.reshape(batch_size, seq_len, hidden_dim)

        return final_hidden_states, router_logits # shape 是 (b * s, expert_number)


def test_token_level_moe():
    x = torch.rand(2, 4, 16)
    config = MOEConfig(16, 2, 2)
    token_level_moe = SparseMOE(config)
    out = token_level_moe(x)
    print(out[0].shape, out[1].shape)


test_token_level_moe()
        
        
                