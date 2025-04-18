import torch
import torch.nn as nn
import torch.nn.functional as F
from .Attention import Attention
from .DropPath import DropPath
from .MLP import Mlp

class Block(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_ratio=0.,
                 attn_drop_ratio=0.,
                 drop_path_ratio=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super(Block,self).__init__()
        self.norm1=norm_layer(dim)
        self.attn=Attention(dim,
                            num_heads=num_heads,
                            qkv_bias=qkv_bias,
                            qk_scale=qk_scale,
                            attn_drop_ratio=attn_drop_ratio,
                            proj_drop_ratio=drop_ratio)
        
        self.drop_path=DropPath(drop_path_ratio) if drop_path_ratio > 0 else nn.Identity()
        self.norm2=norm_layer(dim)
        mlp_hidden_dim=int(dim*mlp_ratio)
        self.mlp=Mlp(in_features=dim,
                     hidden_features=mlp_hidden_dim,
                     act_layer=act_layer,
                     drop=drop_ratio)

        def forward(sefl,x):
            x=x+self.drop_path(self.attn(self.norm1(x)))
            x=x+self.drop_path(self.mlp(self.norm2(x)))
            return x
        
    
        
        
