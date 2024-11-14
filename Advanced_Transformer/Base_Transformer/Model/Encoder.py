import torch.nn as nn
import torch
import math
from .FFN import PositinalWiseFeedForward
from .Layer_Norm import LayerNorm
from .Multihead_Attention import MultiheadAttention
from .Positional_Embedding import PositionalEmbedding
from .Token_Embedding import TokenEmbedding
from .Transformer_Embedding import TransformerEmbedding

class EncoderLayer(nn.Module):
    def __init__(self,d_model,ffn_hidden,n_head,drop_prob):
        super(EncoderLayer,self).__init__()
        self.attention=MultiheadAttention(d_model,n_head)
        self.norm1=LayerNorm(d_model)
        self.drop1=nn.Dropout(drop_prob)
        
        self.ffn=PositinalWiseFeedForward(d_model,ffn_hidden,drop_prob)
        self.norm2=LayerNorm(d_model)
        self.drop2=nn.Dropout(drop_prob)
        
    def forward(self,x,Mask=None):
        _x=x
        x=self.attention(x,x,x,Mask)
        
        x=self.drop1(x)
        x=self.norm1(x+_x)
        
        _x=x
        x=self.ffn(x)
        
        x=self.drop2(x)
        x=self.norm2(x+_x)
        
        return x
    
class Encoder(nn.Module):
    def __init__(
        self,
        env_voc_size,
        max_len,
        d_model,
        ffn_hidden,
        n_head,
        n_layer,
        drop_prob,
        device,
    ):
        super(Encoder, self).__init__()
        
        self.embedding=TransformerEmbedding(
            env_voc_size,d_model,max_len,drop_prob,device
        )
        
        self.layers=nn.ModuleList(
            [
                EncoderLayer(d_model,ffn_hidden,n_head,drop_prob)
                for _ in range(n_layer)
            ]
        )
        
    def forward(self,x,s_Mask):
        x=self.embedding(x)
        for layer in self.layers:
            x=layer(x,s_Mask)
            
        return x
            