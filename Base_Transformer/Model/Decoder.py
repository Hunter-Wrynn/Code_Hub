import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import torch
import torch.nn as nn
import torch
import math
from ..Layers.FFN import PositinalWiseFeedForward
from ..Layers.Layer_Norm import LayerNorm
from ..Layers.Multihead_Attention import MultiheadAttention
from ..Layers.Positional_Embedding import PositionalEmbedding
from ..Layers.Token_Embedding import TokenEmbedding
from ..Layers.Transformer_Embedding import TransformerEmbedding


class DecoderLayer(nn.Module):
    def __init__(self,d_model,ffn_hidden,n_head,drop_prob):
        super(DecoderLayer,self).__init__()
        self.attention1=MultiheadAttention(d_model,n_head)
        self.norm1=LayerNorm(d_model)
        self.dropout1=nn.Dropout(drop_prob)

        self.cross_attention=MultiheadAttention(d_model,n_head)
        self.norm2=LayerNorm(d_model)
        self.dropout2=nn.Dropout(drop_prob)

        self.ffn=PositinalWiseFeedForward(d_model,ffn_hidden,drop_prob)
        self.norm3=LayerNorm(d_model)
        self.dropout3=nn.Dropout(drop_prob)

    def forward(self,dec,enc,t_mask,s_mask):
        _x=dec
        x=self.attention1(dec,dec,dec,t_mask)

        x=self.dropout1(x)
        x=self.norm1(x+_x)

        if enc is not None:
            _x=x
            x=self.cross_attention(x,enc,enc,s_mask)

            x=self.dropout2(x)
            x=self.norm2(x+_x)

        _x=x
        x=self.ffn(x)
        
        x=self.dropout3(x)
        x=self.norm3(x+_x)

        return x
    

class Decoder(nn.Module):
    def __init__(self,dec_voc_size,max_len,d_model,ffn_hidden,n_head,n_layer,drop_prob,device):
        super(Decoder, self).__init__()
        
        self.embedding=TransformerEmbedding(d_model,max_len,dec_voc_size,drop_prob,device)
        
        self.layers=nn.ModuleList([
            DecoderLayer(d_model,max_len,n_head,drop_prob) for _ in range(n_layer)
        ])

        self.fc=nn.Linear(d_model,dec_voc_size)

    def forward(self,dec,enc,t_mask,s_mask):
        dec=self.embedding(dec)
        for layer in self.layers:
            dec=layer(dec,enc,t_mask,s_mask)
        dec=self.fc(dec)

        return dec
    