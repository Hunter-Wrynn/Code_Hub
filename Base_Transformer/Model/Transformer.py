import torch
import torch.nn as nn
import torch.nn.functional as F 
import math
from .Encoder import Encoder
from .Decoder import Decoder


class Transformer(nn.Module):
    def __init__(self,src_pad_idx,trg_pad_idx,enc_voc_size,dec_voc_size,
                 max_len,d_model,n_heads,ffn_hidden,n_layers,drop_prob,device):
        super(Transformer, self).__init__()

        self.encoder=Encoder(enc_voc_size,max_len,d_model,ffn_hidden,n_heads,n_layers,drop_prob,device)
        self.decoder=Decoder(dec_voc_size,max_len,d_model,ffn_hidden,n_heads,n_layers,drop_prob,device)

        self.src_pad_idx=src_pad_idx
        self.trg_pad_idx=trg_pad_idx

        self.device=device


    def make_pad_mask(self,q,k,pad_idx_q,pad_idx_k):
        len_q,len_k=q.size(1),k.size(1)

        q=q.ne(pad_idx_q).unsqueeze(1).unsqueeze(3)
        q=q.repeat(1,1,1,len_k)

        k=k.ne(pad_idx_k).unsqueeze(1).unsqueeze(3)
        k=k.repeat(1,1,len_q,1)

        mask=q&k
        return mask


    def make_casual_mask(self,q,k):
        len_q,len_k=q.size(1),k.size(1)
        mask = torch.tril(torch.ones(len_q, len_k, dtype=torch.bool)).to(self.device)
        return mask
    

    def forward(self,src,trg):
        src_mask=self.make_pad_mask(src,src,self.src_pad_idx,self.src_pad_idx)
        trg_mask=self.make_pad_mask(trg,trg,self.trg_pad_idx,self.trg_pad_idx) * self.make_casual_mask(trg,trg)
        src_trg_mask=self.make_pad_mask(trg,src,self.trg_pad_idx,self.src_pad_idx)
    
        enc=self.encoder(src,src_mask)
        output=self.decoder(trg,src,trg_mask,src_trg_mask)
        return output
    


