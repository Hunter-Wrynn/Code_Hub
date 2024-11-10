import torch.nn as nn
import torch.nn.functional as F
import torch

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, maxlen, device):
        super(PositionalEmbedding, self).__init__()
        self.encoding = torch.zeros(maxlen, d_model, device=device)
        self.encoding.requires_grad_(False)

        pos = torch.arange(0, maxlen, device=device)
        pos = pos.float().unsqueeze(1)
        _2i = torch.arange(0, d_model, 2, device=device)

        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))

    def forward(self, x):
        seq_len = x.shape[1]
        return self.encoding[:seq_len, :]