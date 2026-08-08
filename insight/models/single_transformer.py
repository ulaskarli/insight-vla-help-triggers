#!/usr/bin/env python3
import math, torch
import torch.nn as nn

class SinPos(nn.Module):
    def __init__(self, d_model, max_len=4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0)/d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)

    def forward(self, x):  # [B,T,H]
        return x + self.pe[:x.size(1), :]

class SingleStepTransformer(nn.Module):
    """
    Token-level transformer -> attention pooling per step -> step logit.
    Expects a single step (sequence of tokens) as [B,T,D]; use on StepDataset.
    """
    def __init__(self, d_in=4, d_h=64, nhead=4, nlayers=1):
        super().__init__()
        self.proj = nn.Linear(d_in, d_h)
        self.pos  = SinPos(d_h)
        layer = nn.TransformerEncoderLayer(d_model=d_h, nhead=nhead, batch_first=True)
        self.enc  = nn.TransformerEncoder(layer, num_layers=nlayers)
        self.v = nn.Linear(d_h, d_h // 2)
        self.w = nn.Linear(d_h // 2, 1, bias=False)
        self.head = nn.Sequential(nn.Linear(d_h, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, X, tok_pad):   # X:[B,T,D], tok_pad:[B,T]
        x = self.proj(X)
        x = self.pos(x)
        z = self.enc(x, src_key_padding_mask=tok_pad)    # [B,T,H]
        a = self.w(torch.tanh(self.v(z))).squeeze(-1)    # [B,T]
        a = a.masked_fill(tok_pad, -1e9)
        alpha = torch.softmax(a, dim=1).unsqueeze(-1)
        step_emb = (alpha * z).sum(dim=1)                # [B,H]
        step_logit = self.head(step_emb).squeeze(-1)     # [B]
        return step_logit
