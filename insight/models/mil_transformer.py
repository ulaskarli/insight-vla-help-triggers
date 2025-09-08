#!/usr/bin/env python3
import math, torch
import torch.nn as nn
import torch.nn.functional as F

class SinPos(nn.Module):
    def __init__(self, d_model, max_len=4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0)/d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)

    def forward(self, x):  # [N,T,H]
        return x + self.pe[:x.size(1), :]

class TokenEncoder(nn.Module):
    def __init__(self, d_in=4, d_h=64, nhead=4, nlayers=1):
        super().__init__()
        self.proj = nn.Linear(d_in, d_h)
        self.pos  = SinPos(d_h)
        layer = nn.TransformerEncoderLayer(d_model=d_h, nhead=nhead, batch_first=True)
        self.enc  = nn.TransformerEncoder(layer, num_layers=nlayers)
        self.v = nn.Linear(d_h, d_h // 2)
        self.w = nn.Linear(d_h // 2, 1, bias=False)

    def forward(self, X, tok_pad):   # X:[B,S,T,D], tok_pad:[B,S,T]
        B, S, T, _ = X.shape
        x = self.proj(X).view(B * S, T, -1)
        x = self.pos(x)
        m = tok_pad.view(B * S, T)
        z = self.enc(x, src_key_padding_mask=m)          # [B*S,T,H]
        a = self.w(torch.tanh(self.v(z))).squeeze(-1)    # [B*S,T]
        a = a.masked_fill(m, -1e9)
        alpha = torch.softmax(a, dim=1).unsqueeze(-1)
        step_emb = (alpha * z).sum(dim=1).view(B, S, -1) # [B,S,H]
        return step_emb

class MILStepTransformer(nn.Module):
    def __init__(self, d_in=4, d_h=64, nhead=4, nlayers=1, pooling="lse", beta=6.0):
        super().__init__()
        assert pooling in {"max","lse","noisy_or"}
        self.token_enc = TokenEncoder(d_in=d_in, d_h=d_h, nhead=nhead, nlayers=nlayers)
        self.head = nn.Sequential(nn.Linear(d_h, 32), nn.ReLU(), nn.Linear(32, 1))
        self.pooling, self.beta = pooling, beta

    def forward(self, X, step_pad, tok_pad):
        """
        X: [B,S,T,D], step_pad:[B,S] True=PAD, tok_pad:[B,S,T] True=PAD
        Returns:
          step_logits: [B,S]
          bag_prob:    [B]
        """
        step_emb = self.token_enc(X, tok_pad)            # [B,S,H]
        step_logits = self.head(step_emb).squeeze(-1)    # [B,S]
        masked = step_logits.masked_fill(step_pad, float("-inf"))

        if self.pooling == "max":
            pooled, _ = masked.max(dim=1)
            pooled = torch.where(torch.isinf(pooled), torch.full_like(pooled, -1e9), pooled)
            bag_prob = torch.sigmoid(pooled)
        elif self.pooling == "lse":
            safe = torch.where(torch.isinf(masked), torch.full_like(masked, -1e9), masked)
            pooled = torch.logsumexp(self.beta * safe, dim=1) / self.beta
            bag_prob = torch.sigmoid(pooled)
        else:  # noisy_or
            p = torch.sigmoid(step_logits)
            p = torch.where(~step_pad, p, torch.zeros_like(p))
            bag_prob = 1.0 - torch.clamp(1.0 - p, 1e-6, 1.0).prod(dim=1)
        return step_logits, bag_prob