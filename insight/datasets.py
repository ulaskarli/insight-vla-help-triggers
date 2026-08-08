#!/usr/bin/env python3
import os, torch
import torch.nn as nn
from torch.utils.data import Dataset

# ---------- Weak-label (bags) ----------
class BagDataset(Dataset):
    """
    Reads (bags_X, bags_y, bags_src) from *{train,val,test}_bags_{fold}.pt
      - bags_X: List[List[np.ndarray[T,D]]]  (steps per bag)
      - bags_y: np.ndarray [N] (0/1)
    """
    def __init__(self, pt_path: str):
        packed = torch.load(pt_path, weights_only=False)
        if len(packed) == 3:
            X, y, _ = packed
        else:
            X, y = packed[:2]
        self.X = [[torch.tensor(s, dtype=torch.float32) for s in bag] for bag in X]
        self.y = torch.tensor(y, dtype=torch.long)
        assert len(self.X) == len(self.y)

    def __len__(self): return len(self.y)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

def collate_bags(batch):
    """Return:
      X:        [B,Smax,Tmax,D]
      step_pad: [B,Smax]      True=PAD step
      tok_pad:  [B,Smax,Tmax] True=PAD token
      y_bag:    [B]
    """
    bags, ys = zip(*batch)
    B = len(bags)
    Smax = max(len(b) for b in bags)
    Tmax = max((s.shape[0] for bag in bags for s in bag), default=1)
    D = bags[0][0].shape[1] if Smax > 0 else 4

    X = torch.zeros(B, Smax, Tmax, D, dtype=torch.float32)
    step_pad = torch.ones(B, Smax, dtype=torch.bool)
    tok_pad  = torch.ones(B, Smax, Tmax, dtype=torch.bool)
    y_bag    = torch.stack(ys).long()

    for b, steps in enumerate(bags):
        step_pad[b, :len(steps)] = False
        for s, arr in enumerate(steps):
            T = arr.shape[0]
            X[b, s, :T] = arr
            tok_pad[b, s, :T] = False
    return X, step_pad, tok_pad, y_bag

# ---------- Strong-label (steps) ----------
class StepDataset(Dataset):
    """Strong per-step test: (X_steps, y_steps) in *test_steps_{fold}.pt"""
    def __init__(self, pt_path: str):
        X, y = torch.load(pt_path, weights_only=False)
        self.X = [torch.tensor(x, dtype=torch.float32) for x in X]
        self.y = torch.tensor(y, dtype=torch.long)
        assert len(self.X) == len(self.y)

    def __len__(self): return len(self.y)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

def collate_steps(batch):
    xs, ys = zip(*batch)
    lengths = [len(x) for x in xs]
    X = nn.utils.rnn.pad_sequence(xs, batch_first=True)  # [B,Tmax,D]
    B, Tmax, _ = X.shape
    tok_pad = torch.ones(B, Tmax, dtype=torch.bool)
    for i, T in enumerate(lengths):
        tok_pad[i, :T] = False
    y = torch.stack(ys).long()
    return X, tok_pad, y