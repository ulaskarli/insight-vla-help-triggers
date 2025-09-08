#!/usr/bin/env python3
import os, json, argparse, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path

from insight.datasets import BagDataset, collate_bags, StepDataset, collate_steps
from insight.models.mil_transformer import MILStepTransformer
from insight.models.single_transformer import SingleStepTransformer

def set_seed(s: int):
    torch.manual_seed(s); np.random.seed(s)

def train_mil(cfg, device):
    data_dir   = cfg["data_dir"]
    out_dir    = cfg["out_dir"]
    n_folds    = cfg.get("n_folds", 10)
    params     = cfg["model"]
    batch_epi  = cfg.get("batch_epi", 4)
    epochs     = cfg.get("epochs", 100)
    lr         = cfg.get("lr", 1e-4)
    patience   = cfg.get("patience", 10)
    grad_clip  = cfg.get("grad_clip", 1.0)

    os.makedirs(out_dir, exist_ok=True)
    ckpts = []

    for fold in range(n_folds):
        train_bags_pt = os.path.join(data_dir, f"train_bags_{fold}.pt")
        val_bags_pt   = os.path.join(data_dir, f"val_bags_{fold}.pt")
        assert os.path.isfile(train_bags_pt), f"Missing {train_bags_pt}"

        tr_loader = DataLoader(BagDataset(train_bags_pt), batch_size=batch_epi,
                               shuffle=True, collate_fn=collate_bags)
        va_loader = (DataLoader(BagDataset(val_bags_pt), batch_size=batch_epi,
                                shuffle=False, collate_fn=collate_bags)
                     if os.path.isfile(val_bags_pt) else None)

        model = MILStepTransformer(**params).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=lr)

        def eval_loss_acc(loader):
            model.eval(); tot=0.0; n=0; correct=0; total=0
            with torch.no_grad():
                for X, step_pad, tok_pad, y_bag in loader:
                    X, step_pad, tok_pad = X.to(device), step_pad.to(device), tok_pad.to(device)
                    y_bag = y_bag.to(device)
                    _, bag_prob = model(X, step_pad, tok_pad)
                    loss = F.binary_cross_entropy(bag_prob, y_bag.float())
                    tot += loss.item(); n += 1
                    pred = (bag_prob >= 0.5).long()
                    correct += (pred == y_bag).sum().item()
                    total += y_bag.size(0)
            return tot/max(1,n), correct/max(1,total)

        best=float("inf"); wait=0
        ckpt = os.path.join(out_dir, f"mil_fold{fold}.pt")

        for ep in range(1, epochs+1):
            model.train(); ep_loss=0.0
            for X, step_pad, tok_pad, y_bag in tr_loader:
                X, step_pad, tok_pad = X.to(device), step_pad.to(device), tok_pad.to(device)
                y_bag = y_bag.to(device)
                _, bag_prob = model(X, step_pad, tok_pad)
                loss = F.binary_cross_entropy(bag_prob, y_bag.float())
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                opt.step(); ep_loss += loss.item()

            if va_loader is not None:
                vl, vacc = eval_loss_acc(va_loader)
                print(f"[MIL Fold {fold}] Ep {ep:03d} | Train {ep_loss/len(tr_loader):.4f} | Val {vl:.4f} | ValAcc {vacc:.3f}")
                improved = vl < best - 1e-6
                score = vl
            else:
                trl = ep_loss/len(tr_loader)
                print(f"[MIL Fold {fold}] Ep {ep:03d} | Train {trl:.4f} (no val)")
                improved = trl < best - 1e-6
                score = trl

            if improved:
                best = score; wait = 0; torch.save(model.state_dict(), ckpt)
                print(f"[MIL Fold {fold}] ✅ saved → {ckpt}")
            else:
                wait += 1
                if wait >= patience:
                    print(f"[MIL Fold {fold}] ⛔ early stop")
                    break

        ckpts.append(ckpt)
    Path(os.path.join(out_dir, "mil_ckpts.json")).write_text(json.dumps(ckpts, indent=2))

def train_single(cfg, device):
    data_dir   = cfg["data_dir"]
    out_dir    = cfg["out_dir"]
    n_folds    = cfg.get("n_folds", 10)
    params     = cfg["model"]
    batch_sz   = cfg.get("batch_size", 16)
    epochs     = cfg.get("epochs", 100)
    lr         = cfg.get("lr", 1e-4)
    patience   = cfg.get("patience", 10)
    grad_clip  = cfg.get("grad_clip", 1.0)

    os.makedirs(out_dir, exist_ok=True)
    ckpts = []

    for fold in range(n_folds):
        train_steps_pt = os.path.join(data_dir, f"train_steps_{fold}.pt")
        val_steps_pt   = os.path.join(data_dir, f"val_steps_{fold}.pt")
        assert os.path.isfile(train_steps_pt), f"Missing {train_steps_pt}"

        tr_loader = DataLoader(StepDataset(train_steps_pt), batch_size=batch_sz,
                               shuffle=True, collate_fn=collate_steps)
        va_loader = (DataLoader(StepDataset(val_steps_pt), batch_size=batch_sz,
                                shuffle=False, collate_fn=collate_steps)
                     if os.path.isfile(val_steps_pt) else None)

        model = SingleStepTransformer(**params).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        bce = nn.BCEWithLogitsLoss()

        best=float("inf"); wait=0
        ckpt = os.path.join(out_dir, f"single_fold{fold}.pt")

        for ep in range(1, epochs+1):
            model.train(); ep_loss=0.0
            for X, tok_pad, y in tr_loader:
                X, tok_pad, y = X.to(device), tok_pad.to(device), y.to(device).float()
                logit = model(X, tok_pad)
                loss = bce(logit, y)
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                opt.step(); ep_loss += loss.item()

            if va_loader is not None:
                with torch.no_grad():
                    model.eval(); tot=0.0; n=0
                    for X, tok_pad, y in va_loader:
                        X, tok_pad, y = X.to(device), tok_pad.to(device), y.to(device).float()
                        logit = model(X, tok_pad)
                        loss = bce(logit, y)
                        tot += loss.item(); n += 1
                vl = tot/max(1,n)
                print(f"[Single Fold {fold}] Ep {ep:03d} | Train {ep_loss/len(tr_loader):.4f} | Val {vl:.4f}")
                improved = vl < best - 1e-6; score = vl
            else:
                trl = ep_loss/len(tr_loader)
                print(f"[Single Fold {fold}] Ep {ep:03d} | Train {trl:.4f} (no val)")
                improved = trl < best - 1e-6; score = trl

            if improved:
                best = score; wait=0; torch.save(model.state_dict(), ckpt)
                print(f"[Single Fold {fold}] ✅ saved → {ckpt}")
            else:
                wait += 1
                if wait >= patience:
                    print(f"[Single Fold {fold}] ⛔ early stop"); break

        ckpts.append(ckpt)
    Path(os.path.join(out_dir, "single_ckpts.json")).write_text(json.dumps(ckpts, indent=2))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="YAML config")
    args = ap.parse_args()

    import yaml
    cfg = yaml.safe_load(open(args.config))
    set_seed(cfg.get("seed", 1337))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mode = cfg["mode"]  # "mil" or "single"
    if mode == "mil":
        train_mil(cfg, device)
    elif mode == "single":
        train_single(cfg, device)
    else:
        raise ValueError("mode must be 'mil' or 'single'")

if __name__ == "__main__":
    main()