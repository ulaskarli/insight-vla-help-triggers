#!/usr/bin/env python3
import os, json, argparse, numpy as np, torch
from torch.utils.data import DataLoader
from insight.datasets import BagDataset, collate_bags, StepDataset, collate_steps
from insight.models.mil_transformer import MILStepTransformer
from insight.models.single_transformer import SingleStepTransformer
from insight.utils.metrics import confusion, agg_macro, agg_micro

@torch.no_grad()
def eval_bag_level_mil(ckpt_path, data_dir, device, batch_epi=4):
    model = MILStepTransformer().to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device)); model.eval()
    fold = int(os.path.basename(ckpt_path).split("fold")[-1].split(".")[0])
    test_bags_pt = os.path.join(data_dir, f"test_bags_{fold}.pt")
    assert os.path.isfile(test_bags_pt), f"Missing {test_bags_pt}"

    te_loader = DataLoader(BagDataset(test_bags_pt), batch_size=batch_epi,
                           shuffle=False, collate_fn=collate_bags)

    probs, labels = [], []
    for X, step_pad, tok_pad, y_bag in te_loader:
        X, step_pad, tok_pad = X.to(device), step_pad.to(device), tok_pad.to(device)
        y_bag = y_bag.to(device)
        _, bag_prob = model(X, step_pad, tok_pad)
        probs.extend(bag_prob.cpu().tolist()); labels.extend(y_bag.cpu().tolist())

    tp,fp,tn,fn,acc,prec,rec,f1 = confusion(labels, probs, thr=0.5)
    return {"fold": fold, "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "accuracy": acc, "precision": prec, "recall": rec, "f1": f1,
            "n": int(len(labels)), "threshold": 0.5}

@torch.no_grad()
def eval_step_level_mil(ckpt_path, data_dir, device, batch_sz=16):
    model = MILStepTransformer().to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device)); model.eval()
    fold = int(os.path.basename(ckpt_path).split("fold")[-1].split(".")[0])
    test_steps_pt = os.path.join(data_dir, f"test_steps_{fold}.pt")
    assert os.path.isfile(test_steps_pt), f"Missing {test_steps_pt}"

    te_loader = DataLoader(StepDataset(test_steps_pt), batch_size=batch_sz,
                           shuffle=False, collate_fn=collate_steps)

    probs, labels = [], []
    for X, tok_pad, y in te_loader:
        X, tok_pad = X.to(device), tok_pad.to(device)
        # Treat each step as a single-instance bag: [B,1,T,D]
        X4 = X.unsqueeze(1)
        step_pad = torch.zeros(X4.size(0), 1, dtype=torch.bool, device=device)
        tok_pad4 = tok_pad.unsqueeze(1)
        step_logits, _ = model(X4, step_pad, tok_pad4)
        p = torch.sigmoid(step_logits[:, 0]).cpu().tolist()
        probs.extend(p); labels.extend(y.cpu().tolist())

    tp,fp,tn,fn,acc,prec,rec,f1 = confusion(labels, probs, thr=0.5)
    return {"fold": fold, "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "accuracy": acc, "precision": prec, "recall": rec, "f1": f1,
            "n": int(len(labels)), "threshold": 0.5}

@torch.no_grad()
def eval_step_level_single(ckpt_path, data_dir, device, batch_sz=16):
    model = SingleStepTransformer().to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device)); model.eval()
    fold = int(os.path.basename(ckpt_path).split("fold")[-1].split(".")[0])
    test_steps_pt = os.path.join(data_dir, f"test_steps_{fold}.pt")
    assert os.path.isfile(test_steps_pt), f"Missing {test_steps_pt}"

    from insight.datasets import StepDataset, collate_steps
    te_loader = DataLoader(StepDataset(test_steps_pt), batch_size=batch_sz,
                           shuffle=False, collate_fn=collate_steps)

    probs, labels = [], []
    for X, tok_pad, y in te_loader:
        X, tok_pad = X.to(device), tok_pad.to(device)
        logit = model(X, tok_pad)
        p = torch.sigmoid(logit).cpu().tolist()
        probs.extend(p if isinstance(p, list) else [p]); labels.extend(y.cpu().tolist())

    tp,fp,tn,fn,acc,prec,rec,f1 = confusion(labels, probs, thr=0.5)
    return {"fold": fold, "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "accuracy": acc, "precision": prec, "recall": rec, "f1": f1,
            "n": int(len(labels)), "threshold": 0.5}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["mil-weak", "mil-strong", "single-strong"])
    ap.add_argument("--ckpts_json", required=True, help="Path to ckpt list JSON from training")
    ap.add_argument("--data_dir", required=True)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpts = json.loads(open(args.ckpts_json).read())

    per_fold = []
    for ck in ckpts:
        if args.mode == "mil-weak":
            m = eval_bag_level_mil(ck, args.data_dir, device)
        elif args.mode == "mil-strong":
            m = eval_step_level_mil(ck, args.data_dir, device)
        else:
            m = eval_step_level_single(ck, args.data_dir, device)
        per_fold.append(m)

    from insight.utils.metrics import agg_macro, agg_micro
    report = {"per_fold": per_fold, "macro": agg_macro(per_fold), "micro": agg_micro(per_fold)}
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()