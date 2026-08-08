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

@torch.no_grad()
def eval_bag_level_single(ckpt_path, data_dir, device, batch_sz=16):
    """
    Evaluate a step-supervised SingleStepTransformer on bag-level test data.
    - Loads fold id from ckpt filename (…fold{K}.pt)
    - Uses test_bags_{K}.pt
    - For each bag, runs the single model on each step, then pools step logits -> bag prob.
      Pooling: max over step logits (equiv. prob = sigmoid(max_logit)).
    Returns the same metrics dict shape as eval_step_level_single (but for bag labels).
    """
    from insight.datasets import BagDataset, collate_bags
    from torch.utils.data import DataLoader

    # model
    model = SingleStepTransformer().to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device)); model.eval()

    # infer fold from ckpt name
    fold = int(os.path.basename(ckpt_path).split("fold")[-1].split(".")[0])

    # load bag-level test split
    test_bags_pt = os.path.join(data_dir, f"test_bags_{fold}.pt")
    assert os.path.isfile(test_bags_pt), f"Missing {test_bags_pt}"

    te_loader = DataLoader(BagDataset(test_bags_pt), batch_size=batch_sz,
                           shuffle=False, collate_fn=collate_bags)

    probs, labels = [], []

    for X, step_pad, tok_pad, y_bag in te_loader:
        # X: [B,S,T,D], step_pad: [B,S] (True=PAD), tok_pad: [B,S,T], y_bag: [B]
        B, S, T, D = X.shape
        X = X.to(device); step_pad = step_pad.to(device); tok_pad = tok_pad.to(device)

        # collect all real steps across the batch into a flat step-batch for the single model
        step_slices = []  # will hold tuples (b_idx, s_idx, t_len)
        for b in range(B):
            for s in range(S):
                if not step_pad[b, s]:
                    t_len = int((~tok_pad[b, s]).sum().item())
                    step_slices.append((b, s, t_len))

        if len(step_slices) == 0:
            # unlikely, but skip if this batch is entirely padding
            labels.extend(y_bag.tolist()); probs.extend([0.0]*len(y_bag))
            continue

        # build a packed [Nsteps, Tmax, D] and mask for the single model
        Tmax = max(t for _, _, t in step_slices)
        Nst  = len(step_slices)
        X_steps = torch.zeros(Nst, Tmax, D, dtype=torch.float32, device=device)
        tok_mask = torch.ones(Nst, Tmax, dtype=torch.bool, device=device)

        for i, (b, s, t_len) in enumerate(step_slices):
            X_steps[i, :t_len, :] = X[b, s, :t_len, :]
            tok_mask[i, :t_len] = False

        # run single-step model on all steps at once
        step_logits = model(X_steps, tok_mask).detach()   # [Nsteps]

        # pool per bag: max over step logits in that bag
        # map step indices back to their bag index
        bag_logits = torch.full((B,), fill_value=-1e9, dtype=torch.float32, device=device)
        for i, (b, s, t_len) in enumerate(step_slices):
            bag_logits[b] = torch.maximum(bag_logits[b], step_logits[i])

        bag_probs = torch.sigmoid(bag_logits).detach().cpu().tolist()
        probs.extend(bag_probs)
        labels.extend(y_bag.cpu().tolist())

    # same metric shape as eval_step_level_single (bag-level confusion)
    tp, fp, tn, fn, acc, prec, rec, f1 = confusion(labels, probs, thr=0.5)
    return {
        "fold": fold,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "accuracy": acc, "precision": prec, "recall": rec, "f1": f1,
        "n": int(len(labels)), "threshold": 0.5
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["mil-weak", "mil-strong", "single-strong", "single-weak"])
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
        elif args.mode == "single-weak":
            m = eval_bag_level_single(ck, args.data_dir, device)
        else:
            m = eval_step_level_single(ck, args.data_dir, device)
        per_fold.append(m)

    from insight.utils.metrics import agg_macro, agg_micro
    report = {"per_fold": per_fold, "macro": agg_macro(per_fold), "micro": agg_micro(per_fold)}
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()