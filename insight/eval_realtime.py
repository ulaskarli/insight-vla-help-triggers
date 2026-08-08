#!/usr/bin/env python3
import os, json, argparse, numpy as np, torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, List, Any, Tuple

from insight.datasets import BagDataset, collate_bags
from insight.models.single_transformer import SingleStepTransformer
from insight.models.mil_transformer import MILStepTransformer

# ---------------------------
# Utils
# ---------------------------
def sigmoid(x): return 1/(1+np.exp(-x))

def first_true_index(b: np.ndarray) -> int:
    """Return first index where b is True, else -1."""
    idx = np.argmax(b) if b.any() else -1
    if idx == 0 and not b[0]:  # np.argmax returns 0 if all False
        return -1
    return int(idx)

def episode_metrics(flags: List[np.ndarray], y_bag: List[int]) -> Dict[str, float]:
    """
    flags: list of boolean arrays per episode (length = num steps in episode)
    y_bag: list of 0/1 episode labels (0=success, 1=failure)
    """
    assert len(flags) == len(y_bag)
    n = len(flags)
    succ_idx = [i for i,y in enumerate(y_bag) if y == 0]
    fail_idx = [i for i,y in enumerate(y_bag) if y == 1]

    # Success Purity (no flag on success)
    sp = np.mean([float(flags[i].sum() == 0) for i in succ_idx]) if succ_idx else np.nan
    # Failure Detection (at least one flag on failure)
    fd = np.mean([float(flags[i].sum() >= 1) for i in fail_idx]) if fail_idx else np.nan

    # Flag counts
    flags_succ = [int(flags[i].sum()) for i in succ_idx]
    flags_fail = [int(flags[i].sum()) for i in fail_idx]
    mean_flags_succ = float(np.mean(flags_succ)) if flags_succ else np.nan
    mean_flags_fail = float(np.mean(flags_fail)) if flags_fail else np.nan

    # TTFH and normalized TTFH for failures
    ttfh = []
    nttfh = []
    for i in fail_idx:
        f = flags[i]
        k = first_true_index(f)
        if k >= 0:
            ttfh.append(k+1)                     # 1-based step index
            nttfh.append((k+1)/len(f))          # normalized to episode length
    mean_ttfh = float(np.mean(ttfh)) if ttfh else np.nan
    mean_nttfh = float(np.mean(nttfh)) if nttfh else np.nan

    # Flag rate per episode
    rates = [float(flags[i].sum())/max(1,len(flags[i])) for i in range(n)]
    rate_succ = float(np.mean([rates[i] for i in succ_idx])) if succ_idx else np.nan
    rate_fail = float(np.mean([rates[i] for i in fail_idx])) if fail_idx else np.nan

    return dict(
        episodes=n, n_success=len(succ_idx), n_failure=len(fail_idx),
        success_purity=sp, failure_detection=fd,
        mean_flags_success=mean_flags_succ, mean_flags_failure=mean_flags_fail,
        mean_ttfh_fail=mean_ttfh, mean_nttfh_fail=mean_nttfh,
        flag_rate_success=rate_succ, flag_rate_failure=rate_fail,
    )

# ---------------------------
# CP (entropy / perplexity) flags
# ---------------------------
def entropy_score_seq(seq: np.ndarray, agg: str="p90", trim: float=0.10) -> float:
    ent = seq[:,2]
    if ent.size == 0: return float("nan")
    if agg == "mean": return float(ent.mean())
    if agg == "p90":  return float(np.percentile(ent, 90))
    # trimmed mean
    lo, hi = np.percentile(ent, [100*trim, 100*(1-trim)])
    keep = ent[(ent >= lo) & (ent <= hi)]
    return float(keep.mean()) if keep.size else float(ent.mean())

def perplexity_score_seq(seq: np.ndarray) -> float:
    logp = seq[:,3]
    if logp.size == 0: return float("nan")
    return float(np.exp(-np.mean(logp)))

def flags_from_cp(episode_steps: List[np.ndarray], tau_ent: float, tau_ppl: float,
                  ent_agg="p90", ent_trim=0.10, mode="any") -> np.ndarray:
    """
    mode: "any"    -> flag if (entropy > tau_ent) OR (ppl > tau_ppl)
          "both"   -> flag if (entropy > tau_ent) AND (ppl > tau_ppl)
          "ent"    -> only entropy threshold
          "ppl"    -> only perplexity threshold
    """
    f = []
    for seq in episode_steps:
        s_ent = entropy_score_seq(seq, agg=ent_agg, trim=ent_trim)
        s_ppl = perplexity_score_seq(seq)
        b_ent = (s_ent > tau_ent) if not np.isnan(s_ent) else False
        b_ppl = (s_ppl > tau_ppl) if not np.isnan(s_ppl) else False
        if mode == "any":   flag = b_ent or b_ppl
        elif mode == "both":flag = b_ent and b_ppl
        elif mode == "ent": flag = b_ent
        else:               flag = b_ppl
        f.append(flag)
    return np.array(f, dtype=bool)

# ---------------------------
# Single-strong flags (step logits -> threshold)
# ---------------------------
@torch.no_grad()
def flags_from_single(model: SingleStepTransformer,
                      episode_steps: List[np.ndarray],
                      device: torch.device,
                      thr_logit: float = 0.0) -> np.ndarray:
    """
    thr_logit is threshold on the raw logit (0 means prob>=0.5).
    """
    flags = []
    for seq in episode_steps:
        x = torch.tensor(seq, dtype=torch.float32, device=device).unsqueeze(0)  # [1,T,4]
        tok_pad = torch.zeros(1, x.size(1), dtype=torch.bool, device=device)
        logit = model(x, tok_pad).item()
        flags.append(logit >= thr_logit)
    return np.array(flags, dtype=bool)

# ---------------------------
# MIL-weak flags via step logits
# ---------------------------
@torch.no_grad()
def flags_from_mil(model: MILStepTransformer,
                   episode_steps: List[np.ndarray],
                   device: torch.device,
                   thr_logit: float = 0.0) -> np.ndarray:
    """
    Use the MIL head's step logits before pooling as a step signal.
    """
    # Pack into a single-bag batch: [B=1, S, T, D]
    S = len(episode_steps)
    T = max([s.shape[0] for s in episode_steps]) if S>0 else 1
    D = episode_steps[0].shape[1] if S>0 else 4

    X = torch.zeros(1, S, T, D, dtype=torch.float32, device=device)
    step_pad = torch.ones(1, S, dtype=torch.bool, device=device)
    tok_pad  = torch.ones(1, S, T, dtype=torch.bool, device=device)
    for s_idx, seq in enumerate(episode_steps):
        t = seq.shape[0]
        X[0, s_idx, :t, :] = torch.tensor(seq, dtype=torch.float32, device=device)
        step_pad[0, s_idx] = False
        tok_pad[0, s_idx, :t] = False

    step_logits, _ = model(X, step_pad, tok_pad)  # [1,S]
    step_logits = step_logits[0].detach().cpu().numpy()  # [S]
    return (step_logits >= thr_logit)

# ---------------------------
# End-to-end evaluation
# ---------------------------
def evaluate_realtime(cfg: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    data_dir   = cfg["data_dir"]
    n_folds    = int(cfg.get("n_folds", 10))

    run_single = bool(cfg.get("eval_single_on_bags", False))
    run_mil    = bool(cfg.get("eval_mil_steps", False))
    run_cp     = bool(cfg.get("eval_cp", False))

    # thresholds / ckpts
    single_ckpts = cfg.get("single_ckpts", {})     # {fold: path}
    mil_ckpts    = cfg.get("mil_ckpts", {})        # {fold: path}

    thr_single_logit = float(cfg.get("single_thr_logit", 0.0))  # 0 == prob 0.5
    thr_mil_logit    = float(cfg.get("mil_thr_logit", 0.0))

    cp_mode    = cfg.get("cp_mode", "any")         # "any"|"both"|"ent"|"ppl"
    tau_ent    = cfg.get("cp_tau_entropy", None)   # number or {fold: number}
    tau_ppl    = cfg.get("cp_tau_perplex", None)   # number or {fold: number}
    ent_agg    = cfg.get("cp_entropy_agg", "p90")
    ent_trim   = float(cfg.get("cp_entropy_trim", 0.10))

    results = {}

    for fold in range(n_folds):
        test_bags_pt = os.path.join(data_dir, f"test_bags_{fold}.pt")
        assert os.path.isfile(test_bags_pt), f"Missing {test_bags_pt}"
        ds = BagDataset(test_bags_pt)
        loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collate_bags)

        # Prepare models as needed
        single_model = None
        mil_model = None
        if run_single:
            ckpt = single_ckpts.get(str(fold)) or single_ckpts.get(fold)
            assert ckpt and os.path.isfile(ckpt), f"Missing single ckpt for fold {fold}"
            single_model = SingleStepTransformer().to(device)
            single_model.load_state_dict(torch.load(ckpt, map_location=device))
            single_model.eval()

        if run_mil:
            ckpt = mil_ckpts.get(str(fold)) or mil_ckpts.get(fold)
            assert ckpt and os.path.isfile(ckpt), f"Missing MIL ckpt for fold {fold}"
            mil_model = MILStepTransformer().to(device)
            mil_model.load_state_dict(torch.load(ckpt, map_location=device))
            mil_model.eval()

        fold_flags = {"cp": [], "single": [], "mil": []}
        fold_labels = []

        for X, step_pad, tok_pad, y_bag in loader:
            # unpack bag into python lists of np arrays
            X = X[0]               # [S,T,D]
            step_pad = step_pad[0] # [S]
            tok_pad = tok_pad[0]   # [S,T]
            steps = []
            for s in range(X.size(0)):
                if step_pad[s]:
                    continue
                t = (~tok_pad[s]).sum().item()
                steps.append(X[s, :t, :].cpu().numpy())

            fold_labels.append(int(y_bag.item()))

            # CP
            if run_cp:
                te = tau_ent[str(fold)] if isinstance(tau_ent, dict) else tau_ent
                tp = tau_ppl[str(fold)] if isinstance(tau_ppl, dict) else tau_ppl
                assert te is not None and tp is not None, "Provide cp_tau_entropy / cp_tau_perplex"
                f_cp = flags_from_cp(steps, float(te), float(tp), ent_agg, ent_trim, mode=cp_mode)
                fold_flags["cp"].append(f_cp)

            # Single-strong
            if run_single:
                f_single = flags_from_single(single_model, steps, device, thr_logit=thr_single_logit)
                fold_flags["single"].append(f_single)

            # MIL-weak (step logits)
            if run_mil:
                f_mil = flags_from_mil(mil_model, steps, device, thr_logit=thr_mil_logit)
                fold_flags["mil"].append(f_mil)

        # compute episode metrics per method
        per_method = {}
        for k,v in fold_flags.items():
            if len(v) == 0: continue
            per_method[k] = episode_metrics(v, fold_labels)

        results[f"fold_{fold}"] = per_method

    # Aggregate across folds: simple means of the scalars present
    def agg(method: str) -> Dict[str, float]:
        keys = set()
        for f in results.values():
            if method in f:
                keys.update(f[method].keys())
        out = {}
        for k in keys:
            vals = [f[method][k] for f in results.values() if method in f and f[method][k] == f[method][k]]  # drop NaN
            if vals:
                out[k] = float(np.mean(vals))
        return out

    summary = {}
    methods = set()
    for f in results.values(): methods.update(f.keys())
    for m in methods: summary[m] = agg(m)

    return {"per_fold": results, "summary": summary}

# ---------------------------
# CLI
# ---------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="YAML config for realtime eval")
    args = ap.parse_args()

    import yaml
    cfg = yaml.safe_load(open(args.config))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_path = cfg.get("out_path", None)

    report = evaluate_realtime(cfg, device)
    print(json.dumps(report["summary"], indent=2))
    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        json.dump(report, open(out_path, "w"), indent=2)
        print(f"Saved full report → {out_path}")

if __name__ == "__main__":
    main()