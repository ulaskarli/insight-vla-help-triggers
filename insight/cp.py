#!/usr/bin/env python3
import os, json, math, argparse, numpy as np, torch, yaml
from typing import List, Tuple, Dict

# ------------------------
# IO helpers
# ------------------------
def load_split(path: str) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    Each .pt is (X, y).
    X: list of [T, D] arrays; columns expected:
       0: AU, 1: EU, 2: entropy, 3: logp
    y: array/list of 0/1 step labels (0=no-help, 1=help)
    """
    X, y = torch.load(path, weights_only=False)
    X_np = []
    for seq in X:
        arr = np.asarray(seq)
        if arr.ndim != 2 or arr.shape[1] < 4:
            raise ValueError(f"{path}: sequence has shape {arr.shape}, expected [T, >=4].")
        X_np.append(arr)
    y_np = np.asarray(y, dtype=np.float32)
    return X_np, y_np

# ------------------------
# Scores
# ------------------------
def _trimmed_mean(v: np.ndarray, trim: float) -> float:
    lo, hi = np.percentile(v, [100*trim, 100*(1-trim)])
    keep = v[(v >= lo) & (v <= hi)]
    return float(keep.mean()) if keep.size else float(v.mean())

def entropy_score(seq: np.ndarray, agg: str = "p90", trim: float = 0.10) -> float:
    ent = seq[:, 2]
    if ent.size == 0: return float("nan")
    if agg == "mean": return float(ent.mean())
    if agg == "p90":  return float(np.percentile(ent, 90))
    if agg == "trimmed_mean": return _trimmed_mean(ent, trim)
    raise ValueError(f"Unknown ENT_AGG: {agg}")

def perplexity_score(seq: np.ndarray) -> float:
    logp = seq[:, 3]
    if logp.size == 0: return float("nan")
    return float(np.exp(-np.mean(logp)))

# ------------------------
# Conformal quantiles
# ------------------------
def upper_tail_quantile(scores: List[float], q: float) -> float:
    if not scores: raise ValueError("No calibration scores.")
    s = np.sort(np.array(scores, dtype=np.float64))
    n = s.shape[0]
    k = int(math.ceil((n + 1) * q))
    k = max(1, min(k, n))
    return float(s[k - 1])

# ------------------------
# Collect calibration scores by class
# ------------------------
def collect_safe_scores(X_list: List[np.ndarray], y: np.ndarray, ent_agg: str, ent_trim: float) -> Tuple[List[float], List[float]]:
    ent_scores, ppl_scores = [], []
    for seq, lab in zip(X_list, y):
        if lab == 0.0:
            es = entropy_score(seq, agg=ent_agg, trim=ent_trim)
            ps = perplexity_score(seq)
            if not np.isnan(es): ent_scores.append(es)
            if not np.isnan(ps): ppl_scores.append(ps)
    return ent_scores, ppl_scores

def collect_help_scores(X_list: List[np.ndarray], y: np.ndarray, ent_agg: str, ent_trim: float) -> Tuple[List[float], List[float]]:
    ent_scores, ppl_scores = [], []
    for seq, lab in zip(X_list, y):
        if lab == 1.0:
            es = entropy_score(seq, agg=ent_agg, trim=ent_trim)
            ps = perplexity_score(seq)
            if not np.isnan(es): ent_scores.append(es)
            if not np.isnan(ps): ppl_scores.append(ps)
    return ent_scores, ppl_scores

# ------------------------
# Evaluation
# ------------------------
def eval_with_thresholds(
    X_test: List[np.ndarray], y_test: np.ndarray,
    tau_ent: float, tau_ppl: float,
    ent_agg: str, ent_trim: float
) -> Dict[str, float]:
    ent_scores = np.array([entropy_score(seq, agg=ent_agg, trim=ent_trim) for seq in X_test], dtype=np.float64)
    ppl_scores = np.array([perplexity_score(seq) for seq in X_test], dtype=np.float64)

    asks_ent = ent_scores > tau_ent
    asks_ppl = ppl_scores > tau_ppl

    safe_mask = (y_test == 0.0)
    help_mask = (y_test == 1.0)
    Ns, Nh = int(safe_mask.sum()), int(help_mask.sum())

    tp_ent = int(np.logical_and(asks_ent, help_mask).sum())
    fp_ent = int(np.logical_and(asks_ent, safe_mask).sum())
    tn_ent = int(np.logical_and(~asks_ent, safe_mask).sum())
    fn_ent = int(np.logical_and(~asks_ent, help_mask).sum())

    tp_ppl = int(np.logical_and(asks_ppl, help_mask).sum())
    fp_ppl = int(np.logical_and(asks_ppl, safe_mask).sum())
    tn_ppl = int(np.logical_and(~asks_ppl, safe_mask).sum())
    fn_ppl = int(np.logical_and(~asks_ppl, help_mask).sum())

    def derived(tp, fp, tn, fn):
        tot = tp + fp + tn + fn
        acc = (tp + tn) / max(1, tot)
        prec = tp / max(1, tp + fp)
        rec = tp / max(1, tp + fn)
        f1 = (2 * prec * rec) / max(1e-8, (prec + rec))
        return acc, prec, rec, f1

    acc_e, prec_e, rec_e, f1_e = derived(tp_ent, fp_ent, tn_ent, fn_ent)
    acc_p, prec_p, rec_p, f1_p = derived(tp_ppl, fp_ppl, tn_ppl, fn_ppl)

    fa_ent = fp_ent / max(1, Ns) if Ns else np.nan
    fa_ppl = fp_ppl / max(1, Ns) if Ns else np.nan

    rc_ent = tp_ent / max(1, Nh) if Nh else np.nan
    rc_ppl = tp_ppl / max(1, Nh) if Nh else np.nan

    miss_ent = 1.0 - rc_ent if rc_ent==rc_ent else np.nan
    miss_ppl = 1.0 - rc_ppl if rc_ppl==rc_ppl else np.nan

    rate_ent = float(asks_ent.mean())
    rate_ppl = float(asks_ppl.mean())

    return {
        # confusion
        "tp_entropy": tp_ent, "fp_entropy": fp_ent, "tn_entropy": tn_ent, "fn_entropy": fn_ent,
        "tp_perplex": tp_ppl, "fp_perplex": fp_ppl, "tn_perplex": tn_ppl, "fn_perplex": fn_ppl,

        # rates & recall/miss
        "false_ask_entropy": fa_ent, "false_ask_perplex": fa_ppl,
        "recall_entropy": rc_ent,     "recall_perplex": rc_ppl,
        "miss_help_entropy": miss_ent,"miss_help_perplex": miss_ppl,
        "rate_entropy": rate_ent,     "rate_perplex": rate_ppl,

        # derived metrics
        "accuracy_entropy": acc_e, "precision_entropy": prec_e, "f1_entropy": f1_e,
        "accuracy_perplex": acc_p, "precision_perplex": prec_p, "f1_perplex": f1_p,

        "Ns": Ns, "Nh": Nh
    }

# ------------------------
# Aggregation
# ------------------------
def mean_std(key: str, items: List[Dict[str, float]]) -> Tuple[float, float]:
    vals = [it.get(key) for it in items if key in it and not np.isnan(it.get(key))]
    return (float(np.mean(vals)), float(np.std(vals))) if vals else (float("nan"), float("nan"))

def micro_confusion(items: List[Dict[str, float]], which: str) -> Dict[str, int]:
    tp = sum(int(it.get(f"tp_{which}", 0)) for it in items)
    fp = sum(int(it.get(f"fp_{which}", 0)) for it in items)
    tn = sum(int(it.get(f"tn_{which}", 0)) for it in items)
    fn = sum(int(it.get(f"fn_{which}", 0)) for it in items)
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}

def derived_from_confusion(c: Dict[str, int]) -> Dict[str, float]:
    tp, fp, tn, fn = c["tp"], c["fp"], c["tn"], c["fn"]
    tot = tp + fp + tn + fn
    acc = (tp + tn) / max(1, tot)
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    f1 = (2 * prec * rec) / max(1e-8, (prec + rec))
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}

# ------------------------
# Fold runners
# ------------------------
def run_fold_fa(k: int, cfg: Dict) -> Dict[str, float]:
    data_root = cfg["data_root"]
    use_val   = cfg.get("use_val_in_calib", True)
    ent_agg   = cfg.get("ent_agg", "p90")
    ent_trim  = cfg.get("ent_trim", 0.10)
    alpha     = cfg["alpha"]

    train_path = os.path.join(data_root, f"train_{k}.pt")
    test_path  = os.path.join(data_root, f"test_{k}.pt")
    val_path   = os.path.join(data_root, f"val_{k}.pt")

    X_tr, y_tr = load_split(train_path)
    X_te, y_te = load_split(test_path)

    if use_val and os.path.exists(val_path):
        X_va, y_va = load_split(val_path)
        X_cal = X_tr + X_va
        y_cal = np.concatenate([y_tr, y_va], 0)
    else:
        X_cal, y_cal = X_tr, y_tr

    safe_ent_scores, safe_ppl_scores = collect_safe_scores(X_cal, y_cal, ent_agg, ent_trim)
    if len(safe_ent_scores) == 0 or len(safe_ppl_scores) == 0:
        raise ValueError(f"Fold {k}: no SAFE steps for calibration.")

    tau_ent = upper_tail_quantile(safe_ent_scores, q=1.0 - alpha)
    tau_ppl = upper_tail_quantile(safe_ppl_scores, q=1.0 - alpha)

    m = eval_with_thresholds(X_te, y_te, tau_ent, tau_ppl, ent_agg, ent_trim)
    m.update({
        "fold": k, "alpha": alpha, "mode": "fa",
        "tau_entropy": tau_ent, "tau_perplex": tau_ppl,
        "calib_safe_steps_entropy": len(safe_ent_scores),
        "calib_safe_steps_perplex": len(safe_ppl_scores),
    })
    return m

def run_fold_fn(k: int, cfg: Dict) -> Dict[str, float]:
    data_root = cfg["data_root"]
    use_val   = cfg.get("use_val_in_calib", True)
    ent_agg   = cfg.get("ent_agg", "p90")
    ent_trim  = cfg.get("ent_trim", 0.10)
    beta      = cfg["beta"]

    train_path = os.path.join(data_root, f"train_{k}.pt")
    test_path  = os.path.join(data_root, f"test_{k}.pt")
    val_path   = os.path.join(data_root, f"val_{k}.pt")

    X_tr, y_tr = load_split(train_path)
    X_te, y_te = load_split(test_path)

    if use_val and os.path.exists(val_path):
        X_va, y_va = load_split(val_path)
        X_cal = X_tr + X_va
        y_cal = np.concatenate([y_tr, y_va], 0)
    else:
        X_cal, y_cal = X_tr, y_tr

    help_ent_scores, help_ppl_scores = collect_help_scores(X_cal, y_cal, ent_agg, ent_trim)
    if len(help_ent_scores) == 0 or len(help_ppl_scores) == 0:
        raise ValueError(f"Fold {k}: no HELP steps for calibration.")

    tau_ent = upper_tail_quantile(help_ent_scores, q=beta)
    tau_ppl = upper_tail_quantile(help_ppl_scores, q=beta)

    m = eval_with_thresholds(X_te, y_te, tau_ent, tau_ppl, ent_agg, ent_trim)
    m.update({
        "fold": k, "beta": beta, "mode": "fn",
        "tau_entropy": tau_ent, "tau_perplex": tau_ppl,
        "calib_help_steps_entropy": len(help_ent_scores),
        "calib_help_steps_perplex": len(help_ppl_scores),
    })
    return m

# ------------------------
# CLI
# ------------------------
def parse_args():
    ap = argparse.ArgumentParser(description="Conformal prediction over folds (entropy vs perplexity).")
    ap.add_argument("--config", required=False, help="YAML config. If given, overrides CLI defaults.")
    ap.add_argument("--data_root", default="/home/ulas/Downloads/processed_help_10fold_val")
    ap.add_argument("--folds", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--use_val_in_calib", type=int, default=1, help="1/0")
    ap.add_argument("--mode", choices=["fa","fn"], default="fn")
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.05, 0.10, 0.20])
    ap.add_argument("--betas",  type=float, nargs="+", default=[0.05, 0.10, 0.20])
    ap.add_argument("--ent_agg", choices=["p90","mean","trimmed_mean"], default="p90")
    ap.add_argument("--ent_trim", type=float, default=0.10)
    ap.add_argument("--out_dir", default=None, help="Default: <data_root>/cp_kfold_entropy_vs_ppl_<mode>")
    return ap.parse_args()

def main():
    args = parse_args()
    # load YAML if provided
    if args.config:
        cfg = yaml.safe_load(open(args.config))
    else:
        cfg = {}

    # merge precedence: YAML -> CLI defaults overridden by CLI flags if present
    data_root = cfg.get("data_root", args.data_root)
    folds     = cfg.get("folds", args.folds)
    use_val   = cfg.get("use_val_in_calib", bool(args.use_val_in_calib))
    mode      = cfg.get("mode", args.mode)
    alphas    = cfg.get("alphas", args.alphas)
    betas     = cfg.get("betas", args.betas)
    ent_agg   = cfg.get("ent_agg", args.ent_agg)
    ent_trim  = cfg.get("ent_trim", args.ent_trim)
    out_dir   = cfg.get("out_dir", args.out_dir or os.path.join(data_root, f"cp_kfold_entropy_vs_ppl_{mode}"))

    os.makedirs(out_dir, exist_ok=True)

    summary: Dict[str, List[Dict[str, float]]] = {}
    if mode == "fa":
        for alpha in alphas:
            fold_metrics: List[Dict[str, float]] = []
            for k in folds:
                try:
                    m = run_fold_fa(k, {
                        "data_root": data_root,
                        "use_val_in_calib": use_val,
                        "ent_agg": ent_agg,
                        "ent_trim": ent_trim,
                        "alpha": alpha
                    })
                    fold_metrics.append(m)
                    with open(os.path.join(out_dir, f"metrics_fold{k}_alpha{alpha:.2f}.json"), "w") as f:
                        json.dump(m, f, indent=2)
                    print(f"[FA α={alpha:.2f} | fold {k}] "
                          f"FA(ent)={m['false_ask_entropy']:.3f}  FA(ppl)={m['false_ask_perplex']:.3f} | "
                          f"Miss(ent)={m['miss_help_entropy']:.3f}  Miss(ppl)={m['miss_help_perplex']:.3f} | "
                          f"TP/FP/TN/FN(ent)={m['tp_entropy']}/{m['fp_entropy']}/{m['tn_entropy']}/{m['fn_entropy']}")
                except Exception as e:
                    print(f"[WARN] fold {k} failed: {e}")
            summary[f"alpha={alpha:.2f}"] = fold_metrics

    elif mode == "fn":
        for beta in betas:
            fold_metrics: List[Dict[str, float]] = []
            for k in folds:
                try:
                    m = run_fold_fn(k, {
                        "data_root": data_root,
                        "use_val_in_calib": use_val,
                        "ent_agg": ent_agg,
                        "ent_trim": ent_trim,
                        "beta": beta
                    })
                    fold_metrics.append(m)
                    with open(os.path.join(out_dir, f"metrics_fold{k}_beta{beta:.2f}.json"), "w") as f:
                        json.dump(m, f, indent=2)
                    print(f"[FN β={beta:.2f} | fold {k}] "
                          f"Miss(ent)={m['miss_help_entropy']:.3f}  Miss(ppl)={m['miss_help_perplex']:.3f} | "
                          f"FA(ent)={m['false_ask_entropy']:.3f}  FA(ppl)={m['false_ask_perplex']:.3f} | "
                          f"TP/FP/TN/FN(ent)={m['tp_entropy']}/{m['fp_entropy']}/{m['tn_entropy']}/{m['fn_entropy']}")
                except Exception as e:
                    print(f"[WARN] fold {k} failed: {e}")
            summary[f"beta={beta:.2f}"] = fold_metrics

    # Cross-fold aggregation
    report: Dict[str, Dict[str, object]] = {}
    for k_lbl, items in summary.items():
        stats = {
            "false_ask_entropy_mean_std": mean_std("false_ask_entropy", items),
            "false_ask_perplex_mean_std": mean_std("false_ask_perplex", items),
            "recall_entropy_mean_std":    mean_std("recall_entropy", items),
            "recall_perplex_mean_std":    mean_std("recall_perplex", items),
            "miss_help_entropy_mean_std": mean_std("miss_help_entropy", items),
            "miss_help_perplex_mean_std": mean_std("miss_help_perplex", items),
            "rate_entropy_mean_std":      mean_std("rate_entropy", items),
            "rate_perplex_mean_std":      mean_std("rate_perplex", items),
            "tau_entropy_mean_std":       mean_std("tau_entropy", items),
            "tau_perplex_mean_std":       mean_std("tau_perplex", items),
            "Ns_mean_std":                mean_std("Ns", items),
            "Nh_mean_std":                mean_std("Nh", items),
        }

        micro_ent = micro_confusion(items, which="entropy")
        micro_ppl = micro_confusion(items, which="perplex")
        report[k_lbl] = {
            **stats,
            "micro_confusion_entropy": micro_ent,
            "micro_metrics_entropy": derived_from_confusion(micro_ent),
            "micro_confusion_perplex": micro_ppl,
            "micro_metrics_perplex": derived_from_confusion(micro_ppl),
        }

    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(report, f, indent=2)

    print("\n=== CROSS-FOLD SUMMARY ===")
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()