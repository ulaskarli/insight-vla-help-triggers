#!/usr/bin/env python3
import os, json, math, argparse, numpy as np, torch, yaml
from typing import List, Tuple, Dict, Any

# =========================
# Core scoring
# =========================
def _trimmed_mean(v: np.ndarray, trim: float) -> float:
    lo, hi = np.percentile(v, [100*trim, 100*(1-trim)])
    keep = v[(v >= lo) & (v <= hi)]
    return float(keep.mean()) if keep.size else float(v.mean())

def entropy_score(seq: np.ndarray, agg: str = "p90", trim: float = 0.10) -> float:
    # columns: [AU, EU, entropy, logp]
    ent = seq[:, 2]
    if ent.size == 0: return float("nan")
    if agg == "mean":         return float(ent.mean())
    if agg == "p90":          return float(np.percentile(ent, 90))
    if agg == "trimmed_mean": return _trimmed_mean(ent, trim)
    raise ValueError(f"Unknown ENT_AGG: {agg}")

def perplexity_score(seq: np.ndarray) -> float:
    logp = seq[:, 3]
    if logp.size == 0: return float("nan")
    return float(np.exp(-np.mean(logp)))

def upper_tail_quantile(scores: List[float], q: float) -> float:
    if not scores: raise ValueError("No calibration scores.")
    s = np.sort(np.array(scores, dtype=np.float64))
    n = s.shape[0]
    k = int(math.ceil((n + 1) * q))
    k = max(1, min(k, n))
    return float(s[k - 1])

# =========================
# IO helpers
# =========================
def load_step_split(path: str) -> Tuple[List[np.ndarray], np.ndarray]:
    X, y = torch.load(path, weights_only=False)
    X_np = []
    for seq in X:
        arr = np.asarray(seq)
        if arr.ndim != 2 or arr.shape[1] < 4:
            raise ValueError(f"{path}: sequence has shape {arr.shape}, expected [T, >=4].")
        X_np.append(arr)
    y_np = np.asarray(y, dtype=np.int64)
    return X_np, y_np

def load_bag_split(path: str) -> Tuple[List[List[np.ndarray]], np.ndarray]:
    """(bags_X, bags_y, bags_src) where bags_X is List[List[np.ndarray[T,4]]]."""
    X, y, _src = torch.load(path, weights_only=False)
    bags = []
    for bag in X:
        bag_np = []
        for seq in bag:
            arr = np.asarray(seq)
            if arr.ndim != 2 or arr.shape[1] < 4:
                raise ValueError(f"{path}: step has shape {arr.shape}, expected [T, >=4].")
            bag_np.append(arr)
        bags.append(bag_np)
    y_np = np.asarray(y, dtype=np.int64)
    return bags, y_np

# =========================
# Pooling & flagging
# =========================
def pool_values(vals: List[float], mode: str = "max") -> float:
    v = np.array(vals, dtype=np.float64)
    if v.size == 0: return float("nan")
    if mode == "max":  return float(np.max(v))
    if mode == "mean": return float(np.mean(v))
    if mode == "p90":  return float(np.percentile(v, 90))
    raise ValueError(f"Unknown bag_pool: {mode}")

def bag_entropy_score(bag_steps: List[np.ndarray], ent_agg: str, ent_trim: float, bag_pool: str) -> float:
    step_scores = [entropy_score(seq, agg=ent_agg, trim=ent_trim) for seq in bag_steps]
    step_scores = [s for s in step_scores if not np.isnan(s)]
    return pool_values(step_scores, bag_pool)

def bag_perplexity_score(bag_steps: List[np.ndarray], bag_pool: str) -> float:
    step_scores = [perplexity_score(seq) for seq in bag_steps]
    step_scores = [s for s in step_scores if not np.isnan(s)]
    return pool_values(step_scores, bag_pool)

def step_flags_for_episode_metric(
    bag_steps: List[np.ndarray], tau: float, ent_agg: str, ent_trim: float, metric: str
) -> np.ndarray:
    flags = []
    for seq in bag_steps:
        if metric == "entropy":
            s = entropy_score(seq, agg=ent_agg, trim=ent_trim)
        elif metric == "perplex":
            s = perplexity_score(seq)
        else:
            raise ValueError("metric must be 'entropy' or 'perplex'")
        flags.append((s > tau) if not np.isnan(s) else False)
    return np.array(flags, dtype=bool)

def fuse_step_flags_to_episode(flags: np.ndarray, episode_rule: str = "any", k: int = 1) -> bool:
    n_pos, n = int(flags.sum()), len(flags)
    if episode_rule == "any":      return n_pos >= 1
    if episode_rule == "kofn":     return n_pos >= max(1, int(k))
    if episode_rule == "majority": return n_pos > (n // 2)
    raise ValueError(f"Unknown episode_rule: {episode_rule}")

# =========================
# Calibration collectors
# =========================
def collect_step_scores(
    X_steps: List[np.ndarray], y_steps: np.ndarray, target: int, ent_agg: str, ent_trim: float
) -> Tuple[List[float], List[float]]:
    """target=0 for SAFE, 1 for HELP (strong labels)."""
    ent_scores, ppl_scores = [], []
    for seq, lab in zip(X_steps, y_steps):
        if int(lab) == target:
            es = entropy_score(seq, agg=ent_agg, trim=ent_trim)
            ps = perplexity_score(seq)
            if not np.isnan(es): ent_scores.append(es)
            if not np.isnan(ps): ppl_scores.append(ps)
    return ent_scores, ppl_scores

def collect_bag_scores(
    bags_X: List[List[np.ndarray]], y_bag: np.ndarray, target: int, ent_agg: str, ent_trim: float, bag_pool: str
) -> Tuple[List[float], List[float]]:
    """target=0 for SAFE, 1 for HELP (weak labels at episode level)."""
    ent_scores, ppl_scores = [], []
    for bag, lab in zip(bags_X, y_bag):
        if int(lab) == target:
            es = bag_entropy_score(bag, ent_agg, ent_trim, bag_pool)
            ps = bag_perplexity_score(bag, bag_pool)
            if not np.isnan(es): ent_scores.append(es)
            if not np.isnan(ps): ppl_scores.append(ps)
    return ent_scores, ppl_scores

def collect_step_scores_from_bag_labels(
    X_bags: List[List[np.ndarray]], y_bag: np.ndarray, ent_agg: str, ent_trim: float
) -> Tuple[List[float], List[float], List[float], List[float]]:
    """Weak calibration at step-level by expanding bag labels to all steps."""
    safe_ent, safe_ppl, help_ent, help_ppl = [], [], [], []
    for bag, lab in zip(X_bags, y_bag):
        bucket_ent, bucket_ppl = (safe_ent, safe_ppl) if int(lab) == 0 else (help_ent, help_ppl)
        for seq in bag:
            es = entropy_score(seq, agg=ent_agg, trim=ent_trim)
            ps = perplexity_score(seq)
            if not np.isnan(es): bucket_ent.append(es)
            if not np.isnan(ps): bucket_ppl.append(ps)
    return safe_ent, safe_ppl, help_ent, help_ppl

# =========================
# Confusion & metrics
# =========================
def confusion_and_metrics(asks: np.ndarray, y_true: np.ndarray, positive: int = 1) -> Dict[str, float]:
    y = np.asarray(y_true, dtype=np.int64)
    help_mask = (y == positive)
    safe_mask = ~help_mask
    Ns, Nh = int(safe_mask.sum()), int(help_mask.sum())

    tp = int(np.logical_and(asks, help_mask).sum())
    fp = int(np.logical_and(asks, safe_mask).sum())
    tn = int(np.logical_and(~asks, safe_mask).sum())
    fn = int(np.logical_and(~asks, help_mask).sum())

    tot = tp + fp + tn + fn
    acc  = (tp + tn) / max(1, tot)
    prec = tp / max(1, tp + fp)
    rec  = tp / max(1, tp + fn)
    f1   = (2 * prec * rec) / max(1e-8, (prec + rec))
    fa   = fp / max(1, Ns) if Ns else np.nan
    miss = 1.0 - (tp / max(1, Nh)) if Nh else np.nan
    rate = float(np.mean(asks))

    return dict(tp=tp, fp=fp, tn=tn, fn=fn, accuracy=acc, precision=prec, recall=rec, f1=f1,
                false_ask=fa, miss_help=miss, rate=rate, Ns=Ns, Nh=Nh)

def pack_dual_result(me: Dict[str, float], mp: Dict[str, float]) -> Dict[str, float]:
    return {
        "tp_entropy": me["tp"], "fp_entropy": me["fp"], "tn_entropy": me["tn"], "fn_entropy": me["fn"],
        "false_ask_entropy": me["false_ask"], "miss_help_entropy": me["miss_help"], "rate_entropy": me["rate"],
        "accuracy_entropy": me["accuracy"], "precision_entropy": me["precision"], "recall_entropy": me["recall"], "f1_entropy": me["f1"],
        "tp_perplex": mp["tp"], "fp_perplex": mp["fp"], "tn_perplex": mp["tn"], "fn_perplex": mp["fn"],
        "false_ask_perplex": mp["false_ask"], "miss_help_perplex": mp["miss_help"], "rate_perplex": mp["rate"],
        "accuracy_perplex": mp["accuracy"], "precision_perplex": mp["precision"], "recall_perplex": mp["recall"], "f1_perplex": mp["f1"],
        "Ns": me["Ns"], "Nh": me["Nh"]
    }

# =========================
# Evaluation wrappers
# =========================
def eval_with_thresholds_step(
    X_test: List[np.ndarray], y_test: np.ndarray, tau_ent: float, tau_ppl: float, ent_agg: str, ent_trim: float
) -> Dict[str, float]:
    ent_scores = np.array([entropy_score(s, agg=ent_agg, trim=ent_trim) for s in X_test], dtype=np.float64)
    ppl_scores = np.array([perplexity_score(s) for s in X_test], dtype=np.float64)
    asks_ent = ent_scores > tau_ent
    asks_ppl = ppl_scores > tau_ppl
    me = confusion_and_metrics(asks_ent, y_test)
    mp = confusion_and_metrics(asks_ppl, y_test)
    return pack_dual_result(me, mp)

def eval_with_thresholds_bag(
    X_test_bags: List[List[np.ndarray]], y_test_bags: np.ndarray,
    tau_ent: float, tau_ppl: float, ent_agg: str, ent_trim: float, bag_pool: str
) -> Dict[str, float]:
    ent_scores = np.array([bag_entropy_score(b, ent_agg, ent_trim, bag_pool) for b in X_test_bags], dtype=np.float64)
    ppl_scores = np.array([bag_perplexity_score(b, bag_pool) for b in X_test_bags], dtype=np.float64)
    asks_ent = ent_scores > tau_ent
    asks_ppl = ppl_scores > tau_ppl
    me = confusion_and_metrics(asks_ent, y_test_bags)
    mp = confusion_and_metrics(asks_ppl, y_test_bags)
    return pack_dual_result(me, mp)

# =========================
# Fold runners
# =========================
# Step → Step
def run_fold_fa(k: int, cfg: Dict[str, Any]) -> Dict[str, float]:
    root     = cfg["data_root"]
    use_val  = cfg.get("use_val_in_calib", True)
    ent_agg  = cfg.get("ent_agg", "p90")
    ent_trim = cfg.get("ent_trim", 0.10)
    alpha    = cfg["alpha"]

    tr, te, va = [os.path.join(root, f"{pfx}_{k}.pt") for pfx in ("train_steps", "test_steps", "val_steps")]
    X_tr, y_tr = load_step_split(tr)
    X_te, y_te = load_step_split(te)

    if use_val and os.path.exists(va):
        X_va, y_va = load_step_split(va)
        X_cal, y_cal = X_tr + X_va, np.concatenate([y_tr, y_va], 0)
    else:
        X_cal, y_cal = X_tr, y_tr

    safe_ent, safe_ppl = collect_step_scores(X_cal, y_cal, target=0, ent_agg=ent_agg, ent_trim=ent_trim)
    if not safe_ent or not safe_ppl: raise ValueError(f"Fold {k}: no SAFE steps for calibration.")

    tau_ent = upper_tail_quantile(safe_ent, q=1.0 - alpha)
    tau_ppl = upper_tail_quantile(safe_ppl, q=1.0 - alpha)

    m = eval_with_thresholds_step(X_te, y_te, tau_ent, tau_ppl, ent_agg, ent_trim)
    m.update({"fold": k, "mode": "fa_step", "alpha": alpha, "tau_entropy": tau_ent, "tau_perplex": tau_ppl})
    return m

def run_fold_fn(k: int, cfg: Dict[str, Any]) -> Dict[str, float]:
    root     = cfg["data_root"]
    use_val  = cfg.get("use_val_in_calib", True)
    ent_agg  = cfg.get("ent_agg", "p90")
    ent_trim = cfg.get("ent_trim", 0.10)
    beta     = cfg["beta"]

    tr, te, va = [os.path.join(root, f"{pfx}_{k}.pt") for pfx in ("train_steps", "test_steps", "val_steps")]
    X_tr, y_tr = load_step_split(tr)
    X_te, y_te = load_step_split(te)

    if use_val and os.path.exists(va):
        X_va, y_va = load_step_split(va)
        X_cal, y_cal = X_tr + X_va, np.concatenate([y_tr, y_va], 0)
    else:
        X_cal, y_cal = X_tr, y_tr

    help_ent, help_ppl = collect_step_scores(X_cal, y_cal, target=1, ent_agg=ent_agg, ent_trim=ent_trim)
    if not help_ent or not help_ppl: raise ValueError(f"Fold {k}: no HELP steps for calibration.")

    tau_ent = upper_tail_quantile(help_ent, q=beta)
    tau_ppl = upper_tail_quantile(help_ppl, q=beta)

    m = eval_with_thresholds_step(X_te, y_te, tau_ent, tau_ppl, ent_agg, ent_trim)
    m.update({"fold": k, "mode": "fn_step", "beta": beta, "tau_entropy": tau_ent, "tau_perplex": tau_ppl})
    return m

# Bag → Bag
def run_fold_fa_bag(k: int, cfg: Dict[str, Any]) -> Dict[str, float]:
    root     = cfg["data_root"]
    use_val  = cfg.get("use_val_in_calib", True)
    ent_agg  = cfg.get("ent_agg", "p90")
    ent_trim = cfg.get("ent_trim", 0.10)
    bag_pool = cfg.get("bag_pool", "max")
    alpha    = cfg["alpha"]

    tr, te, va = [os.path.join(root, f"{pfx}_{k}.pt") for pfx in ("train_bags", "test_bags", "val_bags")]
    X_tr, y_tr = load_bag_split(tr)
    X_te, y_te = load_bag_split(te)
    if use_val and os.path.exists(va):
        X_va, y_va = load_bag_split(va)
        X_cal, y_cal = X_tr + X_va, np.concatenate([y_tr, y_va], 0)
    else:
        X_cal, y_cal = X_tr, y_tr

    safe_ent, safe_ppl = collect_bag_scores(X_cal, y_cal, target=0, ent_agg=ent_agg, ent_trim=ent_trim, bag_pool=bag_pool)
    if not safe_ent or not safe_ppl: raise ValueError(f"Fold {k}: no SAFE bags for calibration.")

    tau_ent = upper_tail_quantile(safe_ent, q=1.0 - alpha)
    tau_ppl = upper_tail_quantile(safe_ppl, q=1.0 - alpha)

    m = eval_with_thresholds_bag(X_te, y_te, tau_ent, tau_ppl, ent_agg, ent_trim, bag_pool)
    m.update({"fold": k, "mode": "fa_bag", "alpha": alpha, "tau_entropy": tau_ent, "tau_perplex": tau_ppl})
    return m

def run_fold_fn_bag(k: int, cfg: Dict[str, Any]) -> Dict[str, float]:
    root     = cfg["data_root"]
    use_val  = cfg.get("use_val_in_calib", True)
    ent_agg  = cfg.get("ent_agg", "p90")
    ent_trim = cfg.get("ent_trim", 0.10)
    bag_pool = cfg.get("bag_pool", "max")
    beta     = cfg["beta"]

    tr, te, va = [os.path.join(root, f"{pfx}_{k}.pt") for pfx in ("train_bags", "test_bags", "val_bags")]
    X_tr, y_tr = load_bag_split(tr)
    X_te, y_te = load_bag_split(te)
    if use_val and os.path.exists(va):
        X_va, y_va = load_bag_split(va)
        X_cal, y_cal = X_tr + X_va, np.concatenate([y_tr, y_va], 0)
    else:
        X_cal, y_cal = X_tr, y_tr

    help_ent, help_ppl = collect_bag_scores(X_cal, y_cal, target=1, ent_agg=ent_agg, ent_trim=ent_trim, bag_pool=bag_pool)
    if not help_ent or not help_ppl: raise ValueError(f"Fold {k}: no HELP bags for calibration.")

    tau_ent = upper_tail_quantile(help_ent, q=beta)
    tau_ppl = upper_tail_quantile(help_ppl, q=beta)

    m = eval_with_thresholds_bag(X_te, y_te, tau_ent, tau_ppl, ent_agg, ent_trim, bag_pool)
    m.update({"fold": k, "mode": "fn_bag", "beta": beta, "tau_entropy": tau_ent, "tau_perplex": tau_ppl})
    return m

# Step → Bag
def _step2bag_eval(X_te_bags, y_te_bag, tau_ent, tau_ppl, ent_agg, ent_trim, episode_rule, episode_k):
    # entropy
    asks_ent = np.array([
        fuse_step_flags_to_episode(step_flags_for_episode_metric(bag, tau_ent, ent_agg, ent_trim, "entropy"),
                                   episode_rule, episode_k)
        for bag in X_te_bags
    ], dtype=bool)
    # perplexity
    asks_ppl = np.array([
        fuse_step_flags_to_episode(step_flags_for_episode_metric(bag, tau_ppl, ent_agg, ent_trim, "perplex"),
                                   episode_rule, episode_k)
        for bag in X_te_bags
    ], dtype=bool)

    me = confusion_and_metrics(asks_ent, y_te_bag)
    mp = confusion_and_metrics(asks_ppl, y_te_bag)
    return pack_dual_result(me, mp)

def run_fold_fa_step2bag(k: int, cfg: Dict[str, Any]) -> Dict[str, float]:
    root        = cfg["data_root"]
    use_val     = cfg.get("use_val_in_calib", True)
    ent_agg     = cfg.get("ent_agg", "p90")
    ent_trim    = cfg.get("ent_trim", 0.10)
    alpha       = cfg["alpha"]
    episode_rule= cfg.get("episode_rule", "any")
    episode_k   = int(cfg.get("episode_k", 1))

    tr_s, te_b, va_s = [os.path.join(root, f"{pfx}_{k}.pt") for pfx in ("train_steps", "test_bags", "val_steps")]
    X_tr_s, y_tr_s = load_step_split(tr_s)
    if use_val and os.path.exists(va_s):
        X_va_s, y_va_s = load_step_split(va_s)
        X_cal_s, y_cal_s = X_tr_s + X_va_s, np.concatenate([y_tr_s, y_va_s], 0)
    else:
        X_cal_s, y_cal_s = X_tr_s, y_tr_s

    safe_ent, safe_ppl = collect_step_scores(X_cal_s, y_cal_s, target=0, ent_agg=ent_agg, ent_trim=ent_trim)
    if not safe_ent or not safe_ppl: raise ValueError(f"Fold {k}: no SAFE steps for calibration.")
    tau_ent = upper_tail_quantile(safe_ent, q=1.0 - alpha)
    tau_ppl = upper_tail_quantile(safe_ppl, q=1.0 - alpha)

    X_te_bags, y_te_bag = load_bag_split(te_b)
    m = _step2bag_eval(X_te_bags, y_te_bag, tau_ent, tau_ppl, ent_agg, ent_trim, episode_rule, episode_k)
    m.update({"fold": k, "mode": "fa_step2bag", "alpha": alpha, "tau_entropy": tau_ent, "tau_perplex": tau_ppl,
              "episode_rule": episode_rule, "episode_k": episode_k})
    return m

def run_fold_fn_step2bag(k: int, cfg: Dict[str, Any]) -> Dict[str, float]:
    root        = cfg["data_root"]
    use_val     = cfg.get("use_val_in_calib", True)
    ent_agg     = cfg.get("ent_agg", "p90")
    ent_trim    = cfg.get("ent_trim", 0.10)
    beta        = cfg["beta"]
    episode_rule= cfg.get("episode_rule", "any")
    episode_k   = int(cfg.get("episode_k", 1))

    tr_s, te_b, va_s = [os.path.join(root, f"{pfx}_{k}.pt") for pfx in ("train_steps", "test_bags", "val_steps")]
    X_tr_s, y_tr_s = load_step_split(tr_s)
    if use_val and os.path.exists(va_s):
        X_va_s, y_va_s = load_step_split(va_s)
        X_cal_s, y_cal_s = X_tr_s + X_va_s, np.concatenate([y_tr_s, y_va_s], 0)
    else:
        X_cal_s, y_cal_s = X_tr_s, y_tr_s

    help_ent, help_ppl = collect_step_scores(X_cal_s, y_cal_s, target=1, ent_agg=ent_agg, ent_trim=ent_trim)
    if not help_ent or not help_ppl: raise ValueError(f"Fold {k}: no HELP steps for calibration.")
    tau_ent = upper_tail_quantile(help_ent, q=beta)
    tau_ppl = upper_tail_quantile(help_ppl, q=beta)

    X_te_bags, y_te_bag = load_bag_split(te_b)
    m = _step2bag_eval(X_te_bags, y_te_bag, tau_ent, tau_ppl, ent_agg, ent_trim, episode_rule, episode_k)
    m.update({"fold": k, "mode": "fn_step2bag", "beta": beta, "tau_entropy": tau_ent, "tau_perplex": tau_ppl,
              "episode_rule": episode_rule, "episode_k": episode_k})
    return m

# Bag → Step
def run_fold_fa_bag2step(k: int, cfg: Dict[str, Any]) -> Dict[str, float]:
    root     = cfg["data_root"]
    use_val  = cfg.get("use_val_in_calib", True)
    ent_agg  = cfg.get("ent_agg", "p90")
    ent_trim = cfg.get("ent_trim", 0.10)
    alpha    = cfg["alpha"]

    tr_b, te_s, va_b = [os.path.join(root, f"{pfx}_{k}.pt") for pfx in ("train_bags", "test_steps", "val_bags")]
    X_tr_b, y_tr_b = load_bag_split(tr_b)
    if use_val and os.path.exists(va_b):
        X_va_b, y_va_b = load_bag_split(va_b)
        X_cal_b, y_cal_b = X_tr_b + X_va_b, np.concatenate([y_tr_b, y_va_b], 0)
    else:
        X_cal_b, y_cal_b = X_tr_b, y_tr_b

    safe_ent, safe_ppl, _, _ = collect_step_scores_from_bag_labels(X_cal_b, y_cal_b, ent_agg, ent_trim)
    if not safe_ent or not safe_ppl: raise ValueError(f"Fold {k}: no SAFE steps (weak) for calibration.")
    tau_ent = upper_tail_quantile(safe_ent, q=1.0 - alpha)
    tau_ppl = upper_tail_quantile(safe_ppl, q=1.0 - alpha)

    X_te_s, y_te_s = load_step_split(te_s)
    ent_scores = np.array([entropy_score(seq, agg=ent_agg, trim=ent_trim) for seq in X_te_s], dtype=np.float64)
    ppl_scores = np.array([perplexity_score(seq) for seq in X_te_s], dtype=np.float64)
    me = confusion_and_metrics(ent_scores > tau_ent, y_te_s)
    mp = confusion_and_metrics(ppl_scores > tau_ppl, y_te_s)

    m = pack_dual_result(me, mp)
    m.update({"fold": k, "mode": "fa_bag2step", "alpha": alpha, "tau_entropy": tau_ent, "tau_perplex": tau_ppl})
    return m

def run_fold_fn_bag2step(k: int, cfg: Dict[str, Any]) -> Dict[str, float]:
    root     = cfg["data_root"]
    use_val  = cfg.get("use_val_in_calib", True)
    ent_agg  = cfg.get("ent_agg", "p90")
    ent_trim = cfg.get("ent_trim", 0.10)
    beta     = cfg["beta"]

    tr_b, te_s, va_b = [os.path.join(root, f"{pfx}_{k}.pt") for pfx in ("train_bags", "test_steps", "val_bags")]
    X_tr_b, y_tr_b = load_bag_split(tr_b)
    if use_val and os.path.exists(va_b):
        X_va_b, y_va_b = load_bag_split(va_b)
        X_cal_b, y_cal_b = X_tr_b + X_va_b, np.concatenate([y_tr_b, y_va_b], 0)
    else:
        X_cal_b, y_cal_b = X_tr_b, y_tr_b

    _, _, help_ent, help_ppl = collect_step_scores_from_bag_labels(X_cal_b, y_cal_b, ent_agg, ent_trim)
    if not help_ent or not help_ppl: raise ValueError(f"Fold {k}: no HELP steps (weak) for calibration.")
    tau_ent = upper_tail_quantile(help_ent, q=beta)
    tau_ppl = upper_tail_quantile(help_ppl, q=beta)

    X_te_s, y_te_s = load_step_split(te_s)
    ent_scores = np.array([entropy_score(seq, agg=ent_agg, trim=ent_trim) for seq in X_te_s], dtype=np.float64)
    ppl_scores = np.array([perplexity_score(seq) for seq in X_te_s], dtype=np.float64)
    me = confusion_and_metrics(ent_scores > tau_ent, y_te_s)
    mp = confusion_and_metrics(ppl_scores > tau_ppl, y_te_s)

    m = pack_dual_result(me, mp)
    m.update({"fold": k, "mode": "fn_bag2step", "beta": beta, "tau_entropy": tau_ent, "tau_perplex": tau_ppl})
    return m

# =========================
# Aggregation helpers
# =========================
def mean_std(key: str, items: List[Dict[str, float]]) -> Tuple[float, float]:
    vals = [it.get(key) for it in items if key in it and it.get(key) is not None and not np.isnan(it.get(key))]
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
    acc  = (tp + tn) / max(1, tot)
    prec = tp / max(1, tp + fp)
    rec  = tp / max(1, tp + fn)
    f1   = (2 * prec * rec) / max(1e-8, (prec + rec))
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}

def pick_runner(mode: str, calib_on: str, eval_on: str):
    if mode == "fa":
        if   calib_on=="step" and eval_on=="step": return run_fold_fa
        elif calib_on=="step" and eval_on=="bag":  return run_fold_fa_step2bag
        elif calib_on=="bag"  and eval_on=="bag":  return run_fold_fa_bag
        elif calib_on=="bag"  and eval_on=="step": return run_fold_fa_bag2step
    elif mode == "fn":
        if   calib_on=="step" and eval_on=="step": return run_fold_fn
        elif calib_on=="step" and eval_on=="bag":  return run_fold_fn_step2bag
        elif calib_on=="bag"  and eval_on=="bag":  return run_fold_fn_bag
        elif calib_on=="bag"  and eval_on=="step": return run_fold_fn_bag2step
    raise ValueError("Invalid (mode, calib_on, eval_on) combination")

# =========================
# CLI / Main
# =========================
def parse_args():
    ap = argparse.ArgumentParser(description="Conformal prediction across folds (entropy vs perplexity).")
    ap.add_argument("--config", required=False, help="YAML config. If given, overrides CLI defaults.")

    # defaults / CLI
    ap.add_argument("--data_root", default="data/processed/strong_cv")
    ap.add_argument("--folds", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--use_val_in_calib", type=int, default=1, help="1/0")
    ap.add_argument("--mode", choices=["fa","fn"], default="fn")
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.05, 0.10, 0.20])
    ap.add_argument("--betas",  type=float, nargs="+", default=[0.05, 0.10, 0.20])
    ap.add_argument("--ent_agg", choices=["p90","mean","trimmed_mean"], default="p90")
    ap.add_argument("--ent_trim", type=float, default=0.10)
    ap.add_argument("--bag_pool", choices=["max","mean","p90"], default="max", help="Pooling for bag calibration/eval")
    ap.add_argument("--out_dir", default=None, help="Default: <data_root>/cp_kfold_entropy_vs_ppl_<mode>")

    # combo control
    ap.add_argument("--calib_on", choices=["step","bag"], default="step", help="Calibrate on strong step or weak bag")
    ap.add_argument("--eval_on",  choices=["step","bag"], default="step", help="Evaluate on strong step or weak bag")

    # step→bag fusion
    ap.add_argument("--episode_rule", choices=["any","kofn","majority"], default="any")
    ap.add_argument("--episode_k", type=int, default=1)
    return ap.parse_args()

def main():
    args = parse_args()
    cfg = yaml.safe_load(open(args.config)) if args.config else {}

    # merge (YAML overrides defaults; CLI flags override both where provided)
    data_root   = cfg.get("data_root", args.data_root)
    folds       = cfg.get("folds", args.folds)
    use_val     = cfg.get("use_val_in_calib", bool(args.use_val_in_calib))
    mode        = cfg.get("mode", args.mode)
    alphas      = cfg.get("alphas", args.alphas)
    betas       = cfg.get("betas", args.betas)
    ent_agg     = cfg.get("ent_agg", args.ent_agg)
    ent_trim    = cfg.get("ent_trim", args.ent_trim)
    bag_pool    = cfg.get("bag_pool", args.bag_pool)
    out_dir     = cfg.get("out_dir", args.out_dir or os.path.join(data_root, f"cp_kfold_entropy_vs_ppl_{mode}"))
    calib_on    = cfg.get("calib_on", args.calib_on)
    eval_on     = cfg.get("eval_on", args.eval_on)
    episode_rule= cfg.get("episode_rule", args.episode_rule)
    episode_k   = int(cfg.get("episode_k", args.episode_k))

    os.makedirs(out_dir, exist_ok=True)
    summary: Dict[str, List[Dict[str, float]]] = {}

    if mode == "fa":
        for alpha in alphas:
            fold_metrics = []
            for k in folds:
                try:
                    runner = pick_runner("fa", calib_on, eval_on)
                    m = runner(k, {
                        "data_root": data_root,
                        "use_val_in_calib": use_val,
                        "ent_agg": ent_agg,
                        "ent_trim": ent_trim,
                        "alpha": alpha,
                        "bag_pool": bag_pool,
                        "episode_rule": episode_rule,
                        "episode_k": episode_k,
                    })
                    fold_metrics.append(m)
                    with open(os.path.join(out_dir, f"metrics_fold{k}_alpha{alpha:.2f}.json"), "w") as f:
                        json.dump(m, f, indent=2)
                    print(f"[FA α={alpha:.2f} | fold {k}] "
                          f"TP/FP/TN/FN(ent)={m['tp_entropy']}/{m['fp_entropy']}/{m['tn_entropy']}/{m['fn_entropy']}  "
                          f"TP/FP/TN/FN(ppl)={m['tp_perplex']}/{m['fp_perplex']}/{m['tn_perplex']}/{m['fn_perplex']}")
                except Exception as e:
                    print(f"[WARN] fold {k} failed: {e}")
            summary[f"alpha={alpha:.2f}"] = fold_metrics

    elif mode == "fn":
        for beta in betas:
            fold_metrics = []
            for k in folds:
                try:
                    runner = pick_runner("fn", calib_on, eval_on)
                    m = runner(k, {
                        "data_root": data_root,
                        "use_val_in_calib": use_val,
                        "ent_agg": ent_agg,
                        "ent_trim": ent_trim,
                        "beta": beta,
                        "bag_pool": bag_pool,
                        "episode_rule": episode_rule,
                        "episode_k": episode_k,
                    })
                    fold_metrics.append(m)
                    with open(os.path.join(out_dir, f"metrics_fold{k}_beta{beta:.2f}.json"), "w") as f:
                        json.dump(m, f, indent=2)
                    print(f"[FN β={beta:.2f} | fold {k}] "
                          f"TP/FP/TN/FN(ent)={m['tp_entropy']}/{m['fp_entropy']}/{m['tn_entropy']}/{m['fn_entropy']}  "
                          f"TP/FP/TN/FN(ppl)={m['tp_perplex']}/{m['fp_perplex']}/{m['tn_perplex']}/{m['fn_perplex']}")
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