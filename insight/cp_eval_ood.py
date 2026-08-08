#!/usr/bin/env python3
import os, json, argparse, csv, numpy as np, torch, yaml
from typing import List, Tuple, Dict, Any

# ========= Scoring (no calibration anywhere) =========
def _trimmed_mean(v: np.ndarray, trim: float) -> float:
    lo, hi = np.percentile(v, [100*trim, 100*(1-trim)])
    keep = v[(v >= lo) & (v <= hi)]
    return float(keep.mean()) if keep.size else float(v.mean())

def entropy_score(seq: np.ndarray, agg: str = "p90", trim: float = 0.10) -> float:
    # seq columns: [AU, EU, entropy, logp]
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

# ========= IO =========
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
    # (bags_X, bags_y, bags_src)
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

# ========= Pooling & step→episode flags =========
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

def step_flags_for_episode_metric(bag_steps: List[np.ndarray], tau: float, ent_agg: str, ent_trim: float, metric: str) -> np.ndarray:
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

# ========= Confusion & packing =========
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

# ========= Eval wrappers (fixed taus only) =========
def eval_step(X_test: List[np.ndarray], y_test: np.ndarray, tau_ent: float, tau_ppl: float, ent_agg: str, ent_trim: float):
    ent_scores = np.array([entropy_score(s, agg=ent_agg, trim=ent_trim) for s in X_test], dtype=np.float64)
    ppl_scores = np.array([perplexity_score(s) for s in X_test], dtype=np.float64)
    me = confusion_and_metrics(ent_scores > tau_ent, y_test)
    mp = confusion_and_metrics(ppl_scores > tau_ppl, y_test)
    return pack_dual_result(me, mp)

def eval_bag(X_bags: List[List[np.ndarray]], y_bag: np.ndarray, tau_ent: float, tau_ppl: float,
             ent_agg: str, ent_trim: float, bag_pool: str):
    ent_scores = np.array([bag_entropy_score(b, ent_agg, ent_trim, bag_pool) for b in X_bags], dtype=np.float64)
    ppl_scores = np.array([bag_perplexity_score(b, bag_pool) for b in X_bags], dtype=np.float64)
    me = confusion_and_metrics(ent_scores > tau_ent, y_bag)
    mp = confusion_and_metrics(ppl_scores > tau_ppl, y_bag)
    return pack_dual_result(me, mp)

def eval_step2bag(X_bags: List[List[np.ndarray]], y_bag: np.ndarray, tau_ent: float, tau_ppl: float,
                  ent_agg: str, ent_trim: float, episode_rule: str, episode_k: int):
    asks_ent = np.array([
        fuse_step_flags_to_episode(step_flags_for_episode_metric(b, tau_ent, ent_agg, ent_trim, "entropy"),
                                   episode_rule, episode_k)
        for b in X_bags
    ], dtype=bool)
    asks_ppl = np.array([
        fuse_step_flags_to_episode(step_flags_for_episode_metric(b, tau_ppl, ent_agg, ent_trim, "perplex"),
                                   episode_rule, episode_k)
        for b in X_bags
    ], dtype=bool)
    me = confusion_and_metrics(asks_ent, y_bag)
    mp = confusion_and_metrics(asks_ppl, y_bag)
    return pack_dual_result(me, mp)

# ========= Aggregation =========
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

# ========= OOD fold runner (NO calibration) =========
def run_fold_ood(k: int, cfg: Dict[str, Any]) -> Dict[str, float]:
    root_ood     = cfg["ood_root"]
    eval_on      = cfg["eval_on"]
    ent_agg      = cfg.get("ent_agg", "p90")
    ent_trim     = cfg.get("ent_trim", 0.10)
    bag_pool     = cfg.get("bag_pool", "max")
    tau_ent      = float(cfg["tau_entropy"])
    tau_ppl      = float(cfg["tau_perplex"])
    step2bag     = bool(cfg.get("step_to_bag_flags", True))
    episode_rule = cfg.get("episode_rule", "any")
    episode_k    = int(cfg.get("episode_k", 1))
    calib_on     = cfg.get("calib_on", "step")  # metadata only

    if eval_on == "step":
        te = os.path.join(root_ood, f"test_steps_{k}.pt")
        X_te, y_te = load_step_split(te)
        m = eval_step(X_te, y_te, tau_ent, tau_ppl, ent_agg, ent_trim)
    elif eval_on == "bag":
        te = os.path.join(root_ood, f"test_bags_{k}.pt")
        X_te, y_te = load_bag_split(te)
        if step2bag:
            m = eval_step2bag(X_te, y_te, tau_ent, tau_ppl, ent_agg, ent_trim, episode_rule, episode_k)
        else:
            m = eval_bag(X_te, y_te, tau_ent, tau_ppl, ent_agg, ent_trim, bag_pool)
    else:
        raise ValueError("eval_on must be 'step' or 'bag'")

    m.update({
        "fold": k,
        "eval_on": eval_on,
        "calib_on": calib_on,
        "tau_entropy": tau_ent,
        "tau_perplex": tau_ppl,
        "bag_pool": bag_pool,
        "episode_rule": episode_rule,
        "episode_k": episode_k,
        "step_to_bag_flags": step2bag
    })
    return m

# ========= CLI / Main =========
def parse_args():
    ap = argparse.ArgumentParser(description="OOD CP evaluation with FIXED thresholds (no recalibration).")
    ap.add_argument("--config", required=False, help="YAML config to override defaults.")

    # fixed thresholds (used for all folds)
    ap.add_argument("--tau_entropy", type=float, default=None, help="Required unless provided in YAML")
    ap.add_argument("--tau_perplex", type=float, default=None, help="Required unless provided in YAML")

    # data & control
    ap.add_argument("--ood_root", default="/path/to/processed_kfold_ood")
    ap.add_argument("--folds", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--calib_on", choices=["step","bag"], default="step", help="Metadata only")
    ap.add_argument("--eval_on",  choices=["step","bag"], default="bag")
    ap.add_argument("--ent_agg", choices=["p90","mean","trimmed_mean"], default="p90")
    ap.add_argument("--ent_trim", type=float, default=0.10)
    ap.add_argument("--bag_pool", choices=["max","mean","p90"], default="max")

    # step→bag flagging path (when eval_on=bag)
    ap.add_argument("--step_to_bag_flags", type=int, default=1, help="1: step flags + fusion, 0: pooled bag score vs tau")
    ap.add_argument("--episode_rule", choices=["any","kofn","majority"], default="any")
    ap.add_argument("--episode_k", type=int, default=1)

    ap.add_argument("--out_dir", default=None, help="Output dir for per-fold JSON, CSV, summary.json")
    return ap.parse_args()

def write_foldwise_csv(path: str, rows: List[Dict[str, Any]]):
    if not rows: return
    keys = sorted({k for r in rows for k in r.keys()})
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)

def main():
    args = parse_args()
    cfg = yaml.safe_load(open(args.config)) if args.config else {}

    # thresholds: MUST be provided (CLI or YAML)
    tau_entropy = args.tau_entropy if args.tau_entropy is not None else cfg.get("tau_entropy", None)
    tau_perplex = args.tau_perplex if args.tau_perplex is not None else cfg.get("tau_perplex", None)
    if tau_entropy is None or tau_perplex is None:
        raise ValueError("Provide tau_entropy and tau_perplex (e.g., cross-fold means from ID runs).")

    # merge config (no calibration settings here)
    ood_root     = cfg.get("ood_root", args.ood_root)
    folds        = cfg.get("folds", args.folds)
    calib_on     = cfg.get("calib_on", args.calib_on)     # metadata only
    eval_on      = cfg.get("eval_on", args.eval_on)
    ent_agg      = cfg.get("ent_agg", args.ent_agg)
    ent_trim     = cfg.get("ent_trim", args.ent_trim)
    bag_pool     = cfg.get("bag_pool", args.bag_pool)
    step2bag     = bool(cfg.get("step_to_bag_flags", args.step_to_bag_flags))
    episode_rule = cfg.get("episode_rule", args.episode_rule)
    episode_k    = int(cfg.get("episode_k", args.episode_k))
    out_dir      = cfg.get("out_dir", args.out_dir or os.path.join(ood_root, f"cp_ood_fixed_{calib_on}2{eval_on}"))

    os.makedirs(out_dir, exist_ok=True)

    per_fold: List[Dict[str, float]] = []
    csv_rows: List[Dict[str, Any]] = []

    for k in folds:
        try:
            m = run_fold_ood(k, {
                "ood_root": ood_root,
                "eval_on": eval_on,
                "calib_on": calib_on,
                "ent_agg": ent_agg,
                "ent_trim": ent_trim,
                "bag_pool": bag_pool,
                "tau_entropy": float(tau_entropy),
                "tau_perplex": float(tau_perplex),
                "step_to_bag_flags": step2bag,
                "episode_rule": episode_rule,
                "episode_k": episode_k
            })
            per_fold.append(m)
            tag = f"{calib_on}2{eval_on}"
            with open(os.path.join(out_dir, f"metrics_fold{k}_{tag}.json"), "w") as f:
                json.dump(m, f, indent=2)
            print(f"[OOD {tag} | fold {k}] "
                  f"TP/FP/TN/FN(ent)={m['tp_entropy']}/{m['fp_entropy']}/{m['tn_entropy']}/{m['fn_entropy']}  "
                  f"TP/FP/TN/FN(ppl)={m['tp_perplex']}/{m['fp_perplex']}/{m['tn_perplex']}/{m['fn_perplex']}")
            csv_rows.append({"setting": tag, **m})
        except Exception as e:
            print(f"[WARN][fold {k}] {e}")

    # aggregate to summary
    def mean_std_block(keys: List[str]) -> Dict[str, Tuple[float, float]]:
        return {f"{k}_mean_std": mean_std(k, per_fold) for k in keys}

    keys = [
        "false_ask_entropy","false_ask_perplex",
        "recall_entropy","recall_perplex",
        "miss_help_entropy","miss_help_perplex",
        "rate_entropy","rate_perplex",
        "accuracy_entropy","accuracy_perplex",
        "precision_entropy","precision_perplex",
        "recall_entropy","recall_perplex",
        "f1_entropy","f1_perplex",
        "Ns","Nh"
    ]
    stats = mean_std_block(keys)
    micro_ent = micro_confusion(per_fold, which="entropy")
    micro_ppl = micro_confusion(per_fold, which="perplex")

    summary = {
        "meta": {
            "ood_root": ood_root,
            "calib_on": calib_on,
            "eval_on": eval_on,
            "ent_agg": ent_agg,
            "ent_trim": ent_trim,
            "bag_pool": bag_pool,
            "tau_entropy": float(tau_entropy),
            "tau_perplex": float(tau_perplex),
            "episode_rule": episode_rule,
            "episode_k": episode_k,
            "step_to_bag_flags": step2bag
        },
        **stats,
        "micro_confusion_entropy": micro_ent,
        "micro_metrics_entropy": derived_from_confusion(micro_ent),
        "micro_confusion_perplex": micro_ppl,
        "micro_metrics_perplex": derived_from_confusion(micro_ppl),
        "per_fold": per_fold
    }

    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    write_foldwise_csv(os.path.join(out_dir, "foldwise.csv"), csv_rows)

    print("\n=== OOD SUMMARY (fixed taus) ===")
    print(json.dumps(summary, indent=2))
    print(f"\nSaved per-fold CSV → {os.path.join(out_dir, 'foldwise.csv')}")

if __name__ == "__main__":
    main()