#!/usr/bin/env python3
import os, json, math, argparse, numpy as np, torch, yaml
from typing import List, Tuple, Dict, Optional

# ------------- Common scoring -------------
def _trimmed_mean(v: np.ndarray, trim: float) -> float:
    lo, hi = np.percentile(v, [100*trim, 100*(1-trim)])
    keep = v[(v >= lo) & (v <= hi)]
    return float(keep.mean()) if keep.size else float(v.mean())

def entropy_score(seq: np.ndarray, agg: str="p90", trim: float=0.10) -> float:
    ent = seq[:, 2]
    if ent.size == 0: return float("nan")
    if agg == "mean": return float(ent.mean())
    if agg == "p90":  return float(np.percentile(ent, 90))
    if agg == "trimmed_mean": return _trimmed_mean(ent, trim)
    raise ValueError(f"Unknown ent_agg={agg}")

def perplexity_score(seq: np.ndarray) -> float:
    logp = seq[:, 3]
    if logp.size == 0: return float("nan")
    return float(np.exp(-np.mean(logp)))

def pool_values(vals: List[float], mode: str="max") -> float:
    v = np.array(vals, dtype=np.float64)
    if v.size == 0: return float("nan")
    if mode == "max":  return float(np.max(v))
    if mode == "mean": return float(np.mean(v))
    if mode == "p90":  return float(np.percentile(v, 90))
    raise ValueError(f"Unknown bag_pool={mode}")

# ------------- IO loaders -------------
def load_step_split(path: str) -> Tuple[List[np.ndarray], np.ndarray]:
    X, y = torch.load(path, weights_only=False)
    X_np = []
    for seq in X:
        arr = np.asarray(seq)
        if arr.ndim != 2 or arr.shape[1] < 4:
            raise ValueError(f"{path}: seq has shape {arr.shape}, expected [T, >=4].")
        X_np.append(arr)
    y_np = np.asarray(y, dtype=np.int64)
    return X_np, y_np

def load_bag_split(path: str):
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
    y_np = np.asarray(y, dtype=np.float32)
    return bags, y_np

# ------------- Bag helpers -------------
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

def fuse_step_flags_to_episode(flags: np.ndarray, rule: str="any", k: int=1) -> bool:
    n_pos = int(flags.sum()); n = len(flags)
    if rule == "any":       return n_pos >= 1
    if rule == "kofn":      return n_pos >= max(1, int(k))
    if rule == "majority":  return n_pos > (n // 2)
    raise ValueError(f"Unknown episode_rule={rule}")

# ------------- Metrics -------------
def confusion_from_bool(asks: np.ndarray, y_pos_mask: np.ndarray, y_neg_mask: np.ndarray) -> Dict[str, float]:
    tp = int(np.logical_and(asks, y_pos_mask).sum())
    fp = int(np.logical_and(asks, y_neg_mask).sum())
    tn = int(np.logical_and(~asks, y_neg_mask).sum())
    fn = int(np.logical_and(~asks, y_pos_mask).sum())
    tot = tp + fp + tn + fn
    acc = (tp + tn) / max(1, tot)
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    f1 = (2 * prec * rec) / max(1e-8, (prec + rec))
    Ns = int(y_neg_mask.sum()); Nh = int(y_pos_mask.sum())
    fa = fp / max(1, Ns) if Ns else np.nan
    miss = 1.0 - (tp / max(1, Nh)) if Nh else np.nan
    rate = float(asks.mean())
    return {
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "accuracy": acc, "precision": prec, "recall": rec, "f1": f1,
        "false_ask": fa, "miss_help": miss, "rate": rate,
        "Ns": Ns, "Nh": Nh
    }

# ------------- Threshold loading -------------
def load_thresholds(thresholds_json: Optional[str], thresholds_dir: Optional[str], folds: List[int]):
    """
    Returns dict: fold -> {"tau_entropy":float, "tau_perplex":float, "space":"step"|"bag"}
    If thresholds come from earlier CP runs, the per-fold metric JSONs usually contain tau_* fields.
    Space inference:
      - If file has "mode" ending with "_step2bag" or contains "calib_*_steps_*" → space="step"
      - If it has "calib_*_bags_*" only → space="bag"
      - Else default to "step" (safer for step evals)
    """
    out = {}
    if thresholds_json:
        blob = json.load(open(thresholds_json))
        # supports either full mapping or a single dict (applied to all folds)
        if isinstance(blob, dict) and "tau_entropy" in blob:
            for k in folds:
                out[k] = {**blob}
        else:
            for k in folds:
                item = blob[str(k)] if str(k) in blob else blob.get(k)
                if item is None:
                    raise ValueError(f"Missing thresholds for fold {k} in {thresholds_json}")
                out[k] = dict(item)
        return out

    if thresholds_dir:
        for k in folds:
            # pick the first metrics file that contains this fold id
            candidates = [f for f in os.listdir(thresholds_dir)
                          if f"fold{k}" in f and f.endswith(".json")]
            if not candidates:
                raise ValueError(f"No threshold file for fold {k} in {thresholds_dir}")
            path = os.path.join(thresholds_dir, sorted(candidates)[0])
            m = json.load(open(path))
            tau_e = m.get("tau_entropy")
            tau_p = m.get("tau_perplex")
            if tau_e is None or tau_p is None:
                raise ValueError(f"{path} does not contain tau_* fields")
            # infer space
            space = "step"
            keys = list(m.keys())
            if any("calib_safe_bags" in k or "calib_help_bags" in k for k in keys):
                space = "bag"
            if isinstance(m.get("mode"), str) and m["mode"].endswith("_step2bag"):
                space = "step"
            out[k] = {"tau_entropy": float(tau_e), "tau_perplex": float(tau_p), "space": space}
        return out

    raise ValueError("Provide --thresholds_json or --thresholds_dir")

# ------------- Runners -------------
def eval_step(data_root_ood: str, fold: int, tau_e: float, tau_p: float, ent_agg: str, ent_trim: float):
    test_steps = os.path.join(data_root_ood, f"test_steps_{fold}.pt")
    X_te, y_te = load_step_split(test_steps)

    e_scores = np.array([entropy_score(s, agg=ent_agg, trim=ent_trim) for s in X_te], dtype=np.float64)
    p_scores = np.array([perplexity_score(s) for s in X_te], dtype=np.float64)

    asks_e = e_scores > tau_e
    asks_p = p_scores > tau_p

    safe_mask = (y_te == 0)
    help_mask = (y_te == 1)

    ent = confusion_from_bool(asks_e, help_mask, safe_mask)
    ppl = confusion_from_bool(asks_p, help_mask, safe_mask)
    return ent, ppl

def eval_bag_stepspace(
    data_root_ood: str, fold: int, tau_e: float, tau_p: float,
    ent_agg: str, ent_trim: float, episode_rule: str, episode_k: int
):
    test_bags = os.path.join(data_root_ood, f"test_bags_{fold}.pt")
    X_bags, y_bag = load_bag_split(test_bags)

    asks_e = []
    asks_p = []
    for bag in X_bags:
        fe = step_flags_for_episode_metric(bag, tau_e, ent_agg, ent_trim, metric="entropy")
        fp = step_flags_for_episode_metric(bag, tau_p, ent_agg, ent_trim, metric="perplex")
        asks_e.append(fuse_step_flags_to_episode(fe, episode_rule, episode_k))
        asks_p.append(fuse_step_flags_to_episode(fp, episode_rule, episode_k))
    asks_e = np.array(asks_e, dtype=bool)
    asks_p = np.array(asks_p, dtype=bool)

    safe_mask = (y_bag == 0.0)
    help_mask = (y_bag == 1.0)

    ent = confusion_from_bool(asks_e, help_mask, safe_mask)
    ppl = confusion_from_bool(asks_p, help_mask, safe_mask)
    return ent, ppl

def eval_bag_bagspace(
    data_root_ood: str, fold: int, tau_e: float, tau_p: float,
    ent_agg: str, ent_trim: float, bag_pool: str
):
    test_bags = os.path.join(data_root_ood, f"test_bags_{fold}.pt")
    X_bags, y_bag = load_bag_split(test_bags)

    e_scores = np.array([bag_entropy_score(b, ent_agg, ent_trim, bag_pool) for b in X_bags], dtype=np.float64)
    p_scores = np.array([bag_perplexity_score(b, bag_pool) for b in X_bags], dtype=np.float64)

    asks_e = e_scores > tau_e
    asks_p = p_scores > tau_p

    safe_mask = (y_bag == 0.0)
    help_mask = (y_bag == 1.0)

    ent = confusion_from_bool(asks_e, help_mask, safe_mask)
    ppl = confusion_from_bool(asks_p, help_mask, safe_mask)
    return ent, ppl

# ------------- CLI -------------
def parse_args():
    ap = argparse.ArgumentParser(description="Evaluate pre-calibrated CP thresholds on an OOD dataset.")
    ap.add_argument("--data_root_ood", required=True, help="Root of OOD test set (expects test_steps_{k}.pt / test_bags_{k}.pt)")
    ap.add_argument("--folds", type=int, nargs="+", default=list(range(10)))

    # thresholds
    ap.add_argument("--thresholds_json", default=None, help="JSON with per-fold thresholds or a single thresholds dict")
    ap.add_argument("--thresholds_dir", default=None, help="Directory containing per-fold metric JSON files (with tau_*)")

    # eval settings
    ap.add_argument("--eval_on", choices=["step","bag"], default="bag")
    ap.add_argument("--expected_space", choices=["auto","step","bag"], default="auto",
                    help="If not auto, enforce threshold space")

    # scoring options
    ap.add_argument("--ent_agg", choices=["p90","mean","trimmed_mean"], default="p90")
    ap.add_argument("--ent_trim", type=float, default=0.10)
    ap.add_argument("--bag_pool", choices=["max","mean","p90"], default="max")
    ap.add_argument("--episode_rule", choices=["any","kofn","majority"], default="any")
    ap.add_argument("--episode_k", type=int, default=1)

    # output
    ap.add_argument("--out_dir", required=True, help="Where to save per-fold results and cross-fold summary")
    return ap.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    th = load_thresholds(args.thresholds_json, args.thresholds_dir, args.folds)

    per_fold = []
    for k in args.folds:
        info = th[k]
        tau_e = float(info["tau_entropy"])
        tau_p = float(info["tau_perplex"])
        space = info.get("space", "step")

        # enforce space if requested
        if args.expected_space != "auto" and space != args.expected_space:
            raise ValueError(f"Fold {k}: thresholds space={space}, expected {args.expected_space}")

        if args.eval_on == "step":
            if space != "step":
                raise ValueError(f"Fold {k}: cannot eval_on=step with bag-space thresholds")
            ent, ppl = eval_step(args.data_root_ood, k, tau_e, tau_p, args.ent_agg, args.ent_trim)
        else:  # eval_on == "bag"
            if space == "step":
                ent, ppl = eval_bag_stepspace(
                    args.data_root_ood, k, tau_e, tau_p,
                    args.ent_agg, args.ent_trim, args.episode_rule, int(args.episode_k)
                )
            elif space == "bag":
                ent, ppl = eval_bag_bagspace(
                    args.data_root_ood, k, tau_e, tau_p,
                    args.ent_agg, args.ent_trim, args.bag_pool
                )
            else:
                raise ValueError(f"Fold {k}: unknown space={space}")

        out = {
            "fold": k,
            "space": space,
            "eval_on": args.eval_on,
            "tau_entropy": tau_e,
            "tau_perplex": tau_p,
            # entropy stream
            "tp_entropy": ent["tp"], "fp_entropy": ent["fp"], "tn_entropy": ent["tn"], "fn_entropy": ent["fn"],
            "accuracy_entropy": ent["accuracy"], "precision_entropy": ent["precision"],
            "recall_entropy": ent["recall"], "f1_entropy": ent["f1"],
            "false_ask_entropy": ent["false_ask"], "miss_help_entropy": ent["miss_help"], "rate_entropy": ent["rate"],
            "Ns": ent["Ns"], "Nh": ent["Nh"],
            # perplexity stream
            "tp_perplex": ppl["tp"], "fp_perplex": ppl["fp"], "tn_perplex": ppl["tn"], "fn_perplex": ppl["fn"],
            "accuracy_perplex": ppl["accuracy"], "precision_perplex": ppl["precision"],
            "recall_perplex": ppl["recall"], "f1_perplex": ppl["f1"],
            "false_ask_perplex": ppl["false_ask"], "miss_help_perplex": ppl["miss_help"], "rate_perplex": ppl["rate"],
        }
        per_fold.append(out)
        with open(os.path.join(args.out_dir, f"ood_metrics_fold{k}.json"), "w") as f:
            json.dump(out, f, indent=2)
        print(f"[OOD fold {k}] ENT tp/fp/tn/fn={out['tp_entropy']}/{out['fp_entropy']}/{out['tn_entropy']}/{out['fn_entropy']} | "
              f"PPL tp/fp/tn/fn={out['tp_perplex']}/{out['fp_perplex']}/{out['tn_perplex']}/{out['fn_perplex']}")

    # cross-fold summaries
    def micro(items, which):
        tp = sum(int(x[f"tp_{which}"]) for x in items)
        fp = sum(int(x[f"fp_{which}"]) for x in items)
        tn = sum(int(x[f"tn_{which}"]) for x in items)
        fn = sum(int(x[f"fn_{which}"]) for x in items)
        tot = tp + fp + tn + fn
        acc = (tp + tn) / max(1, tot)
        prec = tp / max(1, tp + fp)
        rec = tp / max(1, tp + fn)
        f1 = (2 * prec * rec) / max(1e-8, (prec + rec))
        return {"tp": tp, "fp": fp, "tn": tn, "fn": fn, "accuracy": acc, "precision": prec, "recall": rec, "f1": f1}

    def mean_std(key, arr):
        vals = [x[key] for x in arr if key in x and not np.isnan(x[key])]
        return {"mean": float(np.mean(vals)) if vals else float("nan"),
                "std":  float(np.std(vals))  if vals else float("nan")}

    report = {
        "eval_on": args.eval_on,
        "folds": args.folds,
        "entropy_micro": micro(per_fold, "entropy"),
        "perplex_micro": micro(per_fold, "perplex"),
        "entropy_false_ask_mean_std": mean_std("false_ask_entropy", per_fold),
        "entropy_miss_help_mean_std": mean_std("miss_help_entropy", per_fold),
        "entropy_rate_mean_std":      mean_std("rate_entropy", per_fold),
        "perplex_false_ask_mean_std": mean_std("false_ask_perplex", per_fold),
        "perplex_miss_help_mean_std": mean_std("miss_help_perplex", per_fold),
        "perplex_rate_mean_std":      mean_std("rate_perplex", per_fold),
    }
    with open(os.path.join(args.out_dir, "ood_summary.json"), "w") as f:
        json.dump(report, f, indent=2)
    print("\n=== OOD SUMMARY ===")
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()