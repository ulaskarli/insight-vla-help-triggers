#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, re, json, argparse
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import torch
from sklearn.model_selection import KFold, StratifiedShuffleSplit
from numpy.random import default_rng

# ---------------------------
# Utilities
# ---------------------------
def natural_key(s: str) -> Tuple:
    return tuple(int(t) if t.isdigit() else t for t in re.findall(r"\d+|\D+", s))

def rng_for(seed:int, random_state:int, fold:int, extra:int=0):
    return default_rng(seed + random_state*100 + fold*10 + extra)

def read_rollout_results(path: str) -> Dict[str, int]:
    """
    Map episode_name -> bag label (0=success, 1=failure).
    File lines like:  <episode_name> : Success   OR   <episode_name> : Failure
    """
    if not path or not os.path.isfile(path):
        raise FileNotFoundError("rollout_results.txt is required for weak-only datasets.")
    mapping: Dict[str, int] = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"): continue
            parts = re.split(r"[:\-\s]\s*", line)
            ep = parts[0].strip()
            status = parts[-1].strip().upper()
            if status.startswith("F"): mapping[ep] = 1
            elif status.startswith("S"): mapping[ep] = 0
            else:
                print(f"[WARN] Unrecognized status in line: {line!r} (skipping).")
    return mapping

def list_episode_dirs(base_dir: str, ignore_dirs: List[str]) -> List[str]:
    dirs = []
    for name in os.listdir(base_dir):
        p = os.path.join(base_dir, name)
        if os.path.isdir(p) and name not in ignore_dirs:
            # weak-only: accept any dir that has at least one step_*.npy
            has_step = any(f.endswith(".npy") and f.startswith("step_") for f in os.listdir(p))
            if has_step:
                dirs.append(p)
    dirs.sort(key=lambda p: natural_key(os.path.basename(p)))
    return dirs

# ---------------------------
# Loading and trimming per-step token sequences
# ---------------------------
def load_step_timeseries(step_path: str, trim_head:int, trim_tail:int, min_len_after:int) -> Optional[np.ndarray]:
    """
    Each step .npy -> dict with "outputs/au", "outputs/eu", "outputs/entropy", "outputs/perplexity".
    Returns [T,4] float32 (AU, EU, entropy, perplexity) after trimming.
    """
    try:
        data = np.load(step_path, allow_pickle=True).item()
        au  = np.asarray(data["outputs/au"])
        eu  = np.asarray(data["outputs/eu"])
        ent = np.asarray(data["outputs/entropy"])
        ppl = np.asarray(data["outputs/perplexity"])
    except Exception as e:
        print(f"[WARN] Could not load {step_path}: {e}")
        return None
    if len(au) <= (trim_head + trim_tail):
        return None
    sl = slice(trim_head, len(au) - trim_tail)
    au, eu, ent, ppl = au[sl], eu[sl], ent[sl], ppl[sl]
    if len(au) < min_len_after:
        return None
    return np.stack([au, eu, ent, ppl], axis=-1).astype(np.float32)

def load_episode_weak(ep_dir: str,
                      ep_label_map: Dict[str, int],
                      trim_head:int, trim_tail:int, min_len_after:int) -> tuple[List[np.ndarray], int, str]:
    """
    Weak-only: no labels.json. Gather step_*.npy (sorted), return (steps, bag_label, name).
    """
    name = os.path.basename(ep_dir)
    if name not in ep_label_map:
        raise KeyError(f"Episode '{name}' missing in rollout_results.txt")
    bag_label = int(ep_label_map[name])

    step_files = sorted(
        [os.path.join(ep_dir, f) for f in os.listdir(ep_dir) if f.endswith(".npy") and f.startswith("step_")],
        key=lambda p: natural_key(os.path.basename(p))
    )
    steps: List[np.ndarray] = []
    for fp in step_files:
        arr = load_step_timeseries(fp, trim_head, trim_tail, min_len_after)
        if arr is not None:
            steps.append(arr)

    if len(steps) == 0:
        print(f"[WARN] Episode {name} has no usable steps after trimming (skipping).")
        return [], bag_label, name

    return steps, bag_label, name

# ---------------------------
# Uncertainty heuristics (anchors) + mixing (unchanged)
# ---------------------------
def step_score_uncert(step_arr: np.ndarray) -> float:
    """score = max(max(entropy), max(perplexity)) on [T,4]."""
    ent = step_arr[:, 2]
    ppl = step_arr[:, 3]
    return float(max(ent.max(initial=-1e9), ppl.max(initial=-1e9)))

def compute_anchor_indices(steps: List[np.ndarray], top_pct: float, min_anchors:int) -> List[int]:
    S = len(steps)
    if S == 0: return []
    scores = np.array([step_score_uncert(s) for s in steps], dtype=np.float32)
    k = max(min_anchors, int(np.ceil(S * top_pct)))
    k = min(k, S)
    order = np.argsort(-scores)  # descending
    return sorted(order[:k].tolist())

def pick_subseq_bounds(rng, length:int, min_len:int, max_len:int,
                       must_include: Optional[List[int]]=None,
                       must_exclude: Optional[List[int]]=None):
    min_len = max(1, min_len)
    max_len = max(min_len, max_len)
    L = length
    for _ in range(50):
        win_len = int(rng.integers(min_len, min(max_len, L)+1))
        start = 0 if L == win_len else int(rng.integers(0, L - win_len + 1))
        end = start + win_len
        idxs = set(range(start, end))
        ok_inc = True
        ok_exc = True
        if must_include:
            ok_inc = any((i in idxs) for i in must_include)
        if must_exclude:
            ok_exc = all((i not in idxs) for i in must_exclude)
        if ok_inc and ok_exc:
            return start, end
    win_len = int(rng.integers(min_len, min(max_len, L)+1))
    start = 0 if L == win_len else int(rng.integers(0, L - win_len + 1))
    return start, start + win_len

def maybe_mix_success_steps(rng, bag_steps: List[np.ndarray], pool_success_steps: List[List[np.ndarray]],
                            mix_prob: float, mix_max_steps: int) -> List[np.ndarray]:
    if mix_prob <= 0 or mix_max_steps <= 0: return bag_steps
    if rng.random() > mix_prob: return bag_steps
    if len(pool_success_steps) == 0: return bag_steps
    donor = pool_success_steps[int(rng.integers(0, len(pool_success_steps)))]
    if len(donor) == 0: return bag_steps
    k = int(rng.integers(1, min(mix_max_steps, len(donor)) + 1))
    idxs = rng.choice(len(donor), size=k, replace=False)
    return bag_steps + [donor[i] for i in idxs]

def build_augmented_bags_for_split(
    X_eps: List[List[np.ndarray]],
    y_eps: np.ndarray,
    names: List[str],
    fold: int,
    seed:int,
    random_state:int,
    # per-episode generation controls
    success_bags_per_ep: int,
    fail_pos_bags_per_ep: int,
    fail_neg_bags_per_ep: int,
    # subseq lengths
    success_min:int, success_max:int,
    fail_min:int, fail_max:int,
    # anchors
    top_pct: float, min_anchors:int,
    # mixing
    mix_prob_success: float, mix_max_success:int,
    mix_prob_fail: float, mix_max_fail:int,
    use_mixing: bool,
) -> tuple[List[List[np.ndarray]], np.ndarray, List[str]]:
    rng = rng_for(seed, random_state, fold, extra=7)
    success_pool = [steps for steps,y in zip(X_eps, y_eps.tolist()) if y==0]

    bags_X: List[List[np.ndarray]] = []
    bags_y: List[int] = []
    bags_src: List[str] = []

    for steps, y, nm in zip(X_eps, y_eps.tolist(), names):
        S = len(steps)
        if S == 0: continue
        if y == 0:
            for b in range(success_bags_per_ep):
                st, en = pick_subseq_bounds(rng, S, success_min, min(success_max, S))
                bag = steps[st:en]
                if use_mixing:
                    bag = maybe_mix_success_steps(rng, bag, success_pool, mix_prob_success, mix_max_success)
                bags_X.append(bag); bags_y.append(0); bags_src.append(f"{nm}#neg{b}")
        else:
            anchors = compute_anchor_indices(steps, top_pct=top_pct, min_anchors=min_anchors)
            for b in range(fail_pos_bags_per_ep):
                st, en = pick_subseq_bounds(rng, S, fail_min, min(fail_max, S), must_include=anchors, must_exclude=None)
                bag = steps[st:en]
                if use_mixing:
                    bag = maybe_mix_success_steps(rng, bag, success_pool, mix_prob_fail, mix_max_fail)
                bags_X.append(bag); bags_y.append(1); bags_src.append(f"{nm}#pos{b}")
            if fail_neg_bags_per_ep > 0 and len(anchors) > 0:
                for b in range(fail_neg_bags_per_ep):
                    st, en = pick_subseq_bounds(rng, S, fail_min, min(fail_max, S), must_include=None, must_exclude=anchors)
                    bag = steps[st:en]
                    if use_mixing:
                        bag = maybe_mix_success_steps(rng, bag, success_pool, mix_prob_fail, mix_max_fail)
                    bags_X.append(bag); bags_y.append(0); bags_src.append(f"{nm}#negNoAnchor{b}")

    return bags_X, np.array(bags_y, dtype=np.int64), bags_src

def build_deterministic_bags(X_eps: List[List[np.ndarray]], y_eps: np.ndarray, names: List[str]):
    """Each episode -> one bag (all steps)."""
    bags_X, bags_y, bags_src = [], [], []
    for steps, y, nm in zip(X_eps, y_eps.tolist(), names):
        bags_X.append(steps[:]); bags_y.append(int(y)); bags_src.append(f"{nm}#all")
    return bags_X, np.array(bags_y, dtype=np.int64), bags_src

# ---------------------------
# CLI
# ---------------------------
def parse_args():
    ap = argparse.ArgumentParser(description="Build K-fold MIL bag datasets (weak labels only).")
    # Paths
    ap.add_argument("--base_dir", required=True, help="Folder containing episode subdirectories (only step_*.npy)")
    ap.add_argument("--rollout_results", required=True, help="Path to rollout_results.txt mapping episode -> Success/Failure")
    ap.add_argument("--out_dir", required=True, help="Output directory for processed packs")
    ap.add_argument("--ignore_dirs", nargs="*", default=["processed_episode_mil_10fold", "processed_help_10fold_val", "processed_kfold"])
    # Splits
    ap.add_argument("--n_splits", type=int, default=10)
    ap.add_argument("--random_state", type=int, default=42)
    ap.add_argument("--val_frac", type=float, default=0.12)
    ap.add_argument("--stratify_by_task", type=int, default=0, help="1 to stratify train/val by (task, label)")
    # Trimming
    ap.add_argument("--trim_head", type=int, default=3)
    ap.add_argument("--trim_tail", type=int, default=2)
    ap.add_argument("--min_len_after_trim", type=int, default=1)
    # Bag augmentation toggle
    ap.add_argument("--augment_bags", type=int, default=1, help="1 enables augmentation; 0 = one bag per episode")
    # Random seed for augmentation
    ap.add_argument("--seed", type=int, default=1337)
    # Augmentation params (train)
    ap.add_argument("--success_bags_per_ep", type=int, default=3)
    ap.add_argument("--success_subseq_min_steps", type=int, default=4)
    ap.add_argument("--success_subseq_max_steps", type=int, default=12)
    ap.add_argument("--success_mix_prob", type=float, default=0.35)
    ap.add_argument("--success_mix_max_steps", type=int, default=4)
    ap.add_argument("--fail_pos_bags_per_ep", type=int, default=4)
    ap.add_argument("--fail_neg_bags_per_ep", type=int, default=0)
    ap.add_argument("--fail_subseq_min_steps", type=int, default=4)
    ap.add_argument("--fail_subseq_max_steps", type=int, default=12)
    ap.add_argument("--fail_mix_prob", type=float, default=0.0)
    ap.add_argument("--fail_mix_max_steps", type=int, default=0)
    ap.add_argument("--anchor_top_pct", type=float, default=0.20)
    ap.add_argument("--anchor_min_count", type=int, default=1)
    # Val augmentation (light / off)
    ap.add_argument("--val_success_bags_per_ep", type=int, default=1)
    ap.add_argument("--val_fail_pos_bags_per_ep", type=int, default=1)
    ap.add_argument("--val_fail_neg_bags_per_ep", type=int, default=0)
    ap.add_argument("--val_use_mixing", type=int, default=0)
    return ap.parse_args()

# ---------------------------
# Main
# ---------------------------
def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    ep_label_map = read_rollout_results(args.rollout_results)
    ep_dirs = list_episode_dirs(args.base_dir, ignore_dirs=args.ignore_dirs)

    # Load episodes (weak)
    X_eps, y_eps, names = [], [], []
    for ep_dir in ep_dirs:
        try:
            steps, bag_y, name = load_episode_weak(
                ep_dir, ep_label_map,
                trim_head=args.trim_head, trim_tail=args.trim_tail, min_len_after=args.min_len_after_trim
            )
        except KeyError as e:
            print(f"[WARN] {e} (skipping)")
            continue
        if len(steps) == 0:
            continue
        X_eps.append(steps); y_eps.append(bag_y); names.append(name)

    n = len(X_eps)
    if n == 0:
        print("No episodes found with usable steps. Exiting."); return

    y_arr = np.array(y_eps, dtype=np.int64)
    print(f"✅ Loaded {n} episodes from {args.base_dir}")
    print(f"   Failure ratio (episodes): {y_arr.mean():.3f}")

    # K-fold
    kf = KFold(n_splits=args.n_splits, shuffle=True, random_state=args.random_state)

    for fold, (tr_idx, te_idx) in enumerate(kf.split(range(n))):
        print(f"\n🔁 Fold {fold+1}/{args.n_splits}")

        def subset(ix):
            X = [X_eps[i] for i in ix]
            y = np.array([y_eps[i] for i in ix], dtype=np.int64)
            nm = [names[i] for i in ix]
            return X, y, nm

        X_te, y_te, nm_te = subset(te_idx)
        X_tr_all, y_tr_all, nm_tr_all = subset(tr_idx)

        # Train/Val split from outer train
        if args.stratify_by_task:
            tr_tasks = np.array(["_".join(names[i].split("_")[:2]) for i in tr_idx])
            strat_keys = np.array([f"{y}_{t}" for y, t in zip(y_tr_all, tr_tasks)])
        else:
            strat_keys = y_tr_all

        if args.val_frac > 0.0 and len(np.unique(strat_keys)) > 1:
            sss = StratifiedShuffleSplit(n_splits=1, test_size=args.val_frac,
                                         random_state=args.random_state + fold)
            tr_local, va_local = next(sss.split(np.arange(len(tr_idx)), strat_keys))
            tr_ids = [tr_idx[i] for i in tr_local]
            va_ids = [tr_idx[i] for i in va_local]
        else:
            tr_ids, va_ids = tr_idx, []

        X_tr = [X_eps[i] for i in tr_ids]
        y_tr = np.array([y_eps[i] for i in tr_ids], dtype=np.int64)
        nm_tr= [names[i] for i in tr_ids]

        X_va, y_va, nm_va = [], np.array([]), []
        if len(va_ids) > 0:
            X_va = [X_eps[i] for i in va_ids]
            y_va = np.array([y_eps[i] for i in va_ids], dtype=np.int64)
            nm_va= [names[i] for i in va_ids]

        print(f"   Episodes -> Train {len(X_tr)} | Val {len(X_va)} | Test {len(X_te)}")
        if len(y_tr) > 0:
            msg = f"   Episode fail ratio: train {y_tr.mean():.3f}"
            if len(y_va) > 0: msg += f" | val {y_va.mean():.3f}"
            msg += f" | test {y_te.mean():.3f}"
            print(msg)

        # ===== Save EPISODE packs (reference, optional) =====
        torch.save((X_tr, y_tr, nm_tr), os.path.join(args.out_dir, f"train_{fold}.pt"))
        if len(X_va) > 0:
            torch.save((X_va, y_va, nm_va), os.path.join(args.out_dir, f"val_{fold}.pt"))
        torch.save((X_te, y_te, nm_te), os.path.join(args.out_dir, f"test_{fold}.pt"))

        # ===== BAG DATASETS =====
        if args.augment_bags:
            # Train bags (augmented)
            if len(X_tr) > 0:
                tr_bags_X, tr_bags_y, tr_bags_src = build_augmented_bags_for_split(
                    X_tr, y_tr, nm_tr, fold=fold, seed=args.seed, random_state=args.random_state,
                    success_bags_per_ep=args.success_bags_per_ep,
                    fail_pos_bags_per_ep=args.fail_pos_bags_per_ep,
                    fail_neg_bags_per_ep=args.fail_neg_bags_per_ep,
                    success_min=args.success_subseq_min_steps,
                    success_max=args.success_subseq_max_steps,
                    fail_min=args.fail_subseq_min_steps,
                    fail_max=args.fail_subseq_max_steps,
                    top_pct=args.anchor_top_pct, min_anchors=args.anchor_min_count,
                    mix_prob_success=args.success_mix_prob, mix_max_success=args.success_mix_max_steps,
                    mix_prob_fail=args.fail_mix_prob, mix_max_fail=args.fail_mix_max_steps,
                    use_mixing=True,
                )
                torch.save((tr_bags_X, tr_bags_y, tr_bags_src), os.path.join(args.out_dir, f"train_bags_{fold}.pt"))
                print(f"   💾 train_bags_{fold}.pt  bags={len(tr_bags_y)}  pos={int(tr_bags_y.sum())}  neg={int((tr_bags_y==0).sum())}")
            # Val bags (light augmentation or off)
            if len(X_va) > 0:
                va_bags_X, va_bags_y, va_bags_src = build_augmented_bags_for_split(
                    X_va, y_va, nm_va, fold=fold, seed=args.seed, random_state=args.random_state,
                    success_bags_per_ep=args.val_success_bags_per_ep,
                    fail_pos_bags_per_ep=args.val_fail_pos_bags_per_ep,
                    fail_neg_bags_per_ep=args.val_fail_neg_bags_per_ep,
                    success_min=args.success_subseq_min_steps,
                    success_max=args.success_subseq_max_steps,
                    fail_min=args.fail_subseq_min_steps,
                    fail_max=args.fail_subseq_max_steps,
                    top_pct=args.anchor_top_pct, min_anchors=args.anchor_min_count,
                    mix_prob_success=(args.success_mix_prob if args.val_use_mixing else 0.0),
                    mix_max_success=(args.success_mix_max_steps if args.val_use_mixing else 0),
                    mix_prob_fail=(args.fail_mix_prob if args.val_use_mixing else 0.0),
                    mix_max_fail=(args.fail_mix_max_steps if args.val_use_mixing else 0),
                    use_mixing=bool(args.val_use_mixing),
                )
                torch.save((va_bags_X, va_bags_y, va_bags_src), os.path.join(args.out_dir, f"val_bags_{fold}.pt"))
                print(f"   💾 val_bags_{fold}.pt    bags={len(va_bags_y)}  pos={int(va_bags_y.sum())}  neg={int((va_bags_y==0).sum())}")
        else:
            # No augmentation: one bag per episode
            tr_bags_X, tr_bags_y, tr_bags_src = build_deterministic_bags(X_tr, y_tr, nm_tr)
            torch.save((tr_bags_X, tr_bags_y, tr_bags_src), os.path.join(args.out_dir, f"train_bags_{fold}.pt"))
            print(f"   💾 train_bags_{fold}.pt  bags={len(tr_bags_y)}  pos={int(tr_bags_y.sum())}  neg={int((tr_bags_y==0).sum())}")
            if len(X_va) > 0:
                va_bags_X, va_bags_y, va_bags_src = build_deterministic_bags(X_va, y_va, nm_va)
                torch.save((va_bags_X, va_bags_y, va_bags_src), os.path.join(args.out_dir, f"val_bags_{fold}.pt"))
                print(f"   💾 val_bags_{fold}.pt    bags={len(va_bags_y)}  pos={int(va_bags_y.sum())}  neg={int((va_bags_y==0).sum())}")

        # Test bags (deterministic, one per episode)
        te_bags_X, te_bags_y, te_bags_src = build_deterministic_bags(X_te, y_te, nm_te)
        torch.save((te_bags_X, te_bags_y, te_bags_src), os.path.join(args.out_dir, f"test_bags_{fold}.pt"))
        print(f"   💾 test_bags_{fold}.pt   bags={len(te_bags_y)}  pos={int(te_bags_y.sum())}  neg={int((te_bags_y==0).sum())}")

    print(f"\n📦 Done! Saved to '{args.out_dir}'")

if __name__ == "__main__":
    main()