"""Score saved token-level INSIGHT features with Strong checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from insight.strong_trigger import StrongHelpTrigger


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True, help=".npy array with shape [T,4]")
    ap.add_argument("--checkpoint-dir", required=True)
    ap.add_argument("--threshold-logit", type=float, default=0.5)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    features = np.load(args.features)
    trigger = StrongHelpTrigger.from_directory(
        args.checkpoint_dir,
        threshold_logit=args.threshold_logit,
        device=args.device,
    )
    decision = trigger.predict_features(features)
    print(json.dumps({
        "should_help": decision.should_help,
        "logit": decision.logit,
        "probability": decision.probability,
        "member_logits": decision.member_logits,
        "num_tokens": decision.num_tokens,
    }, indent=2))


if __name__ == "__main__":
    main()
