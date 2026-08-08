"""Inference helpers for the strongly supervised INSIGHT help trigger.

The released ICRA 2026 checkpoints were trained on token features in the
historical implementation order:

    [aleatoric uncertainty, epistemic uncertainty, entropy, chosen-token log p]

The last feature was historically stored under the key ``perplexity`` even
though it contains the chosen token log-probability.  This module intentionally
preserves that contract so released checkpoints receive the same inputs they
were trained on.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import torch

from insight.models.single_transformer import SingleStepTransformer


@dataclass(frozen=True)
class TriggerDecision:
    """Result of applying INSIGHT Strong to one VLA inference step."""

    should_help: bool
    logit: float
    probability: float
    member_logits: tuple[float, ...]
    num_tokens: int


class StrongHelpTrigger:
    """Load one or more INSIGHT Strong checkpoints and score one VLA chunk.

    Multiple checkpoints are treated as a simple deployment ensemble by
    averaging their raw logits.  The paper's reported cross-validation metrics
    use the fold-matched checkpoint rather than this ensemble; averaging folds
    is provided only as a practical option for new real-robot states.
    """

    FEATURE_KEYS = ("au", "eu", "entropy", "perplexity")

    def __init__(
        self,
        checkpoint_paths: Sequence[str | Path],
        *,
        threshold_logit: float = 0.5,
        trim_head: int = 3,
        trim_tail: int = 2,
        device: str | torch.device | None = None,
    ) -> None:
        paths = [Path(p).expanduser() for p in checkpoint_paths]
        if not paths:
            raise ValueError("At least one Strong checkpoint is required.")
        missing = [str(p) for p in paths if not p.is_file()]
        if missing:
            raise FileNotFoundError(f"Missing checkpoint(s): {missing}")

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.threshold_logit = float(threshold_logit)
        self.trim_head = int(trim_head)
        self.trim_tail = int(trim_tail)
        self.checkpoint_paths = tuple(paths)
        self.models: list[SingleStepTransformer] = []

        for path in paths:
            model = SingleStepTransformer().to(self.device)
            try:
                state = torch.load(path, map_location=self.device, weights_only=True)
            except TypeError:  # Older PyTorch.
                state = torch.load(path, map_location=self.device)
            model.load_state_dict(state)
            model.eval()
            self.models.append(model)

    @classmethod
    def from_directory(
        cls,
        checkpoint_dir: str | Path,
        *,
        pattern: str = "single_fold*.pt",
        **kwargs,
    ) -> "StrongHelpTrigger":
        checkpoint_dir = Path(checkpoint_dir).expanduser()
        paths = sorted(checkpoint_dir.glob(pattern), key=lambda p: p.name)
        if not paths:
            raise FileNotFoundError(
                f"No checkpoints matching {pattern!r} under {checkpoint_dir}"
            )
        return cls(paths, **kwargs)

    def features_from_policy_output(self, output: Mapping[str, object]) -> np.ndarray:
        """Convert an uncertainty-enabled OpenPI response to ``[T, 4]`` features.

        Expected keys are ``au``, ``eu``, ``entropy`` and the historical
        ``perplexity`` field (which stores chosen-token log probability).
        The default 3-token head / 2-token tail trim exactly matches the data
        preparation used for the released experiments.
        """
        arrays: list[np.ndarray] = []
        for key in self.FEATURE_KEYS:
            if key not in output:
                raise KeyError(
                    f"Policy output is missing {key!r}. Got keys: {sorted(output.keys())}"
                )
            arrays.append(np.asarray(output[key], dtype=np.float32).reshape(-1))

        lengths = {len(a) for a in arrays}
        if len(lengths) != 1:
            raise ValueError(
                "Uncertainty arrays must be token-aligned; got lengths "
                f"{dict(zip(self.FEATURE_KEYS, map(len, arrays)))}"
            )

        n = lengths.pop()
        if n <= self.trim_head + self.trim_tail:
            raise ValueError(
                f"Only {n} generated tokens, too short for trim_head={self.trim_head} "
                f"and trim_tail={self.trim_tail}."
            )

        stop = n - self.trim_tail if self.trim_tail > 0 else n
        arrays = [a[self.trim_head:stop] for a in arrays]
        return np.stack(arrays, axis=-1).astype(np.float32, copy=False)

    @torch.no_grad()
    def predict_features(self, features: np.ndarray) -> TriggerDecision:
        features = np.asarray(features, dtype=np.float32)
        if features.ndim != 2 or features.shape[1] != 4:
            raise ValueError(f"Expected features [T,4], got {features.shape}")
        if features.shape[0] == 0:
            raise ValueError("Cannot score an empty token sequence.")

        x = torch.from_numpy(features).unsqueeze(0).to(self.device)
        tok_pad = torch.zeros((1, x.shape[1]), dtype=torch.bool, device=self.device)

        member_logits = tuple(float(model(x, tok_pad).item()) for model in self.models)
        logit = float(np.mean(member_logits))
        probability = float(1.0 / (1.0 + np.exp(-logit)))
        return TriggerDecision(
            should_help=logit >= self.threshold_logit,
            logit=logit,
            probability=probability,
            member_logits=member_logits,
            num_tokens=int(features.shape[0]),
        )

    def predict_policy_output(self, output: Mapping[str, object]) -> TriggerDecision:
        return self.predict_features(self.features_from_policy_output(output))


def resolve_checkpoints(paths: Iterable[str | Path]) -> list[Path]:
    """Resolve checkpoint files/directories into a deterministic path list."""
    out: list[Path] = []
    for raw in paths:
        path = Path(raw).expanduser()
        if path.is_dir():
            out.extend(sorted(path.glob("single_fold*.pt"), key=lambda p: p.name))
        else:
            out.append(path)
    return out
