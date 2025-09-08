import numpy as np

def shannon_entropy(probs: np.ndarray) -> float:
    """Shannon entropy over a 1D probability vector."""
    p = np.clip(probs, 1e-12, 1.0)
    return float(-(p * np.log(p)).sum())

def perplexity(nll: float) -> float:
    """Perplexity from negative log-likelihood (base e)."""
    return float(np.exp(nll))

def seq_aggregate(values, mask=None, mode="mean"):
    """Aggregate per-token values into a sequence score."""
    v = np.asarray(values)
    if mask is not None:
        v = v[np.asarray(mask).astype(bool)]
    if v.size == 0:
        return float("nan")
    if mode == "mean":
        return float(np.mean(v))
    if mode == "max":
        return float(np.max(v))
    if mode == "sum":
        return float(np.sum(v))
    raise ValueError(f"Unknown mode: {mode}")
