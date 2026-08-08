# Data

Large robot rollouts and processed fold files are intentionally not stored in
normal Git history.

See `../docs/DATA_FORMAT.md` for the exact token-feature representation and fold
file formats expected by the training/evaluation scripts.

If processed paper data is released, place or link it under a versioned layout
such as:

```text
data/processed/
  strong_cv/
  weak_cv/
  jumbo/
```

Raw real-robot images/video should be privacy-reviewed before any public release.
