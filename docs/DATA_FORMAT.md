# INSIGHT data format

INSIGHT operates on uncertainty sequences from individual VLA inference steps.
Each inference step corresponds to one autoregressively generated FAST token
sequence that decodes into an action chunk.

## Historical raw rollout step

The data used for the paper stored each inference step as a NumPy dictionary with
at least:

```text
outputs/au
outputs/eu
outputs/entropy
outputs/perplexity
```

`outputs/perplexity` is a historical/misleading name: the OpenPI experiment code
stored the selected token's **log probability** in that field.

## Feature construction

`prepare_data.py` trims the first 3 and final 2 generated tokens and stacks the
remaining token features as

```text
[AU, EU, entropy, chosen-token log probability]
```

producing one variable-length `float32` array of shape `[T, 4]` per VLA inference
step. This implementation order is the contract expected by the released
checkpoints, even if another ordering appears in the paper text.

## Strong supervision

Strong datasets contain individual step sequences and binary labels:

```python
(X_steps, y_steps)
```

where each `X_steps[i]` is `[T_i, 4]` and `y_steps[i]` is `0/1` for no-help/help.
The 10-fold files consumed by the training code are named
`train_steps_{fold}.pt`, `val_steps_{fold}.pt`, and `test_steps_{fold}.pt`.

## Weak supervision

Weak datasets group the same step sequences into episode bags:

```python
(bags_X, bags_y, bags_src?)
```

where `bags_X[i]` is a list of `[T,4]` step arrays and `bags_y[i]` is the
episode-level success/failure label. Fold files use the names
`train_bags_{fold}.pt`, `val_bags_{fold}.pt`, and `test_bags_{fold}.pt`.

Raw robot images, private lab captures, and large rollout archives are not meant
to be committed to Git. If processed feature folds are released, publish them as
versioned data assets and document their checksum/version separately.
