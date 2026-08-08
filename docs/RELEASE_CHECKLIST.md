# Public release checklist

## Commit to Git

- Core model/training/evaluation code under `insight/`.
- Canonical experiment configs under `experiments/configs/` with relative paths.
- Data-preparation scripts under `scripts/`.
- `integrations/gello/run_vla_insight.py` and the integration docs.
- Documentation, tests, CI, citation metadata, and a chosen repository license.

## Keep out of Git history

- `experiments/results/` (checkpoints and generated metrics/results).
- Raw real-robot rollouts, videos, camera captures, or intervention demonstrations.
- Processed datasets unless deliberately released as versioned data assets.
- OpenPI base/fine-tuned VLA weights.
- Local machine paths, robot IP defaults, USB IDs, `.env`, caches, WandB/TensorBoard directories.

## Publish separately as release/model assets

- `single_strong/single_fold{0..9}.pt`
- `single_strong_jumbo/single_fold{0..9}.pt`
- `mil_weak/mil_fold{0..9}.pt` if weak checkpoints are part of the public release.
- Optional processed uncertainty-feature folds if redistribution/privacy review permits it.

## Before tagging v1.0

1. Validate `integrations/gello/run_vla_insight.py --mode inspect` against the real uncertainty-enabled OpenPI server.
2. Record the exact OpenPI Git revision and the exact fine-tuned π0-FAST checkpoint identifier/path used for validation.
3. Verify that Strong/Strong-Jumbo scores from live responses match offline scoring on saved copies of those same responses.
4. Validate that a positive trigger rejects the entire action chunk before the robot receives any command from it.
5. Validate GELLO takeover from the halted state and inspect the saved intervention data.
6. Choose and add a license for the INSIGHT repository itself.
7. Publish checkpoint assets and replace any temporary local checkpoint paths in documentation with release URLs/checksums.
