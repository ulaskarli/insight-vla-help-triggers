# INSIGHT checkpoints

Binary checkpoints are intentionally not committed to Git history.

The ICRA 2026 experiment archive contains these model families:

| Family | Historical experiment path | Meaning |
| --- | --- | --- |
| Strong | `experiments/results/single_strong/single_fold{0..9}.pt` | 10 fold-matched strongly supervised models used for the main real-robot evaluation. |
| Strong Jumbo | `experiments/results/single_strong_jumbo/single_fold{0..9}.pt` | Strong models trained on the larger/Jumbo training set. |
| Weak | `experiments/results/mil_weak/mil_fold{0..9}.pt` | Weakly supervised MIL models. |

For paper reproduction, use the checkpoint matching the evaluation fold. Do not
select the fold with the best test score as a "best" deployment checkpoint.

For new real-robot states, `StrongHelpTrigger` can load all ten Strong/Strong
Jumbo checkpoints and average their raw logits as a deployment ensemble. This is
not the cross-validation protocol used to report the paper results.

The public release should attach these weights to a GitHub Release (or a model
host such as Hugging Face) rather than add them to the repository's Git history.
