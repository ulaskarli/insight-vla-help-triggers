# OpenPI integration for INSIGHT

INSIGHT does not require vendoring the full OpenPI repository. The real-robot
experiments used a modified JAX π0-FAST inference path that exposes token-level
uncertainty statistics together with each action chunk.

## Required policy-server output contract

For every fresh π0-FAST action-chunk inference, the websocket policy should
return at least:

```python
{
    "actions": np.ndarray,      # [action_horizon, action_dim]
    "au": np.ndarray,           # [num_generated_tokens]
    "eu": np.ndarray,           # [num_generated_tokens]
    "entropy": np.ndarray,      # [num_generated_tokens]
    "perplexity": np.ndarray,   # historical name; actually chosen-token log p
    "probability": np.ndarray,  # optional; chosen-token probability
}
```

The released INSIGHT checkpoints were trained with the implementation feature
order

```text
[AU, EU, entropy, chosen-token log probability]
```

and the data-preparation code removes the first 3 and final 2 generated tokens
before classification. Do not reorder or negate these features when using the
released checkpoints.

> Historical naming note: the OpenPI experiment code stored the chosen token
> log-probability in a tracker/field named `perplexity`. Actual perplexity would
> be `exp(-mean(log p))`. The misleading field name is preserved only for
> compatibility with the original experiment data/checkpoints.

## Minimal OpenPI changes used by the experiments

The historical implementation changed the following OpenPI paths:

| OpenPI file | Purpose |
| --- | --- |
| `src/openpi/models/pi0_fast.py` | During autoregressive decoding, compute entropy, chosen-token log-probability, chosen-token probability, and LogTokU AU/EU from each token's vocabulary logits; return those trackers with the generated tokens. |
| `src/openpi/policies/policy.py` | Unpack the additional arrays returned by `sample_actions()` and place them in the policy result dictionary. |
| `src/openpi/transforms.py` | Keep the uncertainty arrays aligned with non-padding generated tokens while decoding FAST tokens into an action chunk. |
| robot policy adapter (our `iqrl_policy.py`) | Map xArm/GELLO observations into π0-FAST inputs and expose the uncertainty fields in the websocket response. |
| `scripts/serve_policy.py` / training config | Register the IQRL/xArm policy configuration and serve the fine-tuned π0-FAST checkpoint. |

### AU / EU implementation

The experiment implementation used the top 30 vocabulary logits. For the
selected top-K logits `z`, evidence was

```python
alpha = relu(z) + 1e-6
```

followed by the LogTokU Dirichlet AU/EU expressions used in the paper.
Categorical entropy was computed from `softmax(logits)`, and the fourth INSIGHT
feature was the log-softmax value of the token actually generated.

## Exact IQRL / xArm π0-FAST configuration used

The experiment OpenPI tree added an `iqrl_policy` adapter and a
`LeRobotIQRLDataConfig`. The relevant π0-FAST training/inference config was:

```python
TrainConfig(
    name="pi0_fast_iqrl",
    model=pi0_fast.Pi0FASTConfig(
        action_dim=8,
        action_horizon=30,
        max_token_len=180,
    ),
    data=LeRobotIQRLDataConfig(
        repo_id="iqrl_pi0_data",
        base_config=DataConfig(
            local_files_only=True,
            prompt_from_task=True,
        ),
    ),
    weight_loader=weight_loaders.CheckpointWeightLoader(
        "s3://openpi-assets/checkpoints/pi0_fast_base/params"
    ),
    num_train_steps=30_000,
)
```

`LeRobotIQRLDataConfig` repacks the dataset fields as
`observation/image`, `observation/wrist_image`, `observation/state`, `actions`,
and `prompt`; applies `IQRLInputs` / `IQRLOutputs`; and converts all eight
action dimensions to deltas during training and back to absolute actions on
output. The IQRL setup intentionally mirrors the custom π0-FAST LIBERO setup
used in the same OpenPI tree: both `LeRobotLiberoDataConfig` and
`LeRobotIQRLDataConfig` retain the `DataConfig` default
`use_quantile_norm=False` (standard z-score normalization). The IQRL-specific
changes were limited to the robot/data interface and the required action-space
and horizon settings; this normalization behavior should therefore be preserved
for reproduction rather than treated as an accidental omission.

Because no freeze filter is set in `pi0_fast_iqrl`, this configuration performs
full-parameter fine-tuning from the π0-FAST base weights.

### Serving the fine-tuned robot checkpoint

Do **not** rely on the modified `serve_policy.py` IQRL default. In the historical
file the IQRL default points to `s3://openpi-assets/checkpoints/pi0_fast_base`,
which has the right base architecture but not the robot fine-tuned weights.
Serve the trained checkpoint explicitly:

```bash
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi0_fast_iqrl \
  --policy.dir=/path/to/your/pi0_fast_iqrl/checkpoint
```

A locally trained run normally lives below a path of the form
`checkpoints/pi0_fast_iqrl/<experiment>/<step>/`. Use the exact checkpoint that
was used for the ICRA experiments when validating the release.

## Why this repository does not copy OpenPI

OpenPI is an actively developed upstream project. Copying the historical OpenPI
source tree into INSIGHT would make it harder to distinguish our changes from
upstream code. Instead, this repository documents the small inference-interface
changes and the exact IQRL configuration required by INSIGHT.

For a faithful reproduction, pin the historical OpenPI revision used on the
robot and apply only these changes. The currently reconstructed candidate base
revision is `29068dd`; the release should record it as validated only after the
real-robot rerun succeeds from a clean checkout.

## Validation order

1. Start the uncertainty-enabled OpenPI server with the original fine-tuned
   π0-FAST checkpoint.
2. Run `integrations/gello/run_vla_insight.py --mode inspect ...` and verify that
   the returned arrays and Strong score are sensible **without moving the robot**.
3. Run autonomous gating and verify that a help prediction rejects the entire
   newly inferred chunk before any action in that chunk is sent to the robot.
4. Only after that, enable GELLO intervention collection.
