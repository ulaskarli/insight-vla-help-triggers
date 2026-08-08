# Real-robot validation and GELLO intervention collection

> [!IMPORTANT]
> **Hardware validation status:** This integration is experimental and is being validated on the real xArm7/GELLO setup. Use the inspection-only mode before enabling robot motion.


The GELLO integration is deliberately separated from the core INSIGHT package.
INSIGHT only needs an uncertainty-enabled VLA response; GELLO is one convenient
way to execute the robot and collect human recovery demonstrations.

## Control flow

```text
robot observation
      |
      v
uncertainty-enabled π0-FAST inference
      |
      +--> action chunk [H, D]
      +--> AU / EU / entropy / chosen-token log p
                         |
                         v
                  INSIGHT Strong
                    /         \
               execute       HELP
                  |            |
            buffer chunk       +--> reject entire VLA chunk
                  |            +--> hold at current state
                  v            +--> optional GELLO takeover
             robot actions                    |
                                              v
                                      recovery demonstration
                                              |
                                              v
                                      fresh VLA inference
```

The important ordering is **VLA inference -> INSIGHT decision -> action chunk
execution**. Putting the gate after a standard action-chunk broker is too late,
because the broker has already accepted the chunk and exposes it one control
step at a time.

## 1. Inspection-only test

Start the same robot/camera ZMQ nodes and uncertainty-enabled OpenPI server used
by your existing GELLO setup. Then run:

```bash
uv run python integrations/gello/run_vla_insight.py \
  --mode inspect \
  --checkpoint-dir experiments/results/single_strong_jumbo \
  --policy-host 127.0.0.1 \
  --robot-host 127.0.0.1 \
  --prompt "lift the corn"
```

`inspect` performs one policy inference, evaluates INSIGHT, saves the observation,
rejected/returned action chunk, features, and fold logits, and exits. It never
calls `env.step()`.

## 2. Autonomous halt test

After inspection is correct:

```bash
uv run python integrations/gello/run_vla_insight.py \
  --mode autonomous \
  --checkpoint-dir experiments/results/single_strong_jumbo \
  --threshold-logit 0.5 \
  --prompt "lift the corn"
```

A safe inference executes up to `action_horizon` actions. A help prediction
saves the halt snapshot and rejects the complete action chunk. At the prompt,
`r` asks π0-FAST to infer again from the unchanged state and `q` exits.

The default `0.5` threshold is a threshold on the **raw Strong logit**, matching
the saved real-time evaluation configuration. It is not the same as probability
0.5 (raw logit 0).

## 3. Collect a GELLO recovery from a halted state

```bash
uv run python integrations/gello/run_vla_insight.py \
  --mode intervention \
  --checkpoint-dir experiments/results/single_strong_jumbo \
  --gello-port /dev/serial/by-id/<YOUR-GELLO-DEVICE> \
  --intervention-dir ~/insight_interventions \
  --prompt "lift the corn"
```

When INSIGHT halts, choose `g`. The script asks the operator to align the GELLO
leader with the halted robot pose. GELLO is then initialized using the current
robot joint state. Press `s` to begin the recovery demonstration and `q` to end
it. The recovery frames/actions are saved with GELLO's existing `save_frame`
format under the same halt directory as the rejected VLA chunk and INSIGHT
metadata.

After the recovery ends, the rejected VLA chunk is discarded and π0-FAST is
queried again from the new state.

## Deployment checkpoint choice

For paper reproduction, evaluate each fold with its matching Strong checkpoint.
For new real-robot states there is no naturally matching fold. The integration
script therefore loads all `single_fold*.pt` files in the selected directory and
averages their raw logits as a lightweight deployment ensemble. You can point it
at either `single_strong/` or `single_strong_jumbo/`.

This ensemble is a practical validation/deployment choice and should not be
reported as the paper's cross-validation protocol.

## Hardware safety

Use the inspection-only mode first, keep the physical emergency stop accessible,
and begin autonomous testing at conservative robot speed/acceleration limits.
The integration additionally clips each commanded joint-target change with
`--max-joint-delta` (default `0.05` rad); set a nonpositive value only if you
intentionally want to disable that extra guard.
