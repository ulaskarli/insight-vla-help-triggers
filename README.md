# INSIGHT: INference-time Sequence Introspection for Generating Help Triggers in Vision-Language-Action Models

Official implementation of our ICRA 2026 paper:

**INSIGHT: INference-time Sequence Introspection for Generating Help Triggers in Vision-Language-Action Models**

---

## 🔧 Installation

INSIGHT supports two environment workflows. **uv is the recommended setup** because it creates and manages an isolated project environment automatically. Conda is also supported for users who prefer an existing Conda workflow.

### Option 1: uv (recommended)

Install [`uv`](https://docs.astral.sh/uv/getting-started/installation/) if it is not already available, then clone and sync the project:

```bash
git clone https://github.com/ulaskarli/insight-vla-help-triggers.git
cd insight-vla-help-triggers
uv sync
```

`uv sync` creates a project-local `.venv` using the repository's pinned Python 3.10 version and installs the package, runtime dependencies, and development dependencies (including `pytest`). You do **not** need to activate the environment. Run repository commands through `uv run` so they always execute inside the managed environment.

Verify the installation:

```bash
uv run pytest -q
```

### Option 2: Conda

Create a fresh Python 3.10 environment (or use an existing compatible environment):

```bash
conda create -n insight_env python=3.10 -y
conda activate insight_env

git clone https://github.com/ulaskarli/insight-vla-help-triggers.git
cd insight-vla-help-triggers
python -m pip install -e . pytest
```

Verify the installation:

```bash
python -m pytest -q
```

If you already have a Conda environment with the scientific dependencies installed, you can reuse it; just ensure the project and `pytest` are installed in that environment.

## 🚀 Quickstart

The commands below use **uv by default**. If you are using the Conda alternative, replace `uv run python` with `python` and `uv run pytest` with `python -m pytest`.

Train the weakly supervised MIL model:

```bash
uv run python insight/train.py --config experiments/configs/mil_weak.yaml
# CKPT list saved to: experiments/results/mil_weak/mil_ckpts.json
```

Evaluate a MIL model:

```bash
# weak (bag-level)
uv run python insight/evaluate.py \
  --mode mil-weak \
  --ckpts_json experiments/results/mil_weak/mil_ckpts.json \
  --data_dir /your/test/data/here/

# strong (step-level)
uv run python insight/evaluate.py \
  --mode mil-strong \
  --ckpts_json experiments/results/mil_weak/mil_ckpts.json \
  --data_dir /your/test/data/here/
```

Train the strongly supervised model:

```bash
uv run python insight/train.py --config experiments/configs/single_strong.yaml
# CKPT list saved to: experiments/results/single_strong/single_ckpts.json
```

Evaluate the strongly supervised model:

```bash
uv run python insight/evaluate.py \
  --mode single-strong \
  --ckpts_json experiments/results/single_strong/single_ckpts.json \
  --data_dir /your/test/data/here/
```

Run conformal prediction for False Not Ask (missed-help control):

```bash
uv run python insight/cp.py --config experiments/configs/cp_fn.yaml
```

Run conformal prediction for False Ask (false-ask control):

```bash
uv run python insight/cp.py --config experiments/configs/cp_fa.yaml
```

### Development checks

```bash
uv run pytest -q
uv run python -m compileall -q insight scripts
```

## 📜 Citation

If you use this work, please cite:

```bibtex
@inproceedings{karli2026insight,
  title={INSIGHT: INference-time Sequence Introspection for Generating Help Triggers in Vision-Language-Action Models},
  author={Karli, Ulas Berk and Shangguan, Ziyao and Fitzgerald, Tesca},
  booktitle={2026 IEEE International Conference on Robotics and Automation (ICRA)},
  year={2026}
}
```

## License

This project is licensed under the Apache License 2.0. See `LICENSE`.
