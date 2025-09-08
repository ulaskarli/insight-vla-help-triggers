# INSIGHT: INference-time Sequence Introspection for Generating Help Triggers in Vision-Language-Action Models

Official implementation of our ICRA 2026 paper:

**INSIGHT: INference-time Sequence Introspection for Generating Help Triggers in Vision-Language-Action Models**

---

## 🔧 Installation

Clone this repo and install dependencies:

```bash
git clone https://github.com/ulaskarli/insight-vla-help-triggers.git
cd insight-vla-help-triggers
pip install -r requirements.txt
```

## 🚀 Quickstart

Run a toy inference example:
```bash
python insight/inference.py --config experiments/configs/example.yaml
```

Train a baseline model:
```bash
python insight/train.py --config experiments/configs/mil_transformer.yaml
```

Train a baseline model:
```bash
python insight/evaluate.py --results_dir experiments/results/mil_transformer/
```

## 📜 Citation

If you use this code, please cite:

```bibtex
```
