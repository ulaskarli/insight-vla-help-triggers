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

Train MIL model:
```bash
python insight/train.py --config experiments/configs/mil_weak.yaml
# CKPT list saved to: experiments/results/mil_weak/mil_ckpts.json
```

Test a MIL model
```bash
# weak (bag-level)
python insight/evaluate.py \
  --mode mil-weak \
  --ckpts_json experiments/results/mil_weak/mil_ckpts.json \
  --data_dir /your/test/data/here/

# strong (step-level)
python insight/evaluate.py \
  --mode mil-strong \
  --ckpts_json experiments/results/mil_weak/mil_ckpts.json \
  --data_dir /your/test/data/here/  
```

Train strong supervised model:
```bash
python insight/train.py --config experiments/configs/single_strong.yaml
# CKPT list saved to: experiments/results/single_strong/single_ckpts.json
```

Test strong supervised model:
```bash
python insight/evaluate.py \
  --mode single-strong \
  --ckpts_json experiments/results/single_strong/single_ckpts.json \
  --data_dir /your/test/data/here/
```

Run Conformal Prediction based on False Not Ask (missed-help control)
```bash
python insight/cp.py --config experiments/configs/cp_fn.yaml
```

Run Conformal Prediction based on False Ask (false-ask control)
```bash
python insight/cp.py --config experiments/configs/cp_fa.yaml
```

## 📜 Citation

If you use this work, please cite:

```bibtex
```
