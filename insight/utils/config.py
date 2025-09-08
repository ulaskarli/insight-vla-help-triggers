import yaml
from dataclasses import dataclass
from typing import Any, Dict

@dataclass
class Config:
    data_dir: str = "data"
    results_dir: str = "experiments/results"
    seed: int = 1337

def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)
