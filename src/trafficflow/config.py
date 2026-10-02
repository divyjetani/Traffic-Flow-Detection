from pathlib import Path
import yaml


def load_config(path=None):
    path = Path(path) if path else Path(__file__).resolve().parents[2] / "configs" / "default.yaml"
    with open(path) as f:
        return yaml.safe_load(f)
