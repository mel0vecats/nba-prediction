import yaml
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CONFIG_PATH = PROJECT_ROOT / "configs" / "parameters.yaml"


def load_config(path: Path = CONFIG_PATH):
    with open(path, "r") as f:
        return yaml.safe_load(f)