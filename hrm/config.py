"""Read current JSON and legacy YAML configuration files."""
import json
from pathlib import Path
import yaml


def read_config(path):
    text = Path(path).read_text()
    try:
        config = json.loads(text)
    except json.JSONDecodeError:
        # Some historical training runs wrote YAML to model_config.json.
        config = yaml.safe_load(text)

    if not isinstance(config, dict):
        raise ValueError(f"Configuration must be a JSON object or YAML mapping: {path}")
    return config
