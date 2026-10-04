"""Safe YAML loading without implicit configuration mutation."""
from pathlib import Path
import yaml


def load_config(path: str | Path) -> dict:
    """Load a mapping; reject empty documents and non-mapping roots."""
    with Path(path).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a YAML mapping")
    return config
