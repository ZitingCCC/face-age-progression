"""Generic output checks; no image generation or format assumptions."""
from pathlib import Path
import json
from .metadata import read_metadata


def output_exists(path: str | Path) -> bool:
    path = Path(path)
    return path.is_file() and path.stat().st_size > 0


def is_completed(output_path: str | Path, metadata_path: str | Path,
                 expected: dict | None = None) -> bool:
    """Skip only nonempty outputs with completed, matching metadata.

    Use expected to match experiment ID, seed, configuration, or other fields
    when reusing paths across runs. Corrupt/missing records remain retryable.
    """
    if not output_exists(output_path):
        return False
    try:
        record = read_metadata(metadata_path)
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    return record.get("status") == "completed" and all(
        record.get(key) == value for key, value in (expected or {}).items())
