"""Atomic per-output JSON metadata, including failed attempts."""
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import tempfile


@dataclass
class ExperimentMetadata:
    experiment_id: str
    method: str
    model: str | None
    source_image: str
    source_age: float | None
    target_age: float | None
    seed: int
    configuration: dict
    device: dict
    runtime: float | None = None  # seconds
    status: str = "pending"
    error_message: str | None = None
    model_version: str | None = None
    candidate_number: int | None = None


def write_metadata(path: str | Path, metadata: ExperimentMetadata) -> None:
    """Replace a record atomically; interruption leaves the previous record intact."""
    payload = json.dumps(asdict(metadata), indent=2, allow_nan=False)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8",
                                         dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(payload + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def read_metadata(path: str | Path) -> dict:
    with Path(path).open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("Metadata must be a JSON object")
    return payload
