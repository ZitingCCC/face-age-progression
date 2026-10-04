"""Explicit local, Colab, and mounted Drive storage paths."""
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ExperimentPaths:
    project: Path
    storage: Path

    @property
    def datasets(self) -> Path:
        return self.storage / "datasets"

    @property
    def outputs(self) -> Path:
        return self.storage / "outputs"

    @property
    def metrics(self) -> Path:
        return self.outputs / "metrics"

    @property
    def figures(self) -> Path:
        return self.outputs / "figures"

    @property
    def generated(self) -> Path:
        return self.outputs / "generated"

    @property
    def metadata(self) -> Path:
        return self.outputs / "metadata"

    def create_output_dirs(self) -> None:
        for path in (self.metrics, self.figures, self.generated, self.metadata):
            path.mkdir(parents=True, exist_ok=True)


def configure_paths(environment: str = "local", project_root=None,
                    storage_root=None, drive_root=None) -> ExperimentPaths:
    """Drive must already be mounted; this function never mounts or authenticates.

    Explicit storage_root overrides the environment's default storage location.
    Relative paths resolve against project_root, not the caller's working directory.
    """
    if environment not in {"local", "colab", "drive"}:
        raise ValueError("environment must be local, colab, or drive")
    default = (Path(__file__).resolve().parents[2] if environment == "local"
               else Path("/content/face-age-progression"))
    project = Path(project_root).expanduser().resolve() if project_root else default
    if storage_root is not None:
        storage = Path(storage_root).expanduser()
        storage = storage if storage.is_absolute() else project / storage
    elif environment == "drive":
        if drive_root is None:
            raise ValueError("drive_root is required for persistent Drive storage")
        storage = Path(drive_root).expanduser()
        storage = storage if storage.is_absolute() else project / storage
    else:
        storage = project
    return ExperimentPaths(project, storage.resolve())
