# Identity-Preserving Face Age Progression

AI6132 research project investigating plausible child-to-adult face age progression while preserving identity-related facial characteristics. Generated images must never be described as predictions of a person's actual future appearance.

Task 01 provides reusable infrastructure only. No face-aging model, dataset pipeline, candidate ranking, evaluation pipeline, or experimental results have been implemented.

The planned architecture consists of an existing GAN baseline (M0), pretrained diffusion age progression (M1), and identity-aware candidate selection (M2). Future evaluation will measure target-age accuracy, identity similarity, image quality, and inference time. Dataset splits must separate subject IDs.

Codex Cloud is the CPU development and testing environment. Google Colab Free is the intended GPU execution environment, with unpredictable hardware and runtime availability. Google Drive holds persistent data and results; code is maintained in GitHub. GPU execution has not been verified in Task 01.

```text
configs/                 YAML experiment, diffusion, and identity settings
notebooks/               Five markdown-only workflow placeholders
src/
  data/                  Reserved dataset package
  models/                Reserved model package
  evaluation/            Reserved evaluation package
  utils/                 Configuration, seeds, devices, paths, metadata, resume
scripts/                 Reserved preprocessing/inference/evaluation entry points
tests/                   CPU-only infrastructure tests
outputs/
  metrics/               Small CSV/JSON metrics may be committed
  figures/               Final figures may be committed
```

For local development, from the repository root:

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -v
python -c "import src; from src.utils.device import detect_device; detect_device()"
```

The runtime requirements are intentionally minimal and are not a cloud environment freeze. In Colab, retain its installed CUDA-enabled PyTorch and install `requirements.txt` if needed. Diffusers and model-specific dependencies will be added when model implementation is requested. Tests require no GPU, internet access, dataset, or pretrained model once dependencies are installed.

Configuration and utility usage:

```python
from src.utils.config import load_config
from src.utils.reproducibility import set_seed
from src.utils.device import detect_device, device_info
from src.utils.paths import configure_paths
from src.utils.metadata import ExperimentMetadata, write_metadata, read_metadata
from src.utils.resume import is_completed

config = load_config('configs/experiment.yaml')
set_seed(config['seed'], config['deterministic'])
device = detect_device()
paths = configure_paths(**config['paths'])
```

Run scripts from the repository root so `src` is importable. Script placeholders exit with a clear unimplemented message. All experimental hyperparameters belong in YAML; current model names and experiment inputs are deliberately unset. Diffusion and identity YAML values are proposed defaults, not measured results.

`configure_paths()` defaults to the repository location for local/Codex work. `configure_paths('colab')` defaults to `/content/face-age-progression`; pass `project_root` if your checkout is elsewhere. For persistence, mount Drive separately in Colab and pass your chosen project folder as `drive_root` to `configure_paths('drive', drive_root=...)`. No account name is assumed. `storage_root` overrides default storage and relative storage paths resolve under the project root. Path configuration does not mount Drive or create directories; call `create_output_dirs()` explicitly.

Future experiment runners should save each output immediately, then atomically write its `ExperimentMetadata` JSON with status `completed`. Record failures with status `failed` and an error message. Runtime is in seconds; device details come from `device_info()`. Record model version and candidate number where available, and store inference parameters in the configuration field. Metadata supports pending records without invented runtime values.

`is_completed(output_path, metadata_path, expected=...)` requires a nonempty file and completed metadata; use `expected` to match experiment ID, configuration, seed, and input fields before skipping a reused path. Missing, corrupt, or failed records remain retryable. This generic check does not validate image contents or implement generation. Metadata replacement is atomic on filesystems supporting atomic rename; actual mounted Drive persistence and disconnection behavior need validation in Colab.

`set_seed` seeds Python, NumPy, PyTorch, and available CUDA devices, enables deterministic algorithms by default, and disables cuDNN benchmarking. Call it before CUDA initialization. Determinism is limited to a fixed hardware/software environment, and unsupported deterministic operations may raise errors. Python hash randomization requires `PYTHONHASHSEED` before launching Python.

Datasets, checkpoints, Hugging Face caches, generated images, Drive contents, environments, and secrets are ignored. Place external data and generated outputs in the designated ignored directories. Only small genuine metric files and final figures belong in the allowed output folders; no experimental results are supplied here.
