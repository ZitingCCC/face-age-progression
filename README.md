# Identity-Preserving Face Age Progression

AI6132 research project investigating plausible child-to-adult face age progression while preserving identity-related facial characteristics. Generated images must never be described as predictions of a person's actual future appearance.

Tasks 01–03 provide reusable infrastructure, canonical Colab setup, and FG-NET metadata preparation. Task 04A adds candidate model feasibility wrappers; candidate ranking, experimental evaluation, and experimental results remain unimplemented.

The planned architecture consists of an existing GAN baseline (M0), pretrained diffusion age progression (M1), and identity-aware candidate selection (M2). Future evaluation will measure target-age accuracy, identity similarity, image quality, and inference time. Dataset splits must separate subject IDs.

Codex Cloud is the CPU development and testing environment. Google Colab Free is the intended GPU execution environment, with unpredictable hardware and runtime availability. Google Drive holds persistent data and results; code is maintained in GitHub. Actual Colab execution, Drive mounting, and CUDA operation require manual verification.

```text
configs/                 YAML experiment, diffusion, and identity settings
notebooks/               Colab setup/dataset/feasibility notebooks and workflow placeholders
src/
  data/                  FG-NET metadata, subject splits, longitudinal pairs
  models/                Thin SAM/FADING feasibility adapters
  evaluation/            Reserved evaluation package
  utils/                 Configuration, seeds, devices, paths, metadata, resume
scripts/                 Dataset CLI and reserved inference/evaluation entry points
tests/                   CPU-only infrastructure and synthetic dataset tests
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

The runtime requirements are intentionally minimal and are not a cloud environment freeze. The Colab setup notebook installs `requirements.txt` with constraints that preserve the supplied PyTorch and torchvision builds. Diffusers and model-specific dependencies will be added when model implementation is requested. Tests require no GPU, internet access, dataset, or pretrained model once dependencies are installed.

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

## Running on Google Colab

1. Open `notebooks/01_setup.ipynb` in Google Colab.
2. Select **Runtime → Change runtime type → GPU**.
3. Configure `GITHUB_OWNER`, `GITHUB_REPO`, and `GITHUB_BRANCH` if needed (defaults: `ZitingCCC/face-age-progression`, `main`).
4. Only for a private repository, add `GITHUB_TOKEN` to Colab Secrets, enable notebook access, and set `PRIVATE_REPOSITORY = True`. Never paste credentials into notebook cells.
5. Run the notebook from top to bottom and approve Google Drive mounting when prompted.
6. Confirm CUDA diagnostics. CPU setup can pass, but generation experiments should not run without CUDA.
7. Confirm `data`, `models`, `generated`, `results`, and `cache` exist under `/content/drive/MyDrive/AI6132`.
8. Confirm the setup smoke test passes and the setup summary is printed.

The checkout stays at `/content/face-age-progression`; large persistent files and future Hugging Face caches use Drive. Rerunning preserves persistent files and refuses to update a checkout with local changes or a divergent branch. Save local edits before updating. Dependency conflicts stop installation rather than replace Colab's PyTorch builds. Restart the runtime if pip requests it, then rerun setup.

This notebook prepares infrastructure only and downloads no datasets or pretrained models. Its temporary metadata is a setup check, not an experimental result. CPU tests and static notebook validation do not verify actual Colab execution, Drive authorization/persistence, or GPU behavior; check those manually in Colab.

## FG-NET Dataset Preparation

FG-NET is not stored in GitHub. Obtain it separately according to applicable access/license terms and manually extract the real data into Google Drive:

```text
MyDrive/AI6132/data/fgnet/
    raw/
    processed/
```

Run `notebooks/01_setup.ipynb` first, then `notebooks/02_dataset.ipynb`. Set the notebook's configurable `RAW_ROOT` to your extraction root if its layout differs. Missing data produces placement instructions and skips processing; nothing is downloaded automatically. Metadata preparation is CPU-only, leaves images on Drive, and creates no full-dataset copy in `/content`.

**Filename assumption (not verified against real FG-NET):** the full filename stem must match exactly three ASCII subject digits + literal `A` (case-insensitive) + exactly two ASCII age digits + an optional single ASCII letter. Extensions `.jpg`, `.jpeg`, `.png` are case-insensitive. Examples: `001A05.jpg`, `001A05a.JPG`, `002a00.png`. The optional letter distinguishes same-age images and does not alter identity or age. Subject IDs retain three-digit leading zeros; ages are non-negative integers. Verify this convention against the actual files; adjust the isolated `FILENAME_PATTERN` in `src/data/fgnet.py` if necessary. Malformed supported images fail clearly instead of being skipped or guessed. Discovery is recursive and validates metadata, not decoded pixels, face content, authenticity, or dataset completeness.

The reusable modules are `src/data/fgnet.py` (discovery, parsing, metadata validation, summary, output/orchestration), `splits.py` (subject assignment and leakage validation), and `pairs.py` (longitudinal pairing, filters, and pair integrity). The notebook and CLI call these functions. No GPU dependency is imported by the data pipeline.

Outputs under `processed/`:

- `metadata.csv`: `subject_id, age, image_path, split`, ordered by subject ID, age, image path. Paths are relative POSIX strings under the configured raw root; resolve them as `raw_root / image_path`. Preserve subject IDs as text when reading CSV and convert ages to integers.
- `pairs.csv`: `subject_id, source_image, source_age, target_image, target_age, age_gap, age_gap_group, split`, ordered by subject, source age/path, target age/path. Empty pair tables still have headers.
- `preparation.json`: raw-root location, preparation configuration, and computed dataset summary. These are data counts, never model metrics or experimental results.

Subject-level splitting prevents identity leakage across splits. Sorted unique subject IDs are shuffled using a local `random.Random(seed)` (default `42`, integer in `[0, 2**32)`). Default subject ratios are train/val/test = 0.70/0.15/0.15, configurable in `configs/dataset.yaml`. Largest-remainder rounding floors each subject count, then allocates the remainder by descending fractional part, ties in train/val/test order. Small datasets can have empty splits. All ages of an identity stay within the same split; images are never independently randomized. The reusable validator explicitly fails on overlap between every pair of splits.

Longitudinal pairing is a separate within-subject operation inside each assigned split. By default it generates every valid image combination with `target_age > source_age` and `age_gap = target_age - source_age`. For one subject at ages 5, 10, 18, 30 this gives six pairs, all within that subject's single split. Multiple images at the same age can pair with older images but never with each other. Optional inclusive source/target age bounds and gap bounds restrict pairs; the default minimum gap is 1. The optional `max_pairs_per_subject` keeps the first K valid pairs in the documented stable order, with no randomness. Without a cap the number of pairs can grow quadratically per subject. Age-gap group labels are `0-5` for gaps 1–5, `6-10` for 6–10, and `11+` for 11 or more; gap 0 is invalid. No performance metrics are computed.

Local CLI (no Drive or CUDA required):

```bash
python scripts/prepare_fgnet.py --help
python scripts/prepare_fgnet.py --raw-root /path/to/fgnet/raw --output-root /path/to/fgnet/processed
```

The CLI reuses Task 01's YAML loader. `--config` selects dataset YAML; seed, ratios, age constraints, pair gap, and pair cap can be overridden with CLI flags. Randomness is local rather than using Task 01's global PyTorch seed helper, keeping dataset work free of GPU initialization. CLI configuration records effective defaults/overrides. The raw and output roots must be separate non-overlapping directories.

Reruns refuse existing named outputs by default. Inspect them before using `--overwrite` or notebook `OVERWRITE = True`; this explicitly replaces only `metadata.csv`, `pairs.csv`, and `preparation.json`, preserving other output names and all raw images. Non-regular output destinations and symlinks are rejected. All tables are validated and serialized in staging before writing; individual replacements are atomic where the filesystem supports rename, but the bundle is not a single transaction. An interruption can leave a partial bundle; inspect and rerun with explicit overwrite. Use one writer per output root. Actual mounted Drive persistence and rename behavior require Colab validation.

Automated tests use only temporary synthetic filename/content placeholders, with no downloads, real FG-NET, or GPU. Real filename compatibility, raw-data integrity/completeness, Google Drive behavior, and top-to-bottom execution in real Colab remain to be verified for Task 03. Raw FG-NET, processed datasets, archives, model weights, caches, and generated images must never be committed to GitHub.


## Task 04A: SAM and FADING feasibility

[SAM](https://github.com/yuval-alaluf/SAM) is the candidate M0 GAN baseline;
[FADING](https://github.com/MunchkinChen/FADING) is the candidate M1 diffusion method.
Neither is validated or accepted until a real Colab GPU smoke test succeeds.
CPU fake tests verify adapters only. This is not an experiment and outputs are
not research results or predictions of a person's true future appearance.

Run `notebooks/01_setup.ipynb` first, prepare the external dependencies/checkpoints
using the [exact setup and GPU validation procedure](docs/model_feasibility.md),
then run `notebooks/03_model_feasibility.ipynb`. It deterministically selects one
real FG-NET child image from Drive metadata, prints subject/source/target ages,
and requests one saved output per candidate. Smoke JSON records include status,
failures, actual measured worker runtime/peak CUDA memory, paths and versions.

External repositories, checkpoints, caches and generated images stay outside Git
under the documented `/content` and persistent Drive paths. Verify upstream code,
checkpoint and dataset licenses/provenance before experimental use. SAM uses an
isolated venv retaining canonical torch/torchvision; FADING uses an isolated
Python 3.10 legacy/compatibility environment. **Never destructively downgrade the
Task 02 canonical environment.** Modern diffusers/PyTorch compatibility and T4
memory feasibility are untested. SAM computes at 1024px internally (256px saved);
FADING uses a two-prompt internal batch (one source/one saved 512px output). These
internal exceptions are explicitly configured. FP32 is retained until FP16 is
verified safe. No M2 selection or evaluation is implemented.
