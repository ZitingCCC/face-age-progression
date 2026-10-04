# Task 04A: one-image feasibility, pending real Colab validation

SAM is a candidate M0 GAN baseline; FADING is a candidate M1 diffusion method.
Neither is accepted or validated. This workflow produces a plausible visual edit,
never a prediction of a child's true future appearance. It is not an experiment;
outputs and smoke metadata must not be presented as research results.

## Upstream inspection and constraints

The inspected SAM revision is `c1895aef275e702fba7560284dc16df60d65210e`.
Its README releases `sam_ffhq_aging.pt`; `models/psp.py` loads the encoder,
decoder, latent average, and (when configured) pretrained encoder from that
checkpoint. Auxiliary training checkpoints are unnecessary for this inference
path. Its original environment is Python 3.6 / torch 1.6.0 / torchvision 0.4.2.
We do not install that environment into Colab.

SAM alignment uses the upstream FFHQ/dlib alignment routine and an independently
obtained 68-landmark predictor. The bridge follows the upstream inference
notebook: RGB, aligned 256px input, normalization, upstream age channel,
`randomize_noise=False`, and `torch.no_grad()`. The pretrained decoder computes
at **1024px internally** and pools to the saved 256px output. Resizing the output
does not reduce its native decoder memory requirement. A different decoder size
cannot simply be substituted for the released checkpoint.

Official FADING revision `14fefbf7edbf5b1d3f867bd4720a61bfc1bdf2c0` releases
specialized weights (`finetune_double_prompt_150_random.zip`) and an inference
entry point `age_editing.py`. Project-side specialization/training is unnecessary.
The selected compatibility implementation is the community
[FADING_stable](https://github.com/gh-BumsooKim/FADING_stable) at
`b1fc2e84fc02a2e048593766803627e219bfd017`. This is not the official repository;
metadata records both provenances. Its requirements pin torch 2.0.1,
torchvision 0.15.2, diffusers 0.27.1, transformers 4.36.2 and accelerate 0.28.0.
Its compatibility changes are claims by that upstream, **not verified here**.

FADING takes one source and one target age, saves only the aged image, but
uses a **two-prompt internal batch** (source reconstruction + edit), with classifier
free guidance increasing internal work further. It operates at 512px. The
upstream runs 50 DDIM steps, guidance 7.5, seed 0, attention replacements 0.8/0.5;
these are upstream defaults captured by the pinned revision, not tuning results.
Null-text inversion optimizes per-image unconditional embeddings; it does not
train model weights, LoRA, or textual inversion tokens. Gradients must remain
enabled for this stage, while upstream sampling uses no-grad. No claim is made
that FP32 FADING fits a 15.64GB T4.

The researcher authorized these internal size exceptions for Task 04A. Both
`allow_upstream_internal_sizes` settings are explicit. Setting either to false
blocks generation clearly. Saved outputs remain 256px/512px, with one source,
one target age, and one saved output per candidate. FP32 is the only implemented
precision: SAM custom kernels and FADING inversion need separate FP16 validation.
The adapters reject unsupported precision rather than misreport it.

## Dependency strategy (manual setup; no installation at import or in tests)

Run `notebooks/01_setup.ipynb` first. Keep canonical torch/torchvision unchanged.
Run these commands explicitly in a Colab terminal (or subprocess argument lists),
not in the canonical setup notebook. All dependency installation here is a
**proposed unverified recipe**; capture pip logs, `pip check`, package versions and
failures. Do not treat a successful package install as inference feasibility.

SAM: create a separate venv inheriting canonical torch/torchvision. Constrain both
to the exact installed builds. Install only inference/alignment/build dependencies.
This environment can still fail to compile StyleGAN CUDA extensions on torch
2.11/cu130. Such a failure is a feasibility result; do not replace canonical torch.

```bash
git clone https://github.com/yuval-alaluf/SAM.git /content/SAM
git -C /content/SAM checkout c1895aef275e702fba7560284dc16df60d65210e
python -m venv --system-site-packages /content/sam-env
python -c "from importlib.metadata import version; from pathlib import Path; Path('/content/sam-constraints.txt').write_text('torch=='+version('torch')+'\ntorchvision=='+version('torchvision')+'\n')"
/content/sam-env/bin/python -m pip install -c /content/sam-constraints.txt 'numpy<2' scipy matplotlib tqdm dlib Pillow PyYAML
/content/sam-env/bin/python -m pip install --ignore-installed --no-deps ninja
/content/sam-env/bin/python -m pip check
```

The SAM worker provides two narrowly scoped compatibility bridges: Pillow's
removed `ANTIALIAS` name maps to `Resampling.LANCZOS`, and trusted legacy SAM
checkpoint loading explicitly sets `weights_only=False`, restoring `torch.load`
after initialization. This allows pickle execution: verify origin before setting
`trusted_checkpoint=True` or passing `--trust-sam-checkpoint`. The bridges do not
establish CUDA extension compatibility and never patch canonical files.

### SAM Ninja failure: observed Colab evidence and scoped fix

The researcher reported this **failed** real Colab smoke attempt:

- Tesla T4, `15637086208` total VRAM bytes (14.56 GiB).
- `/content/sam-env`, Python 3.12.3, torch `2.5.1+cu121`,
  torchvision `0.20.1+cu121`, CUDA build 12.1.
- CUDA available, one device, CUDA tensor allocation succeeded; the SAM worker
  printed the same GPU name/memory and selected `cuda:0`.
- The original `python scripts/smoke_sam.py --check` passed assets/revision checks.
- Generation failed with `RuntimeError: RuntimeError: Ninja is required to load C++ extensions`,
  return code 1, reported wall time 4.69 seconds, and **no output image**.
  This is failed-attempt evidence, not a successful inference benchmark.

At the pinned SAM revision, our worker's `from models.psp import pSp` imports
`models/stylegan2/model.py`, then `models/stylegan2/op/__init__.py`. That module
imports `fused_act.py` first and `upfirdn2d.py` second. Both call
`torch.utils.cpp_extension.load()` at module scope, for the `fused` and
`upfirdn2d` C++/CUDA extensions respectively. The first extension uses
`fused_bias_act.cpp` and `fused_bias_act_kernel.cu`. This happens **before** our
worker reaches `torch.load(checkpoint)`.

[PyTorch 2.5.1's loader](https://github.com/pytorch/pytorch/blob/v2.5.1/torch/utils/cpp_extension.py)
checks `is_ninja_available()` by executing `['ninja', '--version']` through PATH;
any execution error returns false. `verify_ninja_availability()` then raises the
exact reported message. It does not test whether the Python package is importable.
A local pip installation normally supplies the environment's `bin/ninja`:
[older Ninja wheels](https://github.com/scikit-build/ninja-python-distributions/blob/1.11.1/setup.py)
register `ninja=ninja:ninja` as a console script, while
[current builds](https://github.com/scikit-build/ninja-python-distributions/blob/master/CMakeLists.txt)
install the executable to the wheel's scripts directory. With
`--system-site-packages`, a generic `pip install ninja` can also be satisfied by
an inherited installation, without creating a local script. The explicit
`--ignore-installed --no-deps ninja` setup command above ensures a local
installation without uninstalling inherited packages or replacing PyTorch.

**Confirmed project-side cause:** the original launcher copied `os.environ`
and selected `/content/sam-env/bin/python`, but never prepended
`/content/sam-env/bin` to PATH. Selecting an interpreter by absolute path does
not activate its venv or change PATH. Thus its installed Ninja can be invisible
to PyTorch's subprocess. CPU reproduction using a temporary venv, a fake Ninja
executable and the actual PyTorch availability function returns false under the
old inherited PATH and true with the isolated bin prepended. The supplied Colab
log has no PATH/executable trace, so it cannot rule out an additional missing or
broken Ninja installation; the new preflight explicitly diagnoses those cases.

The fix copies the inherited environment and prepends the configured interpreter's
**lexical** bin directory for SAM only. It deliberately does not resolve Python's
symlink into the canonical interpreter directory. CUDA/Colab variables and the
existing PATH remain intact; FADING's PATH is unchanged. No global environment,
canonical notebook/requirements, model assets or external SAM code are modified.

`smoke_sam.py --check` now checks assets, predictor, executable Python, pinned clean
revision, executable Ninja/version, and PyTorch's own Ninja detection under the
same environment as generation. It prints selected Python/prefix, Ninja path,
Ninja version, torch version and detection status. The check imports PyTorch's
extension utility but imports no SAM model/ops, loads no checkpoint, compiles
nothing, downloads nothing, and requires no source image or GPU. The worker also
checks Ninja in its own process before SAM imports, checkpoint loading or GPU
inference, including when directly invoked. Passing this check establishes Ninja
availability only; compiler/nvcc/ABI compatibility and generation remain pending.

After merge, rerun in the existing GPU Colab session:

1. Save local edits, then update the project checkout to merged `main` with
   `git -C /content/face-age-progression switch main` followed by
   `git -C /content/face-age-progression pull --ff-only origin main`.
   Keep the pinned external SAM checkout and existing Drive assets unchanged.
2. From the project root run `python scripts/smoke_sam.py --check`.
   If it reports a missing/broken Ninja, run only
   `/content/sam-env/bin/python -m pip install --ignore-installed --no-deps ninja`
   and rerun the check. Confirm Ninja is executable and PyTorch detection is true.
3. Rerun the SAM check and SAM one-image cells in `03_model_feasibility.ipynb`
   with the same selected source/age/target, `RUN_SMOKE=True`, and verified
   `TRUST_SAM_CHECKPOINT=True`. FADING need not be rerun. A failed record with no
   PNG remains retryable; do not delete checkpoints or caches to fix PATH.
4. Inspect the actual status/output and new failure details on Drive. Stop on a
   new compiler/CUDA/checkpoint error and record it; do not downgrade canonical
   PyTorch. **Successful SAM GPU generation remains pending until this rerun.**

FADING: use a **separate Python 3.10 interpreter and venv**, with no system site
packages. Obtain Python 3.10 through a reviewed environment manager (for example
`uv python install 3.10`); do not replace Colab's Python. The following assumes
`python3.10` has been made available. If Colab cannot provide a usable isolated
Python/CUDA combination, stop FADING and log that blocker independently.

```bash
git clone https://github.com/gh-BumsooKim/FADING_stable.git /content/FADING_stable
git -C /content/FADING_stable checkout b1fc2e84fc02a2e048593766803627e219bfd017
python3.10 -m venv /content/fading-env
/content/fading-env/bin/python -m pip install -r /content/FADING_stable/requirements.txt 'numpy<2' 'huggingface-hub<0.26' Pillow tqdm matplotlib PyYAML
/content/fading-env/bin/python -m pip check
```

The NumPy cap avoids legacy torch/OpenCV NumPy-2 ABI incompatibilities; the hub
cap avoids removal of `cached_download` used by older diffusers. This still needs
real runtime verification. The legacy torch wheel has its own CUDA runtime;
compatibility with Colab's driver is untested. Never install this requirements
file into the canonical interpreter. No main requirements changes are needed.
Use one model process at a time; subprocess isolation releases its allocations
on exit. The notebook's parent does not load either model.

## Checkpoints, local-only execution and licenses

Manually obtain weights via the links in `configs/models.yaml` and verify their
provenance/access terms. Link advertisement was inspected; actual downloading,
current availability, integrity, and license applicability were **not tested**.
Keep SAM weights and the decompressed dlib landmark predictor under
`/content/drive/MyDrive/AI6132/models/sam/`.
Keep the extracted FADING pipeline under
`/content/drive/MyDrive/AI6132/models/fading/finetune_double_prompt_150_random/`.
It must contain `model_index.json` and local UNet, VAE, text encoder, tokenizer and
scheduler assets. Do not substitute generic Stable Diffusion weights for the
specialized pipeline. Inspect an archive before extraction and reject traversal
entries. No download or archive extraction is automated by this infrastructure.

Obtain the predictor from the source linked in
[SAM's inference notebook](https://github.com/yuval-alaluf/SAM/blob/c1895aef275e702fba7560284dc16df60d65210e/notebooks/inference_playground.ipynb)
(`dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2`) and decompress it on
Drive. Failure to detect an FG-NET face is logged, never silently bypassed.

Check source-code, checkpoint, base-model, predictor and dataset licenses before
final experimental use. Code licenses alone do not establish rights to every
asset. See [SAM](https://github.com/yuval-alaluf/SAM),
[official FADING](https://github.com/MunchkinChen/FADING), and the community repo.
No legal conclusion is asserted. External repositories, weights, cache, alignment
assets, smoke outputs and metadata stay outside Git. HF offline flags prevent
implicit Hub downloads; missing local components fail. Exact upstream HEAD and
tracked modifications are checked before each real invocation.

## Exact real-GPU validation procedure

1. Run `01_setup.ipynb`; select a GPU runtime, mount Drive, and confirm the Task 03
   processed metadata and raw FG-NET data are available. Checkout this Task 04A
   branch in the canonical notebook's `GITHUB_BRANCH` before setup. Run both
   model-specific dependency recipes above; verify canonical torch/torchvision
   versions have not changed. Obtain assets separately as described above.
2. Open `03_model_feasibility.ipynb`. Print actual GPU/runtime diagnostics; do not
   assume a T4. Paths come from the model config; customize isolated Python paths
   there if needed. Run the deterministic selection cell: first child by subject,
   age and relative image path from processed metadata; default target 30, source
   at most 17. Both models use this exact same raw image and target age. No gender
   is inferred from FG-NET: explicitly supply FADING's required prompt choice.
3. Run the local dependency/checkpoint checks. Review SAM pickle provenance and
   explicitly enable its trust flag. Run one SAM invocation, then one FADING
   invocation. Each saves one PNG and immediately writes one atomic metadata JSON
   under Drive `generated/smoke/`. The notebook catches a model failure so the
   other candidate remains independently testable. A hard runtime disconnect can
   leave `pending` metadata; inspect it and rerun, rather than labeling success.
4. Read the summary and inspect the actual PNGs. Record installation, loading,
   preprocessing, generation, observed runtime, peak allocated/reserved CUDA
   memory, GPU, precision and package/upstream versions. Only actual completion
   can establish GPU feasibility; image quality, identity and age accuracy are
   not assessed here. Save installation logs beside the smoke metadata on Drive.

Equivalent CLI invocations (replace source with the printed selected raw path):

```bash
python scripts/smoke_sam.py --check
python scripts/smoke_fading.py --check
python scripts/smoke_sam.py --source /path/to/selected.jpg --source-age 5 --target-age 30 --output /content/drive/MyDrive/AI6132/generated/smoke/sam.png --trust-sam-checkpoint
python scripts/smoke_fading.py --source /path/to/selected.jpg --source-age 5 --target-age 30 --gender female --output /content/drive/MyDrive/AI6132/generated/smoke/fading.png
```

The example age/gender/path arguments must be replaced with the selected source's
metadata and a researcher-supplied prompt; they are not measured results.
Successful matching records are skipped on restart; unmatched existing outputs
are refused. Use a new output path for a changed input/configuration. Per-output
metadata is atomic using Task 01's writer where the filesystem supports rename;
image + JSON is not a single transaction. Interrupted promotion requires manual
inspection of any unmatched image. Do not run multiple writers to the same path.

`wall_clock_inference_seconds` includes loading, preprocessing/alignment,
FADING inversion/editing, and saving; excludes interpreter startup, dependency
setup, downloads and local preflight checks. It is a cold one-image smoke duration,
not steady-state generation latency. CUDA peaks are measured in the worker,
cover loading through save, and describe PyTorch allocated/reserved memory, not
total device/driver usage. Failed workers preserve available measurements; a
killed process can only produce parent failure metadata with null measurements.
CPU fake tests label `test_only=True` and leave runtime/VRAM/resolution null.

## Limits and stop conditions

No SAM/FADING model execution, checkpoint download, FG-NET pixel processing, CUDA
extension build, Colab install, real timing/VRAM measurement or Drive persistence
has been tested in Codex. There is no GPU success, identity preservation,
target-age accuracy or image-quality claim. CPU tests establish only project-side
orchestration/validation behavior. M2, ArcFace, age estimation, LPIPS, training,
research figures and large-scale evaluation remain out of scope.

If released weights are inaccessible/incomplete, stop that model; do not train a
replacement. If isolated dependencies cannot run, record the error and stop that
model without downgrading Task 02. The other candidate can still be tested.
