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

### SAM Matplotlib backend failure after PR #5

The researcher reported that the real Colab rerun passed PR #5's Ninja preflight
and progressed beyond the previous Ninja availability blocker. With Tesla T4,
Python 3.12, torch `2.5.1+cu121`, torchvision `0.20.1+cu121` and CUDA available,
the worker again reported `15637086208` total VRAM bytes and selected `cuda:0`.
The source was FG-NET `001A02.JPG` (age 2), target age 30, with intended output
`/content/drive/MyDrive/AI6132/generated/smoke/sam_001A02_age30.png`.
The attempt ran for a reported **121.77 seconds**, returned code 1, and produced
**no output image**. The new error was:

```text
RuntimeError: ValueError: Key backend: 'module://matplotlib_inline.backend_inline' is not a valid value for backend
```

The reported supported-backend list included `agg`. This is failed-attempt
evidence, not a successful generation/runtime benchmark.

The exact project/upstream path is `model_worker.sam()` →
`from utils.common import tensor2im` →
[SAM utils/common.py at the pinned revision](https://github.com/yuval-alaluf/SAM/blob/c1895aef275e702fba7560284dc16df60d65210e/utils/common.py)
→ module-scope `import matplotlib.pyplot as plt` → Matplotlib initialization.
This import follows `from models.psp import pSp`, and precedes our checkpoint
loading. `tensor2im` itself uses PIL; the pyplot import is incidental to SAM's
shared visualization utilities. No SAM code selects a notebook backend here.

[IPython kernel setup](https://github.com/ipython/ipykernel/blob/main/ipykernel/kernelapp.py)
sets `MPLBACKEND=module://matplotlib_inline.backend_inline` when unset.
[Matplotlib initialization](https://github.com/matplotlib/matplotlib/blob/v3.9.2/lib/matplotlib/__init__.py)
reads `MPLBACKEND` and assigns `rcParams['backend']`.
[The registry](https://github.com/matplotlib/matplotlib/blob/v3.10.7/lib/matplotlib/backends/registry.py)
maps this module URI to `inline`, which requires external backend registration.
Our PR #5 launcher copied the parent environment unchanged for this variable, so
a notebook-specific setting crossed into the isolated batch process and was
rejected by its Matplotlib. The Colab Matplotlib/inline package versions and an
environment snapshot were not supplied; we do not assert a particular installed
version or distinguish an absent plugin from incompatible registration there.
A CPU subprocess with real Matplotlib 3.10.8 and simulated absent inline
registration reproduces the exact `Key backend` ValueError; changing only the
child backend to Agg permits pyplot/backend initialization without any model.

**Scoped fix:** `worker_environment('sam', ...)` explicitly sets
`MPLBACKEND=Agg` in its copied child environment. This built-in non-interactive
backend is appropriate for the one-image batch worker and needs no Jupyter
plugin. It is applied before any Matplotlib import, including the environment
used by the Ninja preflight. Parent/global `os.environ`, the canonical notebook,
both PyTorch installations, external SAM checkout, checkpoints and datasets
remain unchanged. The isolated Ninja bin prepend, inherited PATH and CUDA
variables remain intact. At the time of this SAM fix, FADING inherited its
original backend unchanged; the later FADING-only correction is documented below.

The two `TORCH_CUDA_ARCH_LIST is not set` warnings are separate and non-fatal in
the supplied log. PyTorch defaults to visible-card architectures when the variable
is unset. We leave unset and explicitly configured values unchanged, and do not
hardcode T4 architecture: Colab can assign other GPUs. The warnings and progression
to Matplotlib do not establish whether every extension was freshly compiled,
loaded from cache, or fully validated. **SAM GPU generation remains pending.**

After merge, update the clean project checkout to `main` (`git switch main`, then
`git pull --ff-only origin main` from `/content/face-age-progression`) and run
`python scripts/smoke_sam.py --check`. Reuse the exact previous one-image command
with the verified source path to `001A02.JPG`, `--source-age 2 --target-age 30`,
the same intended output path, and verified `--trust-sam-checkpoint`.
No package install, notebook-backend activation, architecture override, checkpoint
download or cache deletion is required for this fix. Inspect the new status/output
and preserve any new failure details on Drive; a failed JSON with no PNG is
retryable. Do not report successful generation until a real rerun produces it.

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

### FADING Matplotlib backend failure: Task 04B-Fix1

The researcher reported the following **failed real Colab smoke attempt** on
merged main. The worker successfully selected Tesla T4; dependency checks, CUDA
availability, a CUDA tensor allocation, and `python scripts/smoke_fading.py --check`
had passed. The external checkout was `/content/FADING_stable`, pinned to
`b1fc2e84fc02a2e048593766803627e219bfd017`, with Python
`/content/fading-env/bin/python` (3.10.21), torch `2.0.1+cu117`, torchvision
`0.15.2+cu117`, diffusers `0.27.1`, transformers `4.36.2`, accelerate `0.28.0`,
NumPy `1.26.4`, and huggingface_hub `0.25.2`. The specialized checkpoint at
`/content/drive/MyDrive/AI6132/models/fading/finetune_double_prompt_150_random`
existed and passed project preflight. These checks do not establish generation
feasibility.

The source was FG-NET `001A02.JPG`, source age 2, target age 30, gender prompt
`male`. Generation returned code 1 after approximately **13.2 seconds**, with
**no PNG** at the intended output
`/content/drive/MyDrive/AI6132/generated/smoke/fading_001A02_age30_male.png`.
The error was:

```text
RuntimeError: ValueError: Key backend: 'module://matplotlib_inline.backend_inline' is not a valid value for backend
```

The supported-backend list included `agg`. This is failed-attempt evidence,
not a successful inference benchmark. A Transformers offline/cache migration
warning preceded the failure. It remains uninvestigated: Matplotlib initialization
failed first, and there is no evidence yet that the warning blocks generation.
Offline flags and cache behavior are unchanged in this task.

**Cause and import path:** our `model_worker.fading()` executes upstream
`age_editing.py` through `runpy.run_path`. At the pinned revision,
[age_editing.py](https://github.com/gh-BumsooKim/FADING_stable/blob/b1fc2e84fc02a2e048593766803627e219bfd017/age_editing.py)
imports `FADING_util.util`, whose
[module-level pyplot import](https://github.com/gh-BumsooKim/FADING_stable/blob/b1fc2e84fc02a2e048593766803627e219bfd017/FADING_util/util.py)
initializes Matplotlib before the script's pipeline loading. The parent
Colab/IPython notebook supplies `MPLBACKEND=module://matplotlib_inline.backend_inline`;
the launcher previously copied it unchanged into FADING's isolated interpreter.
That notebook backend was rejected there. The supplied evidence does not specify
Matplotlib/inline package versions; we do not assert a particular plugin version.

**Scoped fix:** the FADING branch of `worker_environment` sets exactly
`MPLBACKEND=Agg` in the child copy before process startup and upstream imports.
Agg is built in and non-interactive, so no notebook backend registration is needed.
Parent/global `os.environ`, canonical Colab, both isolated environments, external
source, checkpoints, datasets, package versions, model settings and precision are
unchanged. FADING's PATH and CUDA variables are preserved. SAM's existing Agg
override and Ninja PATH prepend remain exactly as before; the researcher reports
SAM has already passed real GPU generation. No new SAM GPU run was done in Codex.

After merge, update the clean project checkout to main and run the same command:

```bash
git -C /content/face-age-progression switch main
git -C /content/face-age-progression pull --ff-only origin main
cd /content/face-age-progression
python scripts/smoke_fading.py --check
python scripts/smoke_fading.py \
  --source /content/drive/MyDrive/AI6132/data/fgnet/raw/001A02.JPG \
  --source-age 2 --target-age 30 --gender male \
  --output /content/drive/MyDrive/AI6132/generated/smoke/fading_001A02_age30_male.png
```

No install, environment activation, asset download or cache deletion is needed
for this fix. Inspect the new Drive metadata and output, and retain any subsequent
failure details. CPU regressions exercise the real Matplotlib import with notebook
registration unavailable, without loading models or requiring CUDA.
**Real Colab FADING GPU generation remains pending until this rerun.**

### FADING missing IPython after Task 04B-Fix1: Task 04B-Fix2

The researcher reran the exact FADING command above after the Matplotlib fix.
The previous backend failure was resolved. Dependency/checkpoint/preflight/CUDA
checks had passed and Drive was writable. The worker selected Tesla T4 (`cuda:0`,
`15637086208` total GPU memory bytes). The isolated runtime remained Python
3.10.21, torch `2.0.1+cu117`, torchvision `0.15.2+cu117`, diffusers `0.27.1`,
transformers `4.36.2`, accelerate `0.28.0`, NumPy `1.26.4`, and huggingface_hub
`0.25.2`. This second attempt returned code 1 after approximately **13.2 seconds**:

```text
RuntimeError: ModuleNotFoundError: No module named 'IPython'
```

No output PNG was created and no CUDA OOM was observed. This is failed startup
evidence, not a FADING inference benchmark or evidence of successful checkpoint
loading/inference.

**Pinned-source diagnosis:** `model_worker.fading()` runs `age_editing.py`, which
imports `p2p.py`; that imports `FADING_util.ptp_utils`. `null_inversion.py` also
imports the same utility. At revision `b1fc2e84fc02a2e048593766803627e219bfd017`,
[ptp_utils.py](https://github.com/gh-BumsooKim/FADING_stable/blob/b1fc2e84fc02a2e048593766803627e219bfd017/FADING_util/ptp_utils.py)
contains the module-level statement `from IPython.display import display`.
There are **no active references to `display` anywhere in that file**; even
`view_images` returns an image array rather than calling display. IPython is an
unused notebook/display dependency, not a requirement of this batch inference
path. The community repository's pinned requirements and README setup instructions
do not declare IPython. Its adjacent `from tqdm.notebook import tqdm` also has no
active references; actual editing/inversion progress loops use `from tqdm import
tqdm` in `p2p.py` and `null_inversion.py`. We leave those progress imports unchanged.

**Minimal project-side compatibility:** before upstream imports, the FADING
worker prepares the real utility module from its local source, omitting only the
unused `IPython.display` import from an in-memory AST. No external file is rewritten
and no fake IPython module/display implementation is installed. All remaining
upstream code executes unchanged. The guard requires exactly that module-level
import with no `display` references, rejects source changes requiring display,
checks that the utility stays inside the configured checkout, refuses to overwrite
an already imported utility, and removes partial module registration on failure.
Existing clean pinned-revision checks still run before this compatibility step.

The old preflight checked external entrypoints, pipeline assets, interpreter and
revision, but neither imported nor inspected this transitive utility; therefore
it missed the unconditional IPython import. `smoke_fading.py --check` now parses
the utility and validates the compatibility guard. It imports no upstream code,
loads no models/checkpoints, downloads nothing, and requires no GPU. Missing or
changed utility source fails before generation. Passing preflight establishes
this narrow compatibility only, not all dependencies or GPU feasibility.

After merge, update the clean checkout to main and rerun the preflight and exact
one-image command in the preceding section. **No Colab environment installation
command is required:** do not install IPython, modify either isolated environment,
change package versions or delete caches for this fix. SAM, parent/global
environment, FADING `MPLBACKEND=Agg`, PATH/CUDA variables, offline/cache behavior,
model settings and precision remain unchanged. The Transformers offline/cache
message remains uninvestigated. CPU tests use synthetic upstream utilities and
block IPython/model imports; Codex performed no real GPU generation.
**Real Colab FADING GPU generation remains pending until the rerun.**

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
