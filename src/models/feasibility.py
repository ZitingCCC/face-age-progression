"""One-image feasibility orchestration. No torch or upstream imports here."""
import csv
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import subprocess
import tempfile

from src.utils.metadata import ExperimentMetadata, write_metadata, read_metadata
from src.utils.resume import is_completed
from .sam_runtime import check_sam_runtime


@dataclass
class SmokeMetadata(ExperimentMetadata):
    output_path: str | None = None
    precision: str | None = None
    output_resolution: list | None = None
    wall_clock_inference_seconds: float | None = None
    peak_cuda_memory_bytes: int | None = None
    peak_cuda_reserved_bytes: int | None = None
    upstream: dict | None = None
    versions: dict | None = None
    measurement_scope: str | None = None
    test_only: bool = False


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def age(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 100:
        raise ValueError(f'{name} must be an integer in [0, 100]')
    return value


def select_source(metadata_path, raw_root, target_age=30, max_source_age=17):
    """Stable first child by subject, age, relative path; never infer gender."""
    age(target_age, 'target_age')
    age(max_source_age, 'max_source_age')
    root = Path(raw_root).resolve(strict=True)
    with Path(metadata_path).open(newline='', encoding='utf-8') as handle:
        rows = list(csv.DictReader(handle))
    eligible = []
    for row in rows:
        source_age = age(int(row['age']), 'source_age')
        if source_age <= max_source_age and source_age < target_age:
            eligible.append((row['subject_id'], source_age, row['image_path']))
    if not eligible:
        raise ValueError('No child source younger than target age in processed metadata')
    subject, source_age, relative = min(eligible)
    relative_path = Path(relative)
    if relative_path.is_absolute():
        raise ValueError('metadata image_path must be relative to raw_root')
    source = (root / relative_path).resolve(strict=True)
    if not source.is_relative_to(root) or not source.is_file():
        raise ValueError('Selected image must be a regular file inside raw_root')
    return dict(subject_id=subject, source_image=str(source), source_age=source_age,
                target_age=target_age)


def check_external(model, config, *, runtime=True):
    """Check local assets/revision and SAM Ninja runtime without loading models."""
    if model not in ('sam', 'fading'):
        raise ValueError('Unknown feasibility model')
    repo = Path(config['external_repository']).resolve()
    if repo.is_relative_to(PROJECT_ROOT) or Path(config['checkpoint']).resolve().is_relative_to(PROJECT_ROOT):
        raise ValueError('External code/checkpoints must remain outside the project checkout')
    required = ('models/psp.py', 'datasets/augmentations.py', 'scripts/align_all_parallel.py') if model == 'sam' else ('age_editing.py', 'p2p.py', 'null_inversion.py')
    for relative in required:
        if not (repo / relative).is_file():
            raise FileNotFoundError(f'Missing external {model} code: {repo / relative}; see docs/model_feasibility.md')
    checkpoint = Path(config['checkpoint']).resolve()
    if model == 'sam':
        if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
            raise FileNotFoundError(f'Missing SAM checkpoint: {checkpoint}')
        predictor = Path(config['landmark_predictor'])
        if not predictor.is_file() or predictor.stat().st_size == 0:
            raise FileNotFoundError(f'Missing landmark predictor: {predictor}')
    else:
        for relative in ('model_index.json', 'unet/config.json', 'vae/config.json', 'scheduler/scheduler_config.json', 'text_encoder/config.json', 'tokenizer/tokenizer_config.json'):
            if not (checkpoint / relative).is_file():
                raise FileNotFoundError(f'Missing specialized FADING pipeline component: {checkpoint / relative}')
        for component in ('unet', 'vae', 'text_encoder'):
            if not any(p.is_file() and p.stat().st_size for pattern in ('*.bin', '*.safetensors') for p in (checkpoint / component).glob(pattern)):
                raise FileNotFoundError(f'Missing specialized FADING weights: {checkpoint / component}')
    python = Path(config['python'])
    if not python.is_file() or not os.access(python, os.X_OK):
        raise FileNotFoundError(f'Missing executable isolated Python: {python}')
    revision = subprocess.run(['git', '-C', str(repo), 'rev-parse', 'HEAD'], check=True,
                              capture_output=True, text=True).stdout.strip()
    if revision != config['revision']:
        raise ValueError(f'External revision differs: {revision}; expected {config["revision"]}')
    dirty = subprocess.run(['git', '-C', str(repo), 'status', '--porcelain', '--untracked-files=no'],
                           check=True, capture_output=True, text=True).stdout
    if dirty:
        raise ValueError('External tracked code is modified; record/review a new pinned revision')
    result = {'revision': revision, 'repository': str(repo), 'checkpoint': str(checkpoint)}
    if model == 'sam' and runtime:
        result['ninja'] = check_sam_runtime(config, worker_environment(model, config))
    return result


def worker_environment(model, config):
    """Copy CUDA/Colab settings; SAM uses its interpreter bin and headless Agg."""
    cache = Path(config['cache']).resolve()
    if cache.is_relative_to(PROJECT_ROOT):
        raise ValueError('Model caches must remain outside the project checkout')
    env = os.environ.copy()
    if model == 'sam':
        # Notebook inline backends are not portable to the isolated batch worker.
        env['MPLBACKEND'] = 'Agg'
        # Do not resolve the executable symlink: that would discard the venv bin.
        isolated_bin = str(Path(config['python']).absolute().parent)
        previous_path = env.get('PATH', '')
        env['PATH'] = isolated_bin + (os.pathsep + previous_path if previous_path else '')
    env.update(HF_HOME=str(cache / 'huggingface'), TORCH_HOME=str(cache / 'torch'),
               TORCH_EXTENSIONS_DIR=str(cache / 'extensions'), HF_HUB_OFFLINE='1',
               TRANSFORMERS_OFFLINE='1', PYTHONDONTWRITEBYTECODE='1')
    return env


def subprocess_backend(request):
    """Run in isolated interpreter; GPU metrics are measured inside that process."""
    config = request['config']
    check_external(request['model'], config)
    cache = Path(config['cache']).resolve()
    if cache.is_relative_to(PROJECT_ROOT):
        raise ValueError('Model caches must remain outside the project checkout')
    cache.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='smoke-', dir=cache) as temporary:
        request_path = Path(temporary) / 'request.json'
        result_path = Path(temporary) / 'result.json'
        request_path.write_text(json.dumps(request), encoding='utf-8')
        env = worker_environment(request['model'], config)
        completed = subprocess.run([config['python'], str(PROJECT_ROOT / 'scripts/model_worker.py'),
                                    str(request_path), str(result_path)],
                                   cwd=config['external_repository'], env=env, check=False)
        if result_path.is_file():
            return read_metadata(result_path)['configuration']
        raise RuntimeError(f'Worker exited {completed.returncode} without metadata; possible OOM/disconnection')


class Adapter:
    model = None

    def __init__(self, config, *, backend=None):
        self.config = dict(config)
        self.backend = backend or subprocess_backend

    def _generate(self, source_image, source_age, target_age, output_path):
        source, output = Path(source_image).resolve(), Path(output_path).absolute()
        metadata_path = output.with_suffix('.json')
        config = dict(self.config)
        record = SmokeMetadata('task-04a-smoke', config['method'], config['name'], str(source),
                                    source_age, target_age, config.get('seed', 0), config,
                                    {'requested': config.get('device')})
        # Refuse unsafe destinations before writing failure metadata.
        if output.suffix.lower() != '.png':
            raise ValueError('Smoke output must use .png')
        if output.is_symlink() or metadata_path.is_symlink():
            raise ValueError('Output/metadata symlinks are not allowed')
        output = output.resolve()
        metadata_path = output.with_suffix('.json')
        if output.is_relative_to(PROJECT_ROOT) or metadata_path == source or output == source:
            raise ValueError('Store generated images outside the checkout and preserve the source')
        if output.exists() and not output.is_file():
            raise ValueError('Output destination must be a regular file')
        if metadata_path.exists() and not metadata_path.is_file():
            raise ValueError('Metadata destination must be a regular file')
        expected = {k: asdict(record)[k] for k in ('experiment_id', 'configuration', 'source_image', 'source_age', 'target_age', 'seed')}
        if is_completed(output, metadata_path, expected):
            return read_metadata(metadata_path)
        if output.exists():
            raise FileExistsError('Existing unmatched output; use a new output path after inspecting metadata')
        record.output_path = str(output)
        record.precision = config.get('precision')
        try:
            if not source.is_file():
                raise FileNotFoundError(f'Missing source image: {source}')
            age(target_age, 'target_age')
            if source_age is not None:
                age(source_age, 'source_age')
            if self.model == 'fading':
                age(source_age, 'source_age')
                if target_age <= source_age:
                    raise ValueError('FADING target_age must be greater than source_age')
                if config.get('gender') not in ('female', 'male'):
                    raise ValueError('FADING requires researcher-supplied gender prompt: female or male; never inferred')
            seed = config.get('seed', 0)
            if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
                raise ValueError('seed must be an integer in [0, 2**32)')
            if self.model == 'fading' and seed != 0:
                raise ValueError('Pinned FADING entry point uses seed 0; other seeds need a reviewed bridge')
            if config.get('device') not in ('cpu', 'cuda:0'):
                raise ValueError('device must be cpu or cuda:0')
            if config.get('precision') != 'fp32':
                raise ValueError('Only fp32 is supported by this unverified upstream bridge; fp16 needs separate validation')
            if config.get('resolution') != (256 if self.model == 'sam' else 512):
                raise ValueError('Supported saved resolution is SAM=256 or FADING=512')
            if self.backend == subprocess_backend:
                if not config.get('allow_upstream_internal_sizes', False):
                    raise ValueError('Blocked by internal-size limits: SAM native decoder=1024; FADING prompt batch=2. Review docs before explicitly allowing.')
                if self.model == 'sam' and not config.get('trusted_checkpoint', False):
                    raise ValueError('Verify SAM checkpoint provenance before enabling trusted_checkpoint (pickle loading)')
            output.parent.mkdir(parents=True, exist_ok=True)
            record.status = 'pending'
            write_metadata(metadata_path, record)
            # Stage next to output so final replacement is atomic on supporting filesystems.
            with tempfile.TemporaryDirectory(prefix='.smoke-', dir=output.parent) as temporary:
                staged = Path(temporary) / 'output.png'
                result = self.backend(dict(model=self.model, config=config, source_image=str(source),
                                           source_age=source_age, target_age=target_age, output_path=str(staged)))
                record.model_version = result.get('revision')
                record.runtime = result.get('wall_clock_inference_seconds')
                record.device.update(result.get('device', {}))
                for key in ('output_resolution', 'wall_clock_inference_seconds', 'peak_cuda_memory_bytes',
                            'peak_cuda_reserved_bytes', 'upstream', 'versions', 'measurement_scope', 'test_only'):
                    if key in result:
                        setattr(record, key, result[key])
                if result.get('status') != 'completed':
                    raise RuntimeError(result.get('error_message') or 'Worker failed')
                if not staged.is_file() or not staged.stat().st_size:
                    raise RuntimeError('Backend produced no output')
                os.replace(staged, output)
            record.status = 'completed'
        except Exception as error:
            record.status = 'failed'
            record.error_message = f'{type(error).__name__}: {error}'
            write_metadata(metadata_path, record)
            raise
        write_metadata(metadata_path, record)
        return asdict(record)
