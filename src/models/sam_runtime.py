"""CPU-safe Ninja probe; no torch/upstream/model imports at module import time."""
import json
from pathlib import Path
import shutil
import subprocess
import sys


PROBE = '''import json, sys
sys.path.insert(0, sys.argv[1])
from src.models.sam_runtime import check_ninja
print(json.dumps(check_ninja()))
'''


def check_ninja():
    """Check executable and PyTorch detection in the current process's PATH."""
    remedy = (f'Install Ninja locally with {sys.executable} -m pip install '
              '--ignore-installed --no-deps ninja, then rerun scripts/smoke_sam.py --check. '
              'Do not replace canonical PyTorch or patch the SAM checkout.')
    executable = shutil.which('ninja')
    if executable is None:
        raise RuntimeError(f'SAM dependency preflight: Ninja executable is missing from worker PATH. {remedy}')
    try:
        result = subprocess.run([executable, '--version'], check=True, capture_output=True,
                                text=True, timeout=10)
    except (OSError, subprocess.SubprocessError) as error:
        raise RuntimeError(f'SAM dependency preflight: Ninja at {executable} cannot run --version. {remedy}') from error
    try:
        import torch
        from torch.utils.cpp_extension import is_ninja_available
        available = is_ninja_available()
    except Exception as error:
        raise RuntimeError('SAM dependency preflight: cannot check torch.utils.cpp_extension '
                           f'in {sys.executable}: {type(error).__name__}: {error}') from error
    if not available:
        raise RuntimeError(f'SAM dependency preflight: PyTorch does not recognize Ninja at {executable}. {remedy}')
    return dict(python=sys.executable, prefix=sys.prefix, ninja_executable=executable,
                ninja_version=result.stdout.strip(), torch_version=torch.__version__,
                torch_ninja_available=True)


def check_sam_runtime(config, env):
    """Probe configured Python with the exact environment used by the worker."""
    root = Path(__file__).resolve().parents[2]
    try:
        result = subprocess.run([config['python'], '-c', PROBE, str(root)], env=env, cwd=root,
                                capture_output=True, text=True, check=False, timeout=30)
    except (OSError, subprocess.SubprocessError) as error:
        raise RuntimeError(f'SAM dependency preflight could not run {config["python"]}: {error}') from error
    if result.returncode:
        detail = result.stderr.strip() or result.stdout.strip() or f'exit {result.returncode}'
        raise RuntimeError(f'SAM dependency preflight failed in {config["python"]}: {detail}')
    return json.loads(result.stdout)
