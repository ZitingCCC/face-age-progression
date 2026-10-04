"""SAM Ninja regression tests: CPU-only probes and fake executable, no models."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from src.models.feasibility import worker_environment
from src.models.sam_runtime import check_ninja, check_sam_runtime


@pytest.fixture
def isolated(tmp_path):
    # Reuse CPU test packages in a temporary venv layout. No package installation.
    root = tmp_path / 'sam env'
    (root / 'bin').mkdir(parents=True)
    (root / 'bin/python').symlink_to(sys.executable)
    (root / 'lib').symlink_to(Path(sys.prefix) / 'lib', target_is_directory=True)
    existing_cfg = Path(sys.prefix) / 'pyvenv.cfg'
    if existing_cfg.is_file():
        (root / 'pyvenv.cfg').write_text(existing_cfg.read_text())
    else:
        (root / 'pyvenv.cfg').write_text(
            f'home = {Path(sys.executable).parent}\ninclude-system-site-packages = true\n')
    ninja = root / 'bin/ninja'
    ninja.write_text('#!/bin/sh\nprintf "fake-ninja-test-only\\n"\n')
    ninja.chmod(0o755)
    return {'python': str(root / 'bin/python'), 'cache': str(tmp_path / 'cache')}, ninja


def test_sam_path_preserves_venv_symlink_and_inherited_settings(isolated, monkeypatch):
    config, _ = isolated
    monkeypatch.setenv('PATH', '/usr/local/cuda/bin:/usr/bin:/bin')
    monkeypatch.setenv('CUDA_HOME', '/usr/local/cuda')
    monkeypatch.setenv('LD_LIBRARY_PATH', '/cuda/lib64')
    monkeypatch.setenv('MAX_JOBS', '1')
    before = dict(os.environ)
    env = worker_environment('sam', config)
    assert env['PATH'] == str(Path(config['python']).parent) + os.pathsep + before['PATH']
    assert Path(config['python']).is_symlink()
    assert not env['PATH'].startswith(str(Path(config['python']).resolve().parent) + os.pathsep)
    for key in ('CUDA_HOME', 'LD_LIBRARY_PATH', 'MAX_JOBS'):
        assert env[key] == before[key]
    assert dict(os.environ) == before


def test_missing_parent_path(isolated, monkeypatch):
    config, _ = isolated
    monkeypatch.delenv('PATH', raising=False)
    assert worker_environment('sam', config)['PATH'] == str(Path(config['python']).parent)
    assert 'PATH' not in os.environ


def test_fading_path_unchanged(isolated, monkeypatch):
    config, _ = isolated
    monkeypatch.setenv('PATH', '/canonical/bin:/cuda/bin')
    assert worker_environment('fading', config)['PATH'] == os.environ['PATH']


def test_actual_pytorch_detection_reproduces_old_path_failure(isolated, monkeypatch):
    config, _ = isolated
    monkeypatch.setenv('PATH', '/no-system-ninja-in-this-test')
    code = 'from torch.utils.cpp_extension import is_ninja_available; print(is_ninja_available())'
    old = subprocess.run([config['python'], '-c', code], env=dict(os.environ),
                         capture_output=True, text=True, check=True)
    corrected = subprocess.run([config['python'], '-c', code], env=worker_environment('sam', config),
                               capture_output=True, text=True, check=True)
    assert old.stdout.strip() == 'False'
    assert corrected.stdout.strip() == 'True'


def test_available_fake_ninja_probe_without_models_or_gpu(isolated, monkeypatch):
    config, ninja = isolated
    monkeypatch.setenv('PATH', '/no-system-ninja-in-this-test')
    report = check_sam_runtime(config, worker_environment('sam', config))
    assert report['python'] == config['python']
    assert report['prefix'] == str(Path(config['python']).parents[1])
    assert report['ninja_executable'] == str(ninja)
    assert report['ninja_version'] == 'fake-ninja-test-only'
    assert report['torch_ninja_available'] is True
    assert report['torch_version']
    assert not Path(config['cache']).exists()  # Probe creates no build/cache assets.


def test_missing_ninja_actionable_before_torch_import(isolated, monkeypatch):
    config, ninja = isolated
    ninja.unlink()
    monkeypatch.setenv('PATH', '/no-system-ninja-in-this-test')
    with pytest.raises(RuntimeError, match='Ninja executable is missing.*--ignore-installed --no-deps ninja'):
        check_sam_runtime(config, worker_environment('sam', config))


def test_broken_ninja_executable(isolated, monkeypatch):
    config, ninja = isolated
    ninja.write_text('#!/bin/sh\nexit 1\n')
    monkeypatch.setenv('PATH', str(ninja.parent))
    with pytest.raises(RuntimeError, match='cannot run --version'):
        check_ninja()


def test_pytorch_rejects_found_ninja(isolated, monkeypatch):
    _, ninja = isolated
    monkeypatch.setenv('PATH', str(ninja.parent))
    from torch.utils import cpp_extension
    monkeypatch.setattr(cpp_extension, 'is_ninja_available', lambda: False)
    with pytest.raises(RuntimeError, match='PyTorch does not recognize Ninja'):
        check_ninja()


def test_probe_subprocess_uses_exact_environment_no_shell(isolated, monkeypatch):
    config, _ = isolated
    env = worker_environment('sam', config)
    def run(args, **kwargs):
        assert args[0] == config['python'] and args[1] == '-c'
        assert kwargs['env'] is env
        assert kwargs['timeout'] == 30
        assert not kwargs.get('shell', False)
        return subprocess.CompletedProcess(args, 0, stdout='{"torch_ninja_available": true}', stderr='')
    monkeypatch.setattr('src.models.sam_runtime.subprocess.run', run)
    assert check_sam_runtime(config, env)['torch_ninja_available']


def test_probe_timeout_actionable(isolated, monkeypatch):
    config, _ = isolated
    def run(args, **kwargs):
        raise subprocess.TimeoutExpired(args, 30)
    monkeypatch.setattr('src.models.sam_runtime.subprocess.run', run)
    with pytest.raises(RuntimeError, match='preflight could not run'):
        check_sam_runtime(config, worker_environment('sam', config))


def test_worker_checks_ninja_before_model_or_cuda(tmp_path, monkeypatch):
    from scripts import model_worker as worker
    config = dict(name='SAM', method='M0-candidate', python=sys.executable, device='cuda:0')
    request = tmp_path / 'request.json'
    result = tmp_path / 'result.json'
    request.write_text(json.dumps(dict(model='sam', config=config, source_image='unused.jpg',
                                       source_age=5, target_age=30, output_path='unused.png')))
    monkeypatch.setattr(sys, 'argv', ['worker', str(request), str(result)])
    def external(model, settings, *, runtime):
        assert model == 'sam' and runtime is False
        return {'revision': 'fake-upstream'}
    def missing():
        raise RuntimeError('test-only: Ninja executable is missing from worker PATH')
    monkeypatch.setattr(worker, 'check_external', external)
    monkeypatch.setattr(worker, 'check_ninja', missing)
    monkeypatch.setattr(worker, 'sam', lambda *_: pytest.fail('Model loaded before Ninja preflight'))
    import torch
    monkeypatch.setattr(torch, 'load', lambda *_a, **_k: pytest.fail('Checkpoint loaded'))
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: pytest.fail('CUDA queried before Ninja preflight'))
    assert worker.main() == 1
    record = json.loads(result.read_text())['configuration']
    assert record['status'] == 'failed' and 'Ninja executable is missing' in record['error_message']
    assert record['wall_clock_inference_seconds'] is None
    assert record['peak_cuda_memory_bytes'] is None


def test_runtime_module_import_is_lazy():
    code = '''
import sys, subprocess, urllib.request
subprocess.run = lambda *a, **k: (_ for _ in ()).throw(AssertionError('subprocess at import'))
urllib.request.urlopen = lambda *a, **k: (_ for _ in ()).throw(AssertionError('download at import'))
from src.models.sam_runtime import check_ninja
from scripts import model_worker
assert all(name not in sys.modules for name in ('torch', 'diffusers', 'models.psp', 'PIL'))
'''
    subprocess.run([sys.executable, '-c', code], check=True)
