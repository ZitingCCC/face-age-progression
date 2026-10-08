"""Synthetic CPU failures preserve frames; no upstream models or real GPU work."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from src.models import feasibility
from src.models.fading_adapter import FADINGAdapter


@pytest.fixture
def upstream(tmp_path):
    path = tmp_path / 'upstream_failure.py'
    path.write_text(
        "import os\n"
        "def fail():\n"
        "    private_token = os.environ.get('TEST_DIAGNOSTIC_SECRET')\n"
        "    model_weights = b'private-weight-local'\n"
        "    try:\n"
        "        raise ValueError('synthetic original cause')\n"
        "    except ValueError as cause:\n"
        "        raise IndexError('synthetic two-dimensional array') from cause\n"
        "fail()\n")
    return path


BOOTSTRAP = '''
import runpy, sys
from scripts import model_worker as worker
failure = sys.argv[3]
worker.check_external = lambda *a, **k: runpy.run_path(failure)
sys.argv = ['model_worker', sys.argv[1], sys.argv[2]]
raise SystemExit(worker.main())
'''


@pytest.mark.parametrize('model', ['sam', 'fading'])
def test_worker_original_frames_and_sam_unchanged(tmp_path, upstream, monkeypatch, model):
    request, result = tmp_path / 'request.json', tmp_path / 'result.json'
    request.write_text(json.dumps(dict(model=model, source_image='unused.jpg', source_age=2,
                                      target_age=30, output_path=str(tmp_path / 'absent.png'),
                                      config=dict(method='test', name=model))))
    monkeypatch.setenv('TEST_DIAGNOSTIC_SECRET', 'sensitive-test-value-not-for-output')
    child = subprocess.run([sys.executable, '-c', BOOTSTRAP, str(request), str(result), str(upstream)],
                           capture_output=True, text=True, check=False)
    assert child.returncode == 1
    record = json.loads(result.read_text())
    payload = record['configuration']
    assert record['status'] == payload['status'] == 'failed'
    assert payload['error_message'] == 'IndexError: synthetic two-dimensional array'
    assert payload['wall_clock_inference_seconds'] is None
    assert payload['peak_cuda_memory_bytes'] is None
    assert not (tmp_path / 'absent.png').exists()
    if model == 'fading':
        trace = payload['error_traceback']
        assert f'File "{upstream}", line 8, in fail' in trace
        assert 'IndexError: synthetic two-dimensional array' in trace
        assert 'ValueError: synthetic original cause' in trace
        assert 'direct cause' in trace
        assert child.stderr == trace
    else:
        assert 'error_traceback' not in payload
        assert child.stderr == ''
    for diagnostic in (result.read_text(), child.stderr):
        assert 'sensitive-test-value-not-for-output' not in diagnostic
        assert 'private-weight-local' not in diagnostic


def settings(tmp_path):
    return dict(name='FADING', method='test', device='cpu', precision='fp32', resolution=512,
                gender='male', seed=0, allow_upstream_internal_sizes=True,
                external_repository=str(tmp_path), cache=str(tmp_path / 'cache'), python=sys.executable)


def test_trace_survives_worker_parent_and_atomic_drive_metadata(tmp_path, upstream, monkeypatch, capfd):
    source, output = tmp_path / 'source.jpg', tmp_path / 'output.png'
    source.write_text('synthetic source path, not FG-NET')
    config = settings(tmp_path)
    monkeypatch.setenv('TEST_DIAGNOSTIC_SECRET', 'sensitive-test-value-not-for-output')
    parent = dict(os.environ)
    original_run = subprocess.run
    worker_result = {}
    monkeypatch.setattr(feasibility, 'check_external', lambda *_: {'revision': 'test'})
    def run(args, **kwargs):
        assert args[1].endswith('scripts/model_worker.py')
        assert not kwargs.get('shell', False)
        child = original_run([args[0], '-c', BOOTSTRAP, *args[2:], str(upstream)],
                             **dict(kwargs, cwd=str(feasibility.PROJECT_ROOT)))
        worker_result.update(json.loads(Path(args[-1]).read_text())['configuration'])
        return child
    monkeypatch.setattr(feasibility.subprocess, 'run', run)
    with pytest.raises(RuntimeError, match='IndexError: synthetic two-dimensional array'):
        FADINGAdapter(config).generate(source, 2, 30, output)
    record = json.loads(output.with_suffix('.json').read_text())
    assert record['status'] == 'failed'
    assert record['error_message'] == 'RuntimeError: IndexError: synthetic two-dimensional array'
    assert record['error_traceback'] == worker_result['error_traceback']
    assert f'File "{upstream}", line 8, in fail' in record['error_traceback']
    assert capfd.readouterr().err == worker_result['error_traceback']
    assert not output.exists()
    assert not list(output.parent.glob('.smoke-*'))
    assert record['configuration'] == config
    assert dict(os.environ) == parent
    assert 'sensitive-test-value-not-for-output' not in json.dumps(record)


def test_parent_only_failure_trace_and_retry_success_unchanged(tmp_path):
    source, output = tmp_path / 'source.jpg', tmp_path / 'output.png'
    source.write_text('synthetic source path')
    def failure(_):
        raise OSError('synthetic parent failure')
    config = settings(tmp_path)
    with pytest.raises(OSError):
        FADINGAdapter(config, backend=failure).generate(source, 2, 30, output)
    record = json.loads(output.with_suffix('.json').read_text())
    assert 'OSError: synthetic parent failure' in record['error_traceback']
    assert record['error_message'] == 'OSError: synthetic parent failure'
    assert not output.exists()
    calls = []
    def success(request):
        calls.append(request)
        Path(request['output_path']).write_text('synthetic output, not a generated face')
        return dict(status='completed', test_only=True)
    adapter = FADINGAdapter(config, backend=success)
    completed = adapter.generate(source, 2, 30, output)
    assert completed['status'] == 'completed'
    assert 'error_traceback' not in completed
    assert completed['error_message'] is None
    assert adapter.generate(source, 2, 30, output) == completed
    assert len(calls) == 1
