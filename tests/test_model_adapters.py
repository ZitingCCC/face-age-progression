"""CPU-only fakes: no generated faces, real images, upstream imports or downloads."""
import csv
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from src.models.sam_adapter import SAMAdapter
from src.models.fading_adapter import FADINGAdapter
from src.models.feasibility import check_external, select_source


@pytest.fixture
def assets(tmp_path):
    source = tmp_path / '001A05.jpg'
    source.write_bytes(b'synthetic path fixture, not an image')
    return source, tmp_path / 'outputs' / 'smoke.png'


def config(model):
    return dict(name=model.upper(), method='candidate', device='cpu', precision='fp32',
                resolution=256 if model == 'sam' else 512, gender='female', seed=0,
                external_repository='/missing/external', checkpoint='/missing/checkpoint',
                python=sys.executable, revision='0'*40, cache='/missing/cache',
                landmark_predictor='/missing/predictor')


def fake(request):
    Path(request['output_path']).write_bytes(b'test-only synthetic output, not a generated face')
    return dict(status='completed', test_only=True, wall_clock_inference_seconds=None,
                peak_cuda_memory_bytes=None, output_resolution=None, revision='fake-revision')


@pytest.mark.parametrize('model', ['sam', 'fading'])
def test_fake_smoke_and_resume(assets, model):
    source, output = assets
    calls = []
    def backend(request):
        calls.append(request)
        return fake(request)
    adapter = (SAMAdapter if model == 'sam' else FADINGAdapter)(config(model), backend=backend)
    result = adapter._generate(source, 5, 30, output)
    assert result['status'] == 'completed' and result['test_only']
    assert result['runtime'] is None and result['peak_cuda_memory_bytes'] is None
    assert result['wall_clock_inference_seconds'] is None
    assert result['output_path'] == str(output)
    assert result['source_age'] == 5 and result['target_age'] == 30
    assert result['precision'] == 'fp32' and result['model_version'] == 'fake-revision'
    assert json.loads(output.with_suffix('.json').read_text()) == result
    assert adapter._generate(source, 5, 30, output) == result
    assert len(calls) == 1
    with pytest.raises(FileExistsError):
        adapter._generate(source, 5, 31, output)


@pytest.mark.parametrize('value', [-1, 101, 5.5, True, '30', None])
@pytest.mark.parametrize('model', ['sam', 'fading'])
def test_invalid_target(assets, value, model):
    source, output = assets
    adapter = (SAMAdapter if model == 'sam' else FADINGAdapter)(config(model), backend=fake)
    with pytest.raises(ValueError, match='target_age'):
        adapter._generate(source, 5, value, output)
    record = json.loads(output.with_suffix('.json').read_text())
    assert record['status'] == 'failed' and record['runtime'] is None
    assert record['peak_cuda_memory_bytes'] is None and not output.exists()


@pytest.mark.parametrize('source_age,target', [(5, 5), (5, 4), (-1, 30), (101, 30), (None, 30), (True, 30)])
def test_fading_ages(assets, source_age, target):
    source, output = assets
    with pytest.raises(ValueError):
        FADINGAdapter(config('fading'), backend=fake).generate(source, source_age, target, output)


@pytest.mark.parametrize('model', ['sam', 'fading'])
def test_missing_source(assets, model):
    source, output = assets
    source.unlink()
    with pytest.raises(FileNotFoundError, match='source'):
        (SAMAdapter if model == 'sam' else FADINGAdapter)(config(model), backend=fake)._generate(source, 5, 30, output)
    assert json.loads(output.with_suffix('.json').read_text())['status'] == 'failed'


def test_failed_worker_metrics_preserved(assets):
    source, output = assets
    def failing(request):
        return dict(status='failed', error_message='simulated dependency failure', versions={'fake': 'test'},
                    wall_clock_inference_seconds=None, peak_cuda_memory_bytes=None)
    with pytest.raises(RuntimeError, match='simulated'):
        SAMAdapter(config('sam'), backend=failing).generate(source, 30, output)
    record = json.loads(output.with_suffix('.json').read_text())
    assert record['versions'] == {'fake': 'test'}
    assert record['status'] == 'failed' and 'simulated' in record['error_message']
    assert not output.exists()


def test_empty_backend_output(assets):
    source, output = assets
    with pytest.raises(RuntimeError, match='no output'):
        SAMAdapter(config('sam'), backend=lambda _: {'status': 'completed'}).generate(source, 30, output)


@pytest.mark.parametrize('field,value', [('device', 'cuda:99'), ('precision', 'fp16'), ('resolution', 1024)])
def test_config_validation(assets, field, value):
    source, output = assets
    settings = config('sam')
    settings[field] = value
    with pytest.raises(ValueError):
        SAMAdapter(settings, backend=fake).generate(source, 30, output)


def test_gender_required(assets):
    source, output = assets
    settings = config('fading')
    settings['gender'] = None
    with pytest.raises(ValueError, match='gender'):
        FADINGAdapter(settings, backend=fake).generate(source, 5, 30, output)


def test_output_protection(assets):
    source, output = assets
    adapter = SAMAdapter(config('sam'), backend=fake)
    with pytest.raises(ValueError, match='.png'):
        adapter.generate(source, 30, output.with_suffix('.jpg'))
    output.parent.mkdir()
    output.symlink_to(source)
    with pytest.raises(ValueError, match='symlink'):
        adapter.generate(source, 30, output)
    assert source.read_bytes().startswith(b'synthetic')


def test_no_import_side_effects():
    code = '''
import sys
import urllib.request
import subprocess
subprocess.run = lambda *a, **k: (_ for _ in ()).throw(AssertionError('subprocess at import'))
urllib.request.urlopen = lambda *a, **k: (_ for _ in ()).throw(AssertionError('download at import'))
from src.models.sam_adapter import SAMAdapter
from src.models.fading_adapter import FADINGAdapter
assert all(x not in sys.modules for x in ('torch', 'diffusers', 'models.psp', 'PIL'))
'''
    subprocess.run([sys.executable, '-c', code], check=True)


@pytest.mark.parametrize('model', ['sam', 'fading'])
def test_missing_external_repository(model):
    with pytest.raises(FileNotFoundError, match='external'):
        check_external(model, config(model))


@pytest.mark.parametrize('model', ['sam', 'fading'])
def test_missing_checkpoint(tmp_path, model):
    settings = config(model)
    settings['external_repository'] = str(tmp_path)
    for relative in (('models/psp.py', 'datasets/augmentations.py', 'scripts/align_all_parallel.py') if model == 'sam' else ('age_editing.py', 'p2p.py', 'null_inversion.py')):
        file = tmp_path / relative
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text('# fake')
    with pytest.raises(FileNotFoundError, match='checkpoint|pipeline'):
        check_external(model, settings)


def test_deterministic_selection_and_escape(tmp_path):
    raw = tmp_path / 'raw'
    raw.mkdir()
    for name in ('002A03.jpg', '001A05.jpg'):
        (raw / name).write_bytes(b'fake')
    metadata = tmp_path / 'metadata.csv'
    metadata.write_text('subject_id,age,image_path,split\n002,3,002A03.jpg,test\n001,5,001A05.jpg,train\n')
    selected = select_source(metadata, raw)
    assert selected['subject_id'] == '001' and selected['source_age'] == 5
    assert selected['target_age'] == 30
    metadata.write_text('subject_id,age,image_path,split\n001,5,../metadata.csv,train\n')
    with pytest.raises(ValueError, match='inside raw_root'):
        select_source(metadata, raw)


def test_real_backend_missing_assets_records_failure(assets):
    source, output = assets
    settings = config('sam')
    settings.update(allow_upstream_internal_sizes=True, trusted_checkpoint=True)
    with pytest.raises(FileNotFoundError, match='external'):
        SAMAdapter(settings).generate(source, 30, output)
    record = json.loads(output.with_suffix('.json').read_text())
    assert record['status'] == 'failed' and record['runtime'] is None
    assert record['output_path'] == str(output) and not output.exists()


def test_internal_size_gate(assets):
    source, output = assets
    with pytest.raises(ValueError, match='internal-size'):
        SAMAdapter(config('sam')).generate(source, 30, output)


@pytest.mark.parametrize('model', ['sam', 'fading'])
def test_subprocess_handoff_isolated_and_offline(tmp_path, monkeypatch, model):
    from src.models import feasibility
    monkeypatch.setenv('MPLBACKEND', 'module://matplotlib_inline.backend_inline')
    settings = config(model)
    settings['cache'] = str(tmp_path / 'cache with spaces')
    settings['external_repository'] = str(tmp_path / 'external with spaces')
    settings['upstream_repository'] = 'https://example.invalid/upstream'
    output = tmp_path / 'output.png'
    request = dict(model=model, config=settings, source_image='/source with spaces;literal.jpg',
                   source_age=5, target_age=30, output_path=str(output))
    monkeypatch.setattr(feasibility, 'check_external', lambda *_: {'revision': 'fake'})
    def run(args, **kwargs):
        assert isinstance(args, list) and not kwargs.get('shell', False)
        assert args[0] == sys.executable
        assert kwargs['cwd'] == settings['external_repository']
        if model == 'sam':
            assert kwargs['env']['PATH'].startswith(str(Path(settings['python']).absolute().parent) + os.pathsep)
            assert kwargs['env']['PATH'].endswith(os.environ.get('PATH', ''))
        else:
            assert kwargs['env']['PATH'] == os.environ.get('PATH', '')
        assert kwargs['env']['MPLBACKEND'] == 'Agg'
        assert os.environ['MPLBACKEND'] == 'module://matplotlib_inline.backend_inline'
        assert kwargs['env']['HF_HUB_OFFLINE'] == '1'
        assert kwargs['env']['TRANSFORMERS_OFFLINE'] == '1'
        assert kwargs['env']['TORCH_EXTENSIONS_DIR'].startswith(settings['cache'])
        transferred = json.loads(Path(args[-2]).read_text())
        assert transferred == request
        from src.utils.metadata import ExperimentMetadata, write_metadata
        payload = fake(transferred)
        write_metadata(args[-1], ExperimentMetadata('fake', 'fake', 'fake', 'fake', 5, 30, 0, payload, {}))
        return subprocess.CompletedProcess(args, 0)
    monkeypatch.setattr(feasibility.subprocess, 'run', run)
    result = feasibility.subprocess_backend(request)
    assert result['test_only'] and result['peak_cuda_memory_bytes'] is None
    assert not list(Path(settings['cache']).glob('smoke-*'))


def test_worker_killed_has_clear_parent_failure(tmp_path, monkeypatch):
    from src.models import feasibility
    settings = config('sam')
    settings['cache'] = str(tmp_path / 'cache')
    monkeypatch.setattr(feasibility, 'check_external', lambda *_: {})
    monkeypatch.setattr(feasibility.subprocess, 'run', lambda args, **kwargs: subprocess.CompletedProcess(args, -9))
    with pytest.raises(RuntimeError, match='without metadata'):
        feasibility.subprocess_backend(dict(model='sam', config=settings))


def test_external_revision_and_weights_check(tmp_path, monkeypatch):
    monkeypatch.setattr("src.models.feasibility.check_sam_runtime", lambda *_: {"test_only": True})
    repo = tmp_path / 'external'
    repo.mkdir()
    for relative in ('models/psp.py', 'datasets/augmentations.py', 'scripts/align_all_parallel.py'):
        file = repo / relative
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text('# fake external source\n')
    subprocess.run(['git', 'init', '-q', str(repo)], check=True)
    subprocess.run(['git', '-C', str(repo), 'add', '.'], check=True)
    subprocess.run(['git', '-C', str(repo), '-c', 'user.name=Test', '-c', 'user.email=test@example.invalid',
                    'commit', '-qm', 'fake fixture'], check=True)
    revision = subprocess.run(['git', '-C', str(repo), 'rev-parse', 'HEAD'], check=True, capture_output=True, text=True).stdout.strip()
    checkpoint, predictor = tmp_path / 'fake.pt', tmp_path / 'fake.dat'
    checkpoint.write_bytes(b'fixture, not weights')
    predictor.write_bytes(b'fixture, not predictor')
    settings = dict(config('sam'), external_repository=str(repo), checkpoint=str(checkpoint),
                    landmark_predictor=str(predictor), revision=revision)
    assert check_external('sam', settings)['revision'] == revision
    settings['revision'] = 'f'*40
    with pytest.raises(ValueError, match='revision differs'):
        check_external('sam', settings)
    settings['revision'] = revision
    (repo / 'models/psp.py').write_text('# modified')
    with pytest.raises(ValueError, match='modified'):
        check_external('sam', settings)


def test_specialized_pipeline_weights_required(tmp_path):
    settings = config('fading')
    repo, pipeline = tmp_path / 'external', tmp_path / 'pipeline'
    repo.mkdir()
    for name in ('age_editing.py', 'p2p.py', 'null_inversion.py'):
        (repo / name).write_text('# fake')
    for relative in ('model_index.json', 'unet/config.json', 'vae/config.json', 'scheduler/scheduler_config.json', 'text_encoder/config.json', 'tokenizer/tokenizer_config.json'):
        file = pipeline / relative
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text('{}')
    settings.update(external_repository=str(repo), checkpoint=str(pipeline))
    with pytest.raises(FileNotFoundError, match='weights'):
        check_external('fading', settings)
