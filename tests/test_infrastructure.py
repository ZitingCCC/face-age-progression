import random
from pathlib import Path
import numpy as np
import pytest
import torch
import yaml
from src.utils.config import load_config
from src.utils.device import detect_device
from src.utils.metadata import ExperimentMetadata, read_metadata, write_metadata
from src.utils.paths import configure_paths
from src.utils.reproducibility import set_seed
from src.utils.resume import output_exists, is_completed

ROOT = Path(__file__).resolve().parents[1]

@pytest.mark.parametrize('name', ['diffusion', 'identity', 'experiment'])
def test_yaml_loading(name):
    assert isinstance(load_config(ROOT / 'configs' / f'{name}.yaml'), dict)

@pytest.mark.parametrize('content', ['', '- item', '!!python/object:builtins.object {}'])
def test_yaml_rejects_invalid_documents(tmp_path, content):
    path = tmp_path / 'config.yaml'
    path.write_text(content)
    with pytest.raises((ValueError, yaml.YAMLError)):
        load_config(path)

def test_deterministic_seeds(monkeypatch):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    def sample():
        return random.random(), np.random.rand(4), torch.rand(4)
    set_seed(42)
    first = sample()
    set_seed(42)
    second = sample()
    assert first[0] == second[0]
    np.testing.assert_array_equal(first[1], second[1])
    assert torch.equal(first[2], second[2])
    set_seed(43)
    assert sample()[0] != first[0]
    assert torch.are_deterministic_algorithms_enabled()

@pytest.mark.parametrize('seed', [-1, 2**32, True, 1.5])
def test_seed_validation(seed):
    with pytest.raises(ValueError):
        set_seed(seed)

def test_device_cpu_fallback(monkeypatch, capsys):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    monkeypatch.setattr(torch.cuda, 'get_device_properties', lambda *_: pytest.fail('GPU queried'))
    assert detect_device().type == 'cpu'
    output = capsys.readouterr().out
    assert f'PyTorch version: {torch.__version__}' in output
    assert 'CUDA available: False' in output
    assert 'Selected device: cpu' in output

def metadata(status='completed'):
    return ExperimentMetadata('test-only', 'infrastructure-test', None, 'fixture',
        10, 20, 42, {'nested': {'steps': 30}}, {'device': 'cpu'},
        status=status, error_message='test message')

def test_metadata_serialization(tmp_path):
    path = tmp_path / 'nested' / 'metadata.json'
    write_metadata(path, metadata())
    record = read_metadata(path)
    assert record['configuration'] == {'nested': {'steps': 30}}
    assert record['error_message'] == 'test message'
    assert record['runtime'] is None
    assert set(record) == set(metadata().__dict__)
    write_metadata(path, metadata('failed'))
    assert read_metadata(path)['status'] == 'failed'
    assert list(path.parent.iterdir()) == [path]

def test_metadata_rejects_nan_without_overwrite(tmp_path):
    path = tmp_path / 'metadata.json'
    write_metadata(path, metadata())
    record = metadata()
    record.runtime = float('nan')
    with pytest.raises(ValueError):
        write_metadata(path, record)
    assert read_metadata(path)['runtime'] is None

def test_local_paths(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert configure_paths().project == ROOT
    paths = configure_paths(project_root=tmp_path, storage_root='persistent')
    assert paths.storage == tmp_path / 'persistent'
    paths.create_output_dirs()
    assert all(p.is_dir() for p in [paths.metrics, paths.figures, paths.generated, paths.metadata])
    assert paths.datasets == tmp_path / 'persistent' / 'datasets'

def test_colab_paths():
    paths = configure_paths('colab')
    assert paths.project == Path('/content/face-age-progression')
    assert paths.storage == paths.project

def test_drive_paths(tmp_path):
    paths = configure_paths('drive', project_root=tmp_path, drive_root=tmp_path / 'mounted-drive')
    assert paths.storage == tmp_path / 'mounted-drive'
    assert not paths.storage.exists()
    assert configure_paths('drive', project_root=tmp_path, storage_root='override').storage == tmp_path / 'override'
    with pytest.raises(ValueError):
        configure_paths('drive')
    with pytest.raises(ValueError):
        configure_paths('unknown')

def test_resume_output_detection(tmp_path):
    output, record = tmp_path / 'output.bin', tmp_path / 'record.json'
    assert not output_exists(output)
    assert not output_exists(tmp_path)
    output.touch()
    assert not output_exists(output)
    output.write_bytes(b'test fixture, not an experimental result')
    assert output_exists(output)
    assert not is_completed(output, record)
    for invalid in ['invalid JSON', '[]']:
        record.write_text(invalid)
        assert not is_completed(output, record)
    write_metadata(record, metadata('failed'))
    assert not is_completed(output, record)
    write_metadata(record, metadata())
    assert is_completed(output, record, {'experiment_id': 'test-only', 'seed': 42})
    assert not is_completed(output, record, {'seed': 43})
    output.unlink()
    assert not is_completed(output, record)
