"""Tiny synthetic inputs only: RGB staging without models or external mutation."""
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image
import pytest

from src.models import feasibility
from src.models.fading_adapter import FADINGAdapter
from src.models.fading_runtime import normalize_source


@pytest.mark.parametrize('mode', ['L', 'RGB', 'RGBA', 'P', 'CMYK'])
def test_lossless_pillow_rgb_conversion_preserves_source(tmp_path, mode):
    source, staged = tmp_path / 'source.tif', tmp_path / 'stage.png'
    image = Image.new(mode, (4, 3))
    if mode == 'P':
        image.putpalette([channel for i in range(256) for channel in (i, 255-i, i//2)])
    values = {'L': 73, 'P': 73, 'RGB': (10, 20, 30), 'RGBA': (10, 20, 30, 0),
              'CMYK': (10, 20, 30, 40)}
    image.putpixel((1, 1), values[mode])
    image.save(source)
    original = source.read_bytes()
    with Image.open(source) as image:
        expected = np.array(image.convert('RGB'))
    note = normalize_source(source, staged)
    assert source.read_bytes() == original
    with Image.open(source) as image:
        assert image.mode == mode and image.size == (4, 3)
        if mode == 'L':
            assert np.array(image).shape == (3, 4)
    with Image.open(staged) as image:
        assert image.mode == 'RGB' and image.format == 'PNG' and image.size == (4, 3)
        np.testing.assert_array_equal(np.array(image), expected)
        assert np.array(image)[:, :, :3].shape == (3, 4, 3)
    if mode == 'RGBA':
        assert tuple(expected[1, 1]) == (10, 20, 30)  # Pillow drops alpha; no compositing.
    assert note == dict(method="Pillow.Image.convert('RGB')", original_mode=mode,
                        staged_mode='RGB', staged_format='PNG', size=[4, 3])


def config(tmp_path):
    return dict(name='FADING', method='test', device='cpu', precision='fp32', resolution=512,
                gender='male', seed=0, allow_upstream_internal_sizes=True,
                external_repository=str(tmp_path / 'external'), cache=str(tmp_path / 'cache'),
                python=sys.executable)


@pytest.mark.parametrize('status', ['completed', 'failed'])
def test_staging_lifecycle_provenance_and_failure_trace(tmp_path, monkeypatch, status):
    source, output = tmp_path / 'original.png', tmp_path / 'output.png'
    Image.new('L', (4, 3), 64).save(source)
    original = source.read_bytes()
    settings = config(tmp_path)
    repo = Path(settings['external_repository'])
    repo.mkdir()
    sentinel = repo / 'untouched.py'
    sentinel.write_text('# unchanged external source\n')
    monkeypatch.setattr(feasibility, 'check_external', lambda *_: {'revision': 'test'})
    seen = []
    def worker(args, **kwargs):
        assert not kwargs.get('shell', False)
        request = json.loads(Path(args[-2]).read_text())
        staged = Path(request['fading_source_image'])
        seen.append(staged)
        assert staged.is_relative_to(Path(settings['cache']))
        assert not staged.is_relative_to(feasibility.PROJECT_ROOT)
        assert request['source_image'] == str(source)
        with Image.open(staged) as image:
            assert image.mode == 'RGB' and image.format == 'PNG'
        payload = dict(status=status, source_preprocessing=request['source_preprocessing'],
                       test_only=True, error_message='IndexError: synthetic failure',
                       error_traceback='Traceback: synthetic upstream.py:23\nIndexError: synthetic failure')
        if status == 'completed':
            Path(request['output_path']).write_text('synthetic output, not a face')
        from src.utils.metadata import ExperimentMetadata, write_metadata
        write_metadata(args[-1], ExperimentMetadata('test', 'test', 'test', str(source), 2, 30, 0, payload, {}))
        return subprocess.CompletedProcess(args, 0 if status == 'completed' else 1)
    monkeypatch.setattr(feasibility.subprocess, 'run', worker)
    adapter = FADINGAdapter(settings)
    if status == 'completed':
        adapter.generate(source, 2, 30, output)
    else:
        with pytest.raises(RuntimeError, match='synthetic failure'):
            adapter.generate(source, 2, 30, output)
    metadata = json.loads(output.with_suffix('.json').read_text())
    assert metadata['source_image'] == str(source)
    assert metadata['source_preprocessing']['original_mode'] == 'L'
    assert metadata['configuration'] == settings
    assert metadata['status'] == status
    if status == 'failed':
        assert metadata['error_traceback'].endswith('IndexError: synthetic failure')
        assert not output.exists()
    assert all(not staged.exists() for staged in seen)
    assert not list(Path(settings['cache']).glob('smoke-*'))
    assert source.read_bytes() == original
    assert sentinel.read_text() == '# unchanged external source\n'
    assert sorted(p.name for p in repo.iterdir()) == ['untouched.py']


def test_decode_failure_before_worker_launch(tmp_path, monkeypatch):
    source, output = tmp_path / 'corrupt.jpg', tmp_path / 'output.png'
    source.write_bytes(b'not an image')
    settings = config(tmp_path)
    monkeypatch.setattr(feasibility, 'check_external', lambda *_: {'revision': 'test'})
    monkeypatch.setattr(feasibility.subprocess, 'run', lambda *_a, **_k: pytest.fail('Worker launched before decode'))
    with pytest.raises(ValueError, match='Cannot decode FADING source image'):
        FADINGAdapter(settings).generate(source, 2, 30, output)
    metadata = json.loads(output.with_suffix('.json').read_text())
    assert metadata['status'] == 'failed' and 'Cannot decode' in metadata['error_traceback']
    assert not output.exists()
    assert not list(Path(settings['cache']).glob('smoke-*'))
    assert source.read_bytes() == b'not an image'


def test_refuse_source_overwrite_or_git_staging(tmp_path):
    source = tmp_path / 'source.png'
    Image.new('RGB', (2, 2)).save(source)
    for destination in (source, feasibility.PROJECT_ROOT / 'forbidden-stage.png'):
        with pytest.raises(ValueError, match='preserve the source and stay outside Git'):
            normalize_source(source, destination)


def test_worker_uses_normalized_path_and_matching_upstream_output_name(tmp_path, monkeypatch):
    from scripts import model_worker as worker
    source, staged, output = tmp_path / 'original.jpg', tmp_path / 'source_rgb.png', tmp_path / 'output.png'
    Image.new('L', (4, 3)).save(source)
    normalize_source(source, staged)
    monkeypatch.setattr(worker, 'prepare_batch_utility', lambda *_: None)
    monkeypatch.setattr(sys, 'argv', list(sys.argv))
    def upstream(*_a, **_k):
        args = sys.argv
        assert args[args.index('--image_path')+1] == str(staged)
        assert args[args.index('--age_init')+1] == '2'
        assert args[args.index('--gender')+1] == 'male'
        assert args[args.index('--target_ages')+1] == '30'
        (output.parent / 'source_rgb_30.png').write_text('synthetic output')
    monkeypatch.setattr(worker.runpy, 'run_path', upstream)
    request = dict(source_image=str(source), fading_source_image=str(staged), source_age=2,
                   target_age=30, output_path=str(output), config=dict(external_repository=str(tmp_path),
                                                                     gender='male', checkpoint='unchanged'))
    worker.fading(request)
    assert output.read_text() == 'synthetic output'
    assert request['source_image'] == str(source)


def test_sam_does_not_normalize_source(tmp_path, monkeypatch):
    monkeypatch.setattr(feasibility, 'normalize_source', lambda *_: pytest.fail('SAM normalized'))
    monkeypatch.setattr(feasibility, 'check_external', lambda *_: {'revision': 'test'})
    settings = config(tmp_path)
    def worker(args, **kwargs):
        request = json.loads(Path(args[-2]).read_text())
        assert request['source_image'] == 'unchanged-unread-source.jpg'
        assert 'fading_source_image' not in request and 'source_preprocessing' not in request
        from src.utils.metadata import ExperimentMetadata, write_metadata
        write_metadata(args[-1], ExperimentMetadata('test', 'test', 'sam', 'unused', 2, 30, 0,
                                                    {'status': 'failed'}, {}))
        return subprocess.CompletedProcess(args, 1)
    monkeypatch.setattr(feasibility.subprocess, 'run', worker)
    feasibility.subprocess_backend(dict(model='sam', config=settings, source_image='unchanged-unread-source.jpg'))
