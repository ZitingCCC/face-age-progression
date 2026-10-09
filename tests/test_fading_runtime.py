"""CPU-only batch import regressions; synthetic upstream code, no real models."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from src.models.fading_runtime import batch_utility_source
from src.models.feasibility import check_external, worker_environment


# Exact unused import from the pinned ptp_utils.py; no upstream code is vendored.
SOURCE = 'from IPython.display import display\n\ndef batch_value():\n    return 42\n'


@pytest.fixture
def repository(tmp_path):
    repo = tmp_path / 'external'
    utility = repo / 'FADING_util/ptp_utils.py'
    utility.parent.mkdir(parents=True)
    utility.write_text(SOURCE)
    return repo


def test_unused_display_import_removed_only_in_memory(repository):
    path, tree = batch_utility_source(repository)
    namespace = {}
    exec(compile(tree, str(path), 'exec'), namespace)
    assert namespace['batch_value']() == 42
    assert 'display' not in namespace
    assert path.read_text() == SOURCE


@pytest.mark.parametrize('source', [
    SOURCE + '\ndisplay(batch_value())\n',
    SOURCE.replace('import display', 'import display as show'),
    SOURCE + 'from IPython.display import display\n',
    'def batch_value():\n    return 42\n',
])
def test_changed_import_or_display_usage_rejected(repository, source):
    (repository / 'FADING_util/ptp_utils.py').write_text(source)
    with pytest.raises(ValueError, match='requires exactly one unused'):
        batch_utility_source(repository)


def test_utility_symlink_cannot_escape_repository(repository, tmp_path):
    utility = repository / 'FADING_util/ptp_utils.py'
    outside = tmp_path / 'outside.py'
    outside.write_text(SOURCE)
    utility.unlink()
    utility.symlink_to(outside)
    with pytest.raises(ValueError, match='inside the external checkout'):
        batch_utility_source(repository)


def test_worker_exact_import_chain_without_ipython(repository, tmp_path, monkeypatch):
    (repository / 'p2p.py').write_text('import FADING_util.ptp_utils as ptp_utils\n')
    (repository / 'age_editing.py').write_text(
        "import argparse\nfrom p2p import *\nfrom pathlib import Path\n"
        "assert ptp_utils.batch_value() == 42\n"
        "parser = argparse.ArgumentParser()\n"
        "parser.add_argument('--save_aged_dir')\n"
        "args, _ = parser.parse_known_args()\n"
        "Path(args.save_aged_dir, 'synthetic_30.png').write_text('test-only, not a face')\n")
    before = {p.relative_to(repository): p.read_bytes() for p in repository.rglob('*.py')}
    request = dict(source_image='synthetic.jpg', fading_source_image='synthetic.jpg', source_age=2, target_age=30,
                   output_path=str(tmp_path / 'output.png'),
                   config=dict(external_repository=str(repository), gender='male', checkpoint='unused'))
    code = '''
import importlib.abc, json, os, sys, urllib.request
class NoModelsOrIPython(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('IPython', 'torch', 'diffusers'):
            raise ModuleNotFoundError("No module named '" + fullname + "'")
sys.meta_path.insert(0, NoModelsOrIPython())
urllib.request.urlopen = lambda *a, **k: (_ for _ in ()).throw(AssertionError('download'))
from scripts.model_worker import fading
request = json.loads(sys.argv[1])
sys.path.insert(0, request['config']['external_repository'])
parent = dict(os.environ)
fading(request)
assert dict(os.environ) == parent
assert all(n not in sys.modules for n in ('IPython', 'torch', 'diffusers'))
'''
    monkeypatch.setenv('MPLBACKEND', 'module://matplotlib_inline.backend_inline')
    parent = dict(os.environ)
    env = worker_environment('fading', dict(python=sys.executable, cache=str(tmp_path / 'cache')))
    subprocess.run([sys.executable, '-c', code, json.dumps(request)], env=env, check=True)
    assert Path(request['output_path']).read_text() == 'test-only, not a face'
    assert dict(os.environ) == parent
    assert before == {p.relative_to(repository): p.read_bytes() for p in repository.rglob('*.py')}
    assert not list(repository.rglob('__pycache__'))


def test_preflight_checks_batch_source_without_imports(repository, tmp_path, monkeypatch):
    for name in ('age_editing.py', 'p2p.py', 'null_inversion.py'):
        (repository / name).write_text("raise AssertionError('upstream loaded during preflight')")
    pipeline = tmp_path / 'pipeline'
    for relative in ('model_index.json', 'unet/config.json', 'vae/config.json',
                     'scheduler/scheduler_config.json', 'text_encoder/config.json',
                     'tokenizer/tokenizer_config.json', 'unet/fake.bin', 'vae/fake.bin',
                     'text_encoder/fake.bin'):
        file = pipeline / relative
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text('synthetic fixture, not weights')
    def git(args, **kwargs):
        assert args[0] == 'git' and not kwargs.get('shell', False)
        return subprocess.CompletedProcess(args, 0, stdout='test-revision' if 'rev-parse' in args else '')
    monkeypatch.setattr('src.models.feasibility.subprocess.run', git)
    config = dict(external_repository=str(repository), checkpoint=str(pipeline),
                  python=sys.executable, revision='test-revision')
    report = check_external('fading', config)
    assert report['batch_compatibility'] == 'unused IPython.display import omitted in memory'
    (repository / 'FADING_util/ptp_utils.py').write_text(SOURCE + 'display(42)\n')
    with pytest.raises(ValueError, match='unused'):
        check_external('fading', config)
    (repository / 'FADING_util/ptp_utils.py').unlink()
    with pytest.raises(FileNotFoundError):
        check_external('fading', config)


def test_compatibility_module_import_is_lazy():
    code = '''
import sys, urllib.request
urllib.request.urlopen = lambda *a, **k: (_ for _ in ()).throw(AssertionError('download'))
import src.models.fading_runtime
assert all(n not in sys.modules for n in ('IPython', 'torch', 'diffusers', 'FADING_util.ptp_utils'))
'''
    subprocess.run([sys.executable, '-c', code], check=True)


def test_failed_utility_import_cleans_module_registration(repository):
    (repository / 'FADING_util/ptp_utils.py').write_text(SOURCE + "raise RuntimeError('test failure')\n")
    code = '''
import sys
from src.models.fading_runtime import MODULE, prepare_batch_utility
sys.path.insert(0, sys.argv[1])
try:
    prepare_batch_utility(sys.argv[1])
except RuntimeError as error:
    assert str(error) == 'test failure'
else:
    raise AssertionError('Expected failure')
assert MODULE not in sys.modules
assert not hasattr(sys.modules['FADING_util'], 'ptp_utils')
sys.modules[MODULE] = object()
try:
    prepare_batch_utility(sys.argv[1])
except RuntimeError as error:
    assert 'before upstream imports' in str(error)
else:
    raise AssertionError('Must not overwrite an already imported utility')
'''
    subprocess.run([sys.executable, '-c', code, str(repository)], check=True)
