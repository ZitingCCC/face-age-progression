"""Guarded batch compatibility for the pinned upstream utility; no model imports."""
import ast
import importlib
import importlib.util
from pathlib import Path
import sys


MODULE = 'FADING_util.ptp_utils'


def batch_utility_source(repository):
    """Validate and omit only an unused notebook display import, in memory."""
    root = Path(repository).resolve()
    path = (root / 'FADING_util/ptp_utils.py').resolve()
    if not path.is_relative_to(root):
        raise ValueError('FADING batch utility must remain inside the external checkout')
    tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    imports = [node for node in tree.body if isinstance(node, ast.ImportFrom)
               and node.module == 'IPython.display' and node.level == 0
               and len(node.names) == 1 and node.names[0].name == 'display'
               and node.names[0].asname is None]
    if len(imports) != 1 or any(isinstance(node, ast.Name) and node.id == 'display'
                                for node in ast.walk(tree)):
        raise ValueError('FADING batch compatibility requires exactly one unused '
                         'from IPython.display import display; review the pinned upstream source')
    tree.body.remove(imports[0])
    return path, tree


def prepare_batch_utility(repository):
    """Load real upstream code minus its unused display import in this worker."""
    path, tree = batch_utility_source(repository)
    if MODULE in sys.modules:
        raise RuntimeError('FADING batch utility must be prepared before upstream imports')
    package = importlib.import_module('FADING_util')
    if path.parent not in [Path(item).resolve() for item in package.__path__]:
        raise ValueError('FADING_util resolves outside the configured external checkout')
    spec = importlib.util.spec_from_file_location(MODULE, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[MODULE] = module
    try:
        exec(compile(tree, str(path), 'exec'), module.__dict__)
    except BaseException:
        del sys.modules[MODULE]
        raise
    package.ptp_utils = module
