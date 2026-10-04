"""Isolated upstream bridge. Executed only by explicit real-model smoke commands."""
import json
import os
import random
from pathlib import Path
import runpy
import sys
import time
from importlib.metadata import version, PackageNotFoundError
from argparse import Namespace
from functools import partial

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from src.models.feasibility import SmokeMetadata, check_external
from src.utils.metadata import write_metadata


def sam(request, torch):
    config = request['config']
    from PIL import Image
    import dlib
    from torchvision import transforms
    # Pillow removed this alias; bridge only this worker's alignment call.
    if not hasattr(Image, 'ANTIALIAS'):
        Image.ANTIALIAS = Image.Resampling.LANCZOS
    from scripts.align_all_parallel import align_face
    from datasets.augmentations import AgeTransformer
    from models.psp import pSp
    from utils.common import tensor2im
    # Legacy SAM stores opts as well as tensors. Explicit provenance opt-in required.
    if not config.get('trusted_checkpoint'):
        raise ValueError('SAM pickle checkpoint is not marked trusted')
    original_load = torch.load
    torch.load = partial(original_load, weights_only=False)
    try:
        checkpoint = torch.load(config['checkpoint'], map_location='cpu')
        opts = dict(checkpoint['opts'])
        opts.update(checkpoint_path=config['checkpoint'], device=config['device'])
        if opts.get('output_size') != 1024:
            raise ValueError('Unexpected SAM checkpoint architecture; review native resolution')
        del checkpoint
        net = pSp(Namespace(**opts)).eval().to(config['device'])
    finally:
        torch.load = original_load
    predictor = dlib.shape_predictor(config['landmark_predictor'])
    aligned = align_face(filepath=request['source_image'], predictor=predictor)
    transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor(),
                                    transforms.Normalize([.5]*3, [.5]*3)])
    image = AgeTransformer(request['target_age'])(transform(aligned)).unsqueeze(0)
    # fp32 deliberately retained for custom StyleGAN CUDA kernels.
    with torch.no_grad():
        generated = net(image.to(config['device']).float(), randomize_noise=False, resize=True)[0]
    tensor2im(generated).save(request['output_path'])


def fading(request):
    config = request['config']
    # Upstream owns inversion/editing; never disable gradients for null inversion.
    # Pipeline is local-only via subprocess HF offline environment variables.
    sys.argv = ['age_editing.py', '--image_path', request['source_image'],
                '--age_init', str(request['source_age']), '--gender', config['gender'],
                '--specialized_path', config['checkpoint'], '--save_aged_dir', str(Path(request['output_path']).parent),
                '--target_ages', str(request['target_age'])]
    runpy.run_path(str(Path(config['external_repository']) / 'age_editing.py'), run_name='__main__')
    name = Path(request['source_image']).name.split('.')[-2]
    generated = Path(request['output_path']).parent / f'{name}_{request["target_age"]}.png'
    if not generated.is_file():
        raise RuntimeError('Expected upstream FADING output was not created')
    os.replace(generated, request['output_path'])


def main():
    request_path, result_path = map(Path, sys.argv[1:])
    request = json.loads(request_path.read_text())
    config = request['config']
    record = SmokeMetadata('task-04a-worker', config['method'], config['name'], request['source_image'],
                           request['source_age'], request['target_age'], config.get('seed', 0), config, {})
    # Worker result is normalized below for the parent adapter.
    result = {'status': 'failed', 'error_message': None, 'wall_clock_inference_seconds': None,
              'peak_cuda_memory_bytes': None, 'peak_cuda_reserved_bytes': None,
              'measurement_scope': 'load + preprocess/alignment + inversion/editing + save; excludes process startup',
              'output_resolution': None, 'versions': {}, 'device': {}}
    torch = None
    started = None
    cuda_measured = False
    try:
        external = check_external(request['model'], config)
        result['revision'] = external['revision']
        result['upstream'] = {k: config[k] for k in ('upstream_repository', 'revision', 'upstream_revision', 'implementation_repository') if k in config}
        sys.path.insert(0, str(Path(config['external_repository']).resolve()))
        for package in ('torch', 'torchvision', 'diffusers', 'transformers', 'accelerate', 'numpy', 'Pillow', 'dlib'):
            try:
                result['versions'][package] = version(package)
            except PackageNotFoundError:
                pass
        result['versions']['python'] = sys.version
        import torch
        result['versions']['cuda_build'] = torch.version.cuda
        cuda = config['device'] == 'cuda:0'
        if cuda and not torch.cuda.is_available():
            raise RuntimeError('Real smoke test requires CUDA; select Colab GPU runtime')
        if request['model'] == 'fading' and not cuda:
            raise ValueError('FADING upstream selects device automatically; bridge requires cuda:0')
        if config['precision'] != 'fp32' or not config.get('allow_upstream_internal_sizes'):
            raise ValueError('Review precision/internal-size configuration before generation')
        import numpy as np
        seed = config.get('seed', 0)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if cuda:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            cuda_measured = True
            props = torch.cuda.get_device_properties(0)
            result['device'] = dict(name=props.name, total_memory_bytes=props.total_memory, selected=config['device'])
            print('GPU diagnostics:', result['device'], flush=True)
        started = time.perf_counter()
        if request['model'] == 'sam':
            sam(request, torch)
        else:
            fading(request)
        if cuda:
            torch.cuda.synchronize()
        from PIL import Image
        with Image.open(request['output_path']) as output:
            output.verify()
        with Image.open(request['output_path']) as output:
            result['output_resolution'] = list(output.size)
        if result['output_resolution'] != [config['resolution']]*2:
            raise ValueError('Actual output resolution differs from configuration')
        result['status'] = 'completed'
    except Exception as error:
        result['error_message'] = f'{type(error).__name__}: {error}'
    finally:
        if started is not None:
            result['wall_clock_inference_seconds'] = time.perf_counter() - started
        if cuda_measured:
            try:
                result['peak_cuda_memory_bytes'] = torch.cuda.max_memory_allocated()
                result['peak_cuda_reserved_bytes'] = torch.cuda.max_memory_reserved()
            except RuntimeError:
                pass
        # Reuse atomic writer, then parent consumes only the explicitly measured fields.
        record.status = result['status']
        record.error_message = result['error_message']
        record.configuration = result
        write_metadata(result_path, record)
    return 0 if result['status'] == 'completed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
