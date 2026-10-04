"""Prepare FG-NET metadata only; raw images must already exist locally."""
import argparse
import json
from pathlib import Path
import sys
from yaml import YAMLError

# Permit invocation from any working directory without requiring packaging changes.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from src.utils.config import load_config
from src.data.fgnet import prepare_fgnet


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-root', required=True, type=Path, help='Manually extracted FG-NET root (recursive search)')
    parser.add_argument('--output-root', required=True, type=Path, help='Separate processed directory; no images copied')
    parser.add_argument('--config', type=Path, default=PROJECT_ROOT / 'configs/dataset.yaml')
    parser.add_argument('--seed', type=int)
    for name in ('train', 'val', 'test'):
        parser.add_argument(f'--{name}-ratio', type=float)
    for name in ('min-source-age', 'max-source-age', 'min-target-age', 'max-target-age',
                 'min-age-gap', 'max-age-gap', 'max-pairs-per-subject'):
        parser.add_argument('--' + name, type=int)
    parser.add_argument('--overwrite', action='store_true', help='Explicitly replace metadata.csv, pairs.csv, preparation.json')
    args = parser.parse_args()
    try:
        config = load_config(args.config)
        dataset = config.get('dataset')
        if not isinstance(dataset, dict) or dataset.get('name') != 'fgnet':
            raise ValueError('dataset.name must be fgnet')
        split, pairs = dict(config['split']), dict(config['pairs'])
        if args.seed is not None:
            split['seed'] = args.seed
        for name in ('train', 'val', 'test'):
            value = getattr(args, name + '_ratio')
            if value is not None:
                split[name] = value
        for name in ('min_source_age', 'max_source_age', 'min_target_age', 'max_target_age',
                     'min_age_gap', 'max_age_gap', 'max_pairs_per_subject'):
            value = getattr(args, name)
            if value is not None:
                pairs[name] = value
        result = prepare_fgnet(args.raw_root, args.output_root, split_options=split,
                               pair_options=pairs, overwrite=args.overwrite)
    except (ValueError, OSError, KeyError, TypeError, YAMLError) as error:
        parser.error(str(error))
    print(json.dumps(result['summary'], indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
