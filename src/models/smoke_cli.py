"""Shared one-image CLI; never sets up dependencies or downloads assets."""
import argparse
import json
from pathlib import Path
from src.utils.config import load_config
from .feasibility import check_external
from .sam_adapter import SAMAdapter
from .fading_adapter import FADINGAdapter


def main(model):
    parser = argparse.ArgumentParser(description=f'One-image {model} feasibility only; run canonical setup first')
    parser.add_argument('--config', type=Path, default=Path(__file__).resolve().parents[2] / 'configs/models.yaml')
    parser.add_argument('--check', action='store_true', help='Local assets/revision check and SAM Ninja runtime probe; loads no model')
    parser.add_argument('--source', type=Path)
    parser.add_argument('--source-age', type=int)
    parser.add_argument('--target-age', type=int)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--gender', choices=['female', 'male'], help='Researcher-supplied FADING prompt only')
    parser.add_argument('--trust-sam-checkpoint', action='store_true', help='Allow legacy pickle only after verifying provenance')
    args = parser.parse_args()
    settings = load_config(args.config)
    config = dict(settings[model], seed=settings['smoke']['seed'])
    if args.gender:
        config['gender'] = args.gender
    if args.trust_sam_checkpoint:
        config['trusted_checkpoint'] = True
    try:
        if args.check:
            print(json.dumps(check_external(model, config), indent=2))
            return
        if args.source is None or args.output is None:
            parser.error('--source and --output are required for generation')
        target_age = settings['smoke']['target_age'] if args.target_age is None else args.target_age
        adapter = SAMAdapter(config) if model == 'sam' else FADINGAdapter(config)
        result = adapter._generate(args.source, args.source_age, target_age, args.output)
        print(json.dumps(result, indent=2))
    except Exception as error:
        raise SystemExit(f'{type(error).__name__}: {error}') from error
