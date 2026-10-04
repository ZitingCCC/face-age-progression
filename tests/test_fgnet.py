"""Synthetic metadata tests only: no network, GPU, or real dataset needed."""
import csv
import json
from pathlib import Path
import subprocess
import sys
import pytest

from src.data.fgnet import (parse_filename, discover_images, build_metadata,
    validate_metadata, write_csv, METADATA_COLUMNS, prepare_fgnet, dataset_summary,
    write_prepared_dataset)
from src.data.splits import split_subjects, validate_splits
from src.data.pairs import build_pairs, validate_pairs, age_gap_group

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def raw(tmp_path):
    root = tmp_path / 'raw'
    for subject in range(1, 11):
        for age in (5, 10, 18, 30):
            path = root / 'nested' / f'{subject:03d}A{age:02d}.JPG'
            path.parent.mkdir(parents=True, exist_ok=True)
            # Contents need not decode: Task 03 validates metadata, not pixels.
            path.write_bytes(b'synthetic test placeholder')
    return root


@pytest.mark.parametrize('filename,subject,age', [
    ('001A05.jpg', '001', 5), ('002a00.PNG', '002', 0),
    ('010A99a.JpEg', '010', 99), ('001A05B.png', '001', 5)])
def test_parser_valid(filename, subject, age):
    assert parse_filename(filename) == (subject, age)


@pytest.mark.parametrize('name', ['1A05.jpg', '001A5.jpg', '001A-5.jpg',
    '001A5.5.jpg', '001A005.jpg', '001_05.jpg', '001A05ab.jpg',
    '001A05_1.jpg', '001A05.gif', 'foo.jpg', '001A.jpg', '００１A05.jpg', '001A05İ.jpg'])
def test_parser_rejects(name):
    with pytest.raises(ValueError, match='expected'):
        parse_filename(name)


def test_recursive_discovery_and_order(raw):
    (raw / '001A00.png').write_bytes(b'x')
    (raw / 'nested' / '001A01.JPEG').write_bytes(b'x')
    (raw / 'ignored.txt').write_text('not an image')
    images = discover_images(raw)
    assert len(images) == 42
    assert images == sorted(images, key=lambda p: p.relative_to(raw).as_posix())
    metadata = build_metadata(raw)
    assert metadata == sorted(metadata, key=lambda r: (r['subject_id'], r['age'], r['image_path']))
    assert metadata[0] == dict(subject_id='001', age=0, image_path='001A00.png')


def test_missing_empty_and_malformed_raw(tmp_path):
    with pytest.raises(ValueError, match='Manually|manually'):
        discover_images(tmp_path / 'absent')
    with pytest.raises(ValueError, match='No .jpg'):
        discover_images(tmp_path)
    (tmp_path / 'bad.jpg').touch()
    with pytest.raises(ValueError, match='Malformed'):
        build_metadata(tmp_path)


def test_metadata_write_and_no_clobber(raw, tmp_path):
    rows = build_metadata(raw)
    path = tmp_path / 'out' / 'metadata.csv'
    write_csv(path, rows, METADATA_COLUMNS)
    first = path.read_bytes()
    with path.open() as handle:
        loaded = list(csv.DictReader(handle))
    assert loaded[0]['subject_id'] == '001'
    with pytest.raises(FileExistsError, match='overwrite'):
        write_csv(path, [], METADATA_COLUMNS)
    assert path.read_bytes() == first
    write_csv(path, rows, METADATA_COLUMNS, overwrite=True)
    assert path.read_bytes() == first
    assert len(list(path.parent.iterdir())) == 1


def test_subject_splits_and_six_longitudinal_pairs(raw):
    metadata = build_metadata(raw)
    split = split_subjects(metadata)
    assert split == split_subjects(list(reversed(metadata)))
    assert split != split_subjects(metadata, seed=43)
    validate_splits(split)
    assignment = {s: {r['subject_id'] for r in split if r['split'] == s} for s in ('train', 'val', 'test')}
    assert [len(assignment[s]) for s in assignment] == [7, 2, 1]
    for subject in {r['subject_id'] for r in split}:
        assert len({r['split'] for r in split if r['subject_id'] == subject}) == 1
    pairs = build_pairs(split)
    assert len(pairs) == 60
    for subject in assignment['train'] | assignment['val'] | assignment['test']:
        subject_pairs = [r for r in pairs if r['subject_id'] == subject]
        assert {(r['source_age'], r['target_age']) for r in subject_pairs} == {
            (5, 10), (5, 18), (5, 30), (10, 18), (10, 30), (18, 30)}
        assert len({r['split'] for r in subject_pairs}) == 1
    assert all(r['target_age'] > r['source_age'] and
               r['age_gap'] == r['target_age'] - r['source_age'] for r in pairs)
    validate_pairs(pairs, split)


@pytest.mark.parametrize('first,second', [('train', 'val'), ('train', 'test'), ('val', 'test')])
def test_leakage_detected(raw, first, second):
    rows = build_metadata(raw)
    rows = [dict(rows[0], split=first), dict(rows[1], split=second)]
    with pytest.raises(ValueError, match=f'{first} and {second}'):
        validate_splits(rows)


@pytest.mark.parametrize('options', [dict(train=-0.1), dict(val=0.2), dict(train=float('nan')),
    dict(seed=True), dict(seed=-1), dict(seed=2**32)])
def test_split_invalid_options(raw, options):
    with pytest.raises(ValueError):
        split_subjects(build_metadata(raw), **options)


def test_filters_limiting_and_same_age(raw):
    (raw / 'nested' / '001A05a.png').write_bytes(b'x')
    rows = split_subjects(build_metadata(raw))
    all_pairs = build_pairs(rows)
    assert not any(p['source_age'] == p['target_age'] for p in all_pairs)
    filtered = build_pairs(rows, min_source_age=5, max_source_age=10,
                           min_target_age=18, max_target_age=30, min_age_gap=8, max_age_gap=20)
    assert filtered == [p for p in all_pairs if 5 <= p['source_age'] <= 10 and
                        18 <= p['target_age'] <= 30 and 8 <= p['age_gap'] <= 20]
    limited = build_pairs(rows, max_pairs_per_subject=2)
    assert limited == build_pairs(list(reversed(rows)), max_pairs_per_subject=2)
    assert len(limited) == 20
    assert limited[:2] == all_pairs[:2]


@pytest.mark.parametrize('options', [dict(min_age_gap=0), dict(min_age_gap=None),
    dict(max_age_gap=0), dict(min_source_age=-1), dict(max_pairs_per_subject=0),
    dict(min_target_age=20, max_target_age=10), dict(max_age_gap=1.5)])
def test_invalid_pair_filters(raw, options):
    with pytest.raises(ValueError):
        build_pairs(split_subjects(build_metadata(raw)), **options)


@pytest.mark.parametrize('gap,group', [(1, '0-5'), (5, '0-5'), (6, '6-10'), (10, '6-10'), (11, '11+')])
def test_age_gap_boundaries(gap, group):
    assert age_gap_group(gap) == group


@pytest.mark.parametrize('gap', [0, -1, 1.5, True])
def test_invalid_age_gap(gap):
    with pytest.raises(ValueError):
        age_gap_group(gap)


@pytest.mark.parametrize('change', [dict(subject_id=''), dict(age=-1), dict(age=5.5),
    dict(age=True), dict(image_path='../escape.jpg'), dict(image_path='/absolute.jpg'),
    dict(image_path='a\\b.jpg'), dict(image_path='a/./b.jpg')])
def test_metadata_integrity(raw, change):
    row = build_metadata(raw)[0] | change
    with pytest.raises(ValueError):
        validate_metadata([row])


def test_missing_columns_duplicate_and_paths(raw):
    rows = build_metadata(raw)
    with pytest.raises(ValueError, match='missing required'):
        validate_metadata([dict(subject_id='001')])
    with pytest.raises(ValueError, match='Duplicate'):
        validate_metadata([rows[0], rows[0]])
    with pytest.raises(ValueError, match='multiple metadata'):
        validate_metadata([rows[0], rows[0] | dict(age=6)])
    missing = rows[0] | dict(image_path='missing.jpg')
    validate_metadata([missing])
    with pytest.raises(ValueError, match='raw_root is required'):
        validate_metadata([missing], check_paths=True)
    with pytest.raises(ValueError, match='missing'):
        validate_metadata([missing], raw_root=raw, check_paths=True)


@pytest.mark.parametrize('change', [dict(subject_id='999'), dict(source_age=0),
    dict(target_age=5), dict(age_gap=100), dict(split='invalid'),
    dict(source_image='absent.jpg'), dict(target_image='nested/002A30.JPG'),
    dict(age_gap_group='11+')])
def test_pair_integrity(raw, change):
    metadata = split_subjects(build_metadata(raw))
    pair = build_pairs(metadata)[0] | change
    with pytest.raises(ValueError):
        validate_pairs([pair], metadata)


def test_duplicate_and_missing_pair_columns(raw):
    metadata = split_subjects(build_metadata(raw))
    pair = build_pairs(metadata)[0]
    with pytest.raises(ValueError, match='Duplicate'):
        validate_pairs([pair, pair], metadata)
    with pytest.raises(ValueError, match='missing required'):
        validate_pairs([{}], metadata)


def test_pipeline_summary_rerun_and_raw_preservation(raw, tmp_path):
    before = {p: p.read_bytes() for p in raw.rglob('*') if p.is_file()}
    output = tmp_path / 'processed'
    result = prepare_fgnet(raw, output)
    summary = result['summary']
    assert summary == dataset_summary(result['metadata'], result['pairs'])
    assert summary == dict(image_count=40, subject_count=10, minimum_age=5, maximum_age=30,
        image_count_per_split=dict(train=28, val=8, test=4),
        subject_count_per_split=dict(train=7, val=2, test=1), pair_count=60,
        pair_count_by_age_gap_group={'0-5': 10, '6-10': 10, '11+': 40})
    payloads = {p.name: p.read_bytes() for p in output.iterdir()}
    assert set(payloads) == {'metadata.csv', 'pairs.csv', 'preparation.json'}
    assert json.loads(payloads['preparation.json'])['configuration']['split']['seed'] == 42
    with pytest.raises(FileExistsError):
        prepare_fgnet(raw, output)
    unrelated = output / 'keep.txt'
    unrelated.write_text('preserve')
    prepare_fgnet(raw, output, overwrite=True)
    assert all((output / name).read_bytes() == payload for name, payload in payloads.items())
    assert unrelated.read_text() == 'preserve'
    assert all(p.read_bytes() == payload for p, payload in before.items())
    with pytest.raises(ValueError, match='non-overlapping'):
        prepare_fgnet(raw, raw / 'processed', overwrite=True)


def test_output_precheck_no_partial_write(raw, tmp_path):
    output = tmp_path / 'processed'
    output.mkdir()
    (output / 'pairs.csv').write_text('existing unrelated table')
    with pytest.raises(FileExistsError):
        prepare_fgnet(raw, output)
    assert not (output / 'metadata.csv').exists()
    assert (output / 'pairs.csv').read_text() == 'existing unrelated table'
    (output / 'pairs.csv').unlink()
    (output / 'pairs.csv').mkdir()
    with pytest.raises(ValueError, match='non-regular'):
        prepare_fgnet(raw, output, overwrite=True)
    assert not (output / 'metadata.csv').exists()


def test_cli_help_and_cpu_pipeline(raw, tmp_path):
    script = ROOT / 'scripts/prepare_fgnet.py'
    help_result = subprocess.run([sys.executable, str(script), '--help'], capture_output=True, text=True, cwd=tmp_path)
    assert help_result.returncode == 0
    assert '--raw-root' in help_result.stdout and '--overwrite' in help_result.stdout
    output = tmp_path / 'cli-output'
    command = [sys.executable, str(script), '--raw-root', str(raw), '--output-root', str(output),
               '--seed', '7', '--max-pairs-per-subject', '1']
    result = subprocess.run(command, capture_output=True, text=True, cwd=tmp_path)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)['pair_count'] == 10
    rerun = subprocess.run(command, capture_output=True, text=True, cwd=tmp_path)
    assert rerun.returncode != 0 and 'overwrite' in rerun.stderr


def test_empty_splits_and_no_pairs(raw):
    metadata = split_subjects(build_metadata(raw)[:1], train=1, val=0, test=0)
    assert build_pairs(metadata) == []
    assert dataset_summary(metadata, [])['pair_count'] == 0


def test_symlink_escape_and_output_symlink(raw, tmp_path):
    outside = tmp_path / 'outside.jpg'
    outside.write_bytes(b'x')
    image_link = raw / '001A00.jpg'
    image_link.symlink_to(outside)
    with pytest.raises(ValueError, match='escapes'):
        discover_images(raw)
    image_link.unlink()
    output = tmp_path / 'processed'
    output.mkdir()
    (output / 'metadata.csv').symlink_to(outside)
    with pytest.raises(ValueError, match='non-regular'):
        prepare_fgnet(raw, output, overwrite=True)
    assert outside.read_bytes() == b'x'


@pytest.mark.parametrize('present', [False, True])
def test_notebook_synthetic_execution_and_absent_data(tmp_path, capsys, present):
    """Execute thin cells with local path overrides; does not emulate real Drive."""
    import ast
    notebook = json.loads((ROOT / 'notebooks/02_dataset.ipynb').read_text())
    assert notebook['nbformat'] == 4
    drive = tmp_path / 'synthetic-storage'
    raw = drive / 'data/fgnet/raw'
    if present:
        raw.mkdir(parents=True)
        for age in (5, 10, 18, 30):
            (raw / f'001A{age:02d}.jpg').write_bytes(b'synthetic notebook fixture')
    namespace = {}
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            ast.parse(source)
            source = source.replace('Path("/content/face-age-progression")', f'Path({str(ROOT)!r})')
            source = source.replace('Path("/content/drive/MyDrive/AI6132")', f'Path({str(drive)!r})')
            exec(compile(source, '02_dataset.ipynb', 'exec'), namespace)
    output = drive / 'data/fgnet/processed'
    if present:
        assert namespace['summary']['image_count'] == 4
        assert namespace['summary']['pair_count'] == 6
        assert (output / 'metadata.csv').is_file()
        assert (output / 'pairs.csv').is_file()
    else:
        assert not output.exists()
        printed = capsys.readouterr().out
        assert 'Place/extract FG-NET manually' in printed
        assert 'No outputs written' in printed
