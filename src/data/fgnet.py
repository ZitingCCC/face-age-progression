"""CPU-only FG-NET metadata, validation, output, and preparation orchestration.

Assumed (not real-data verified) filenames: DDD A DD [letter], without spaces,
where DDD is exactly three ASCII subject digits, A is a case-insensitive literal,
DD is exactly two age digits, and an optional single ASCII letter distinguishes
same-age images. Examples: 001A05.jpg, 001A05a.JPG. Adjust FILENAME_PATTERN after
checking the actual dataset; malformed supported images fail rather than skip.
Image paths are POSIX strings relative to raw_root, never copied or rewritten.
"""
import csv
import json
import os
from pathlib import Path, PurePosixPath
import re
import tempfile
from numbers import Integral

FILENAME_PATTERN = re.compile(r"(?P<subject>[0-9]{3})A(?P<age>[0-9]{2})(?:[a-z])?", re.IGNORECASE | re.ASCII)
EXTENSIONS = {".jpg", ".jpeg", ".png"}
METADATA_COLUMNS = ("subject_id", "age", "image_path")
SPLIT_COLUMNS = (*METADATA_COLUMNS, "split")
PAIR_COLUMNS = ("subject_id", "source_image", "source_age", "target_image",
                "target_age", "age_gap", "age_gap_group", "split")


def nonnegative_integer(value, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer; got {value!r}")
    return int(value)


def parse_filename(path: str | Path) -> tuple[str, int]:
    """Return the zero-padded subject ID and integer age; never infer metadata."""
    path = Path(path)
    match = FILENAME_PATTERN.fullmatch(path.stem)
    if path.suffix.lower() not in EXTENSIONS or match is None:
        raise ValueError(f"Malformed FG-NET filename {path.name!r}: expected "
                         "three subject digits + A + two age digits + optional single letter "
                         "and .jpg/.jpeg/.png (e.g. 001A05a.jpg). Verify real-data naming.")
    return match['subject'], int(match['age'])


def discover_images(raw_root: str | Path) -> list[Path]:
    root = Path(raw_root).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"FG-NET raw root is missing or not a directory: {root}. "
                         "Obtain FG-NET separately and manually extract it here.")
    images = sorted((p for p in root.rglob('*') if p.is_file() and
                     p.suffix.lower() in EXTENSIONS), key=lambda p: p.relative_to(root).as_posix())
    if not images:
        raise ValueError(f"No .jpg/.jpeg/.png images under {root}. Manually extract FG-NET here.")
    for path in images:
        if not path.resolve().is_relative_to(root):
            raise ValueError(f"Image symlink escapes raw root: {path}")
    return images


def validate_metadata(rows: list[dict], *, raw_root: str | Path | None = None,
                      check_paths: bool = False) -> None:
    """Strict metadata validation; path existence checks are opt-in.

    CSV consumers must preserve subject_id as text and convert age to integer.
    Empty tables are allowed (e.g. an empty split); discovery rejects empty data.
    """
    if check_paths and raw_root is None:
        raise ValueError("raw_root is required when check_paths=True")
    seen = set()
    paths = set()
    for index, row in enumerate(rows):
        missing = set(METADATA_COLUMNS) - row.keys()
        if missing:
            raise ValueError(f"Metadata row {index} missing required columns: {sorted(missing)}")
        subject = row['subject_id']
        if not isinstance(subject, str) or not subject.strip() or subject != subject.strip():
            raise ValueError(f"Metadata row {index}: subject_id must be nonempty text")
        age = nonnegative_integer(row['age'], 'age')
        path = row['image_path']
        if not isinstance(path, str) or not path or '\\' in path:
            raise ValueError(f"Metadata row {index}: image_path must be a relative POSIX string")
        portable = PurePosixPath(path)
        if portable.is_absolute() or '..' in portable.parts or ':' in path or portable.as_posix() != path or path == '.':
            raise ValueError(f"Unsafe/nonportable image_path: {path!r}; use a path relative to raw_root")
        key = subject, age, path
        if key in seen:
            raise ValueError(f"Duplicate metadata row: {key}")
        if path in paths:
            raise ValueError(f"Image path has multiple metadata records: {path}")
        seen.add(key)
        paths.add(path)
        if check_paths:
            root = Path(raw_root).expanduser().resolve()
            image = (root / path).resolve()
            if not image.is_relative_to(root) or not image.is_file():
                raise ValueError(f"Referenced image missing or outside raw_root: {path}")


def build_metadata(raw_root: str | Path) -> list[dict]:
    """Stable order: subject_id, age, image_path. Preserve subject leading zeros."""
    root = Path(raw_root).expanduser().resolve()
    rows = []
    for path in discover_images(root):
        subject, age = parse_filename(path)
        rows.append(dict(subject_id=subject, age=age, image_path=path.relative_to(root).as_posix()))
    rows.sort(key=lambda row: (row['subject_id'], row['age'], row['image_path']))
    validate_metadata(rows, raw_root=root, check_paths=True)
    return rows


def _check_destination(path: Path, overwrite: bool) -> None:
    if path.is_symlink() or (path.exists() and not path.is_file()):
        raise ValueError(f"Refusing non-regular output destination: {path}")
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output exists: {path}. Choose a new output root or set overwrite=True/--overwrite.")


def write_csv(path: str | Path, rows: list[dict], columns: tuple[str, ...], *,
              overwrite: bool = False) -> None:
    """Write one table atomically; default refuses existing files.

    With overwrite, replace only this explicitly named regular file. Atomic rename
    semantics on mounted Drive must be checked in real Colab. Use one writer per
    output root; simultaneous preparation jobs are not supported.
    """
    path = Path(path)
    _check_destination(path, overwrite)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', newline='',
                                         dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            writer = csv.DictWriter(handle, fieldnames=columns, lineterminator='\n')
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        _check_destination(path, overwrite)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def dataset_summary(metadata: list[dict], pairs: list[dict]) -> dict:
    from .pairs import validate_pairs, age_gap_group
    from .splits import validate_splits, SPLITS
    validate_splits(metadata)
    validate_pairs(pairs, metadata)
    return dict(image_count=len(metadata), subject_count=len({r['subject_id'] for r in metadata}),
                minimum_age=min((r['age'] for r in metadata), default=None),
                maximum_age=max((r['age'] for r in metadata), default=None),
                image_count_per_split={s: sum(r['split'] == s for r in metadata) for s in SPLITS},
                subject_count_per_split={s: len({r['subject_id'] for r in metadata if r['split'] == s}) for s in SPLITS},
                pair_count=len(pairs), pair_count_by_age_gap_group={
                    g: sum(age_gap_group(r['age_gap']) == g for r in pairs) for g in ('0-5', '6-10', '11+')})


def write_prepared_dataset(raw_root: str | Path, output_root: str | Path,
                           metadata: list[dict], pairs: list[dict], configuration: dict,
                           *, overwrite: bool = False) -> dict:
    """Validate before writing; precheck all outputs; never write under raw_root.

    Individual files are atomic, not a transaction across the whole bundle. An
    interruption can leave a partial bundle; rerun with explicit overwrite after
    inspection. No raw images or unrelated output names are removed.
    """
    from .pairs import validate_pairs
    from .splits import validate_splits
    raw = Path(raw_root).expanduser().resolve()
    output = Path(output_root).expanduser().resolve()
    if output.is_relative_to(raw) or raw.is_relative_to(output):
        raise ValueError("raw_root and output_root must be separate, non-overlapping directories")
    validate_metadata(metadata, raw_root=raw, check_paths=True)
    validate_splits(metadata)
    validate_pairs(pairs, metadata)
    summary = dataset_summary(metadata, pairs)
    record = json.dumps(dict(raw_root=raw.as_posix(), configuration=configuration,
                             summary=summary), indent=2, sort_keys=True, allow_nan=False) + '\n'
    for name in ('metadata.csv', 'pairs.csv', 'preparation.json'):
        _check_destination(output / name, overwrite)
    # Stage serialization first so malformed records cannot leave half-written tables.
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.fgnet-stage-', dir=output) as temporary:
        stage = Path(temporary)
        write_csv(stage / 'metadata.csv', metadata, SPLIT_COLUMNS)
        write_csv(stage / 'pairs.csv', pairs, PAIR_COLUMNS)
        (stage / 'preparation.json').write_text(record, encoding='utf-8')
        for name in ('metadata.csv', 'pairs.csv', 'preparation.json'):
            _check_destination(output / name, overwrite)
            os.replace(stage / name, output / name)
    return summary


def prepare_fgnet(raw_root: str | Path, output_root: str | Path, *,
                  split_options: dict | None = None, pair_options: dict | None = None,
                  overwrite: bool = False) -> dict:
    """Thin orchestration shared by CLI/tests; no downloads, decoding, or GPU use."""
    from .splits import split_subjects
    from .pairs import build_pairs
    split_options = dict(train=0.7, val=0.15, test=0.15, seed=42) | dict(split_options or {})
    pair_options = dict(min_source_age=None, max_source_age=None, min_target_age=None,
                        max_target_age=None, min_age_gap=1, max_age_gap=None,
                        max_pairs_per_subject=None) | dict(pair_options or {})
    metadata = split_subjects(build_metadata(raw_root), **split_options)
    pairs = build_pairs(metadata, **pair_options)
    configuration = dict(split=split_options, pairs=pair_options,
                         filename_pattern=FILENAME_PATTERN.pattern)
    summary = write_prepared_dataset(raw_root, output_root, metadata, pairs,
                                    configuration, overwrite=overwrite)
    return dict(metadata=metadata, pairs=pairs, summary=summary)
