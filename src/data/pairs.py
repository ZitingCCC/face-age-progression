"""Same-subject younger-to-older pairs within an already assigned split."""
from itertools import combinations
from .fgnet import nonnegative_integer
from .splits import validate_splits

REQUIRED = ('subject_id', 'source_image', 'source_age', 'target_image', 'target_age', 'age_gap')


def age_gap_group(age_gap: int) -> str:
    gap = nonnegative_integer(age_gap, 'age_gap')
    if gap == 0:
        raise ValueError('Longitudinal age_gap must be positive')
    return '0-5' if gap <= 5 else '6-10' if gap <= 10 else '11+'


def validate_pairs(pairs: list[dict], metadata: list[dict]) -> None:
    """Resolve both endpoints against metadata to verify identity, ages, and split."""
    validate_splits(metadata)
    lookup = {row['image_path']: row for row in metadata}
    seen = set()
    for index, pair in enumerate(pairs):
        missing = set(REQUIRED) - pair.keys()
        if missing:
            raise ValueError(f"Pair {index} missing required columns: {sorted(missing)}")
        source = lookup.get(pair['source_image'])
        target = lookup.get(pair['target_image'])
        if source is None or target is None:
            raise ValueError(f"Pair {index} references an image absent from metadata")
        if source['subject_id'] != target['subject_id'] or pair['subject_id'] != source['subject_id']:
            raise ValueError(f"Pair {index} has cross-subject endpoints or incorrect subject_id")
        for field in ('source_age', 'target_age', 'age_gap'):
            nonnegative_integer(pair[field], field)
        if pair['source_age'] != source['age'] or pair['target_age'] != target['age']:
            raise ValueError(f"Pair {index} ages disagree with metadata")
        gap = pair['target_age'] - pair['source_age']
        if gap <= 0 or pair['age_gap'] != gap:
            raise ValueError(f"Pair {index} requires target_age > source_age and correct age_gap")
        if source['split'] != target['split'] or pair.get('split', source['split']) != source['split']:
            raise ValueError(f"Pair {index} crosses splits or has an incorrect split")
        if pair.get('age_gap_group', age_gap_group(gap)) != age_gap_group(gap):
            raise ValueError(f"Pair {index} has incorrect age_gap_group")
        key = pair['source_image'], pair['target_image']
        if key in seen:
            raise ValueError(f"Duplicate longitudinal pair: {key}")
        seen.add(key)


def build_pairs(metadata: list[dict], *, min_source_age: int | None = None,
                max_source_age: int | None = None, min_target_age: int | None = None,
                max_target_age: int | None = None, min_age_gap: int = 1,
                max_age_gap: int | None = None, max_pairs_per_subject: int | None = None) -> list[dict]:
    """All valid image combinations by default; equal-age pairs are excluded.

    Optional cap keeps the first K valid pairs in (source_age, source_image,
    target_age, target_image) order per subject. No random subsampling or global
    RNG is used. Iteration stores at most K pairs per subject when capped.
    """
    validate_splits(metadata)
    bounds = dict(min_source_age=min_source_age, max_source_age=max_source_age,
                  min_target_age=min_target_age, max_target_age=max_target_age,
                  min_age_gap=min_age_gap, max_age_gap=max_age_gap,
                  max_pairs_per_subject=max_pairs_per_subject)
    for name, value in bounds.items():
        if value is not None:
            nonnegative_integer(value, name)
    if min_age_gap is None or min_age_gap < 1:
        raise ValueError('min_age_gap must be at least 1')
    for name in ('source_age', 'target_age', 'age_gap'):
        low, high = bounds['min_' + name], bounds['max_' + name]
        if low is not None and high is not None and low > high:
            raise ValueError(f'min_{name} cannot exceed max_{name}')
    if max_pairs_per_subject is not None and max_pairs_per_subject < 1:
        raise ValueError('max_pairs_per_subject must be positive or null')
    grouped = {}
    for row in sorted(metadata, key=lambda r: (r['subject_id'], r['age'], r['image_path'])):
        grouped.setdefault(row['subject_id'], []).append(row)
    result = []
    for subject, rows in grouped.items():
        count = 0
        for source, target in combinations(rows, 2):
            gap = target['age'] - source['age']
            if gap <= 0:
                continue
            values = dict(source_age=source['age'], target_age=target['age'], age_gap=gap)
            if any((bounds['min_' + name] is not None and value < bounds['min_' + name]) or
                   (bounds['max_' + name] is not None and value > bounds['max_' + name])
                   for name, value in values.items()):
                continue
            result.append(dict(subject_id=subject, source_image=source['image_path'],
                               source_age=source['age'], target_image=target['image_path'],
                               target_age=target['age'], age_gap=gap,
                               age_gap_group=age_gap_group(gap), split=source['split']))
            count += 1
            if max_pairs_per_subject is not None and count >= max_pairs_per_subject:
                break
    validate_pairs(result, metadata)
    return result
