"""Deterministic subject-disjoint splitting; longitudinal ages stay together."""
import math
import random
from .fgnet import validate_metadata, nonnegative_integer

SPLITS = ('train', 'val', 'test')


def validate_splits(rows: list[dict]) -> None:
    validate_metadata(rows)
    subjects = {s: set() for s in SPLITS}
    for row in rows:
        if row.get('split') not in SPLITS:
            raise ValueError(f"Missing/invalid split for {row['image_path']}; expected train/val/test")
        subjects[row['split']].add(row['subject_id'])
    for first, second in (('train', 'val'), ('train', 'test'), ('val', 'test')):
        overlap = subjects[first] & subjects[second]
        if overlap:
            raise ValueError(f"Subject leakage between {first} and {second}: {sorted(overlap)}")


def split_subjects(rows: list[dict], *, train: float = 0.7, val: float = 0.15,
                   test: float = 0.15, seed: int = 42) -> list[dict]:
    """Shuffle sorted unique subjects using local RNG; assign every image together.

    Largest-remainder allocation: floor each ratio*N, then distribute remaining
    subjects by descending fractional remainder, ties in train/val/test order.
    Small datasets can have empty splits; ratios describe subjects, not images.
    Returned rows are sorted by subject_id, age, image_path regardless of input.
    """
    validate_metadata(rows)
    seed = nonnegative_integer(seed, 'seed')
    if seed >= 2**32:
        raise ValueError('seed must be in [0, 2**32)')
    ratios = (train, val, test)
    if any(isinstance(r, bool) or not isinstance(r, (float, int)) or not math.isfinite(r) or r < 0 for r in ratios):
        raise ValueError('Split ratios must be finite non-negative numbers')
    if not math.isclose(sum(ratios), 1.0, rel_tol=0, abs_tol=1e-9):
        raise ValueError('train + val + test ratios must sum to 1')
    subjects = sorted({row['subject_id'] for row in rows})
    random.Random(seed).shuffle(subjects)
    exact = [len(subjects) * r for r in ratios]
    counts = [math.floor(n) for n in exact]
    priority = sorted(range(3), key=lambda i: (-(exact[i] - counts[i]), i))
    for i in priority[:len(subjects) - sum(counts)]:
        counts[i] += 1
    assignment = {}
    start = 0
    for name, count in zip(SPLITS, counts):
        assignment.update((subject, name) for subject in subjects[start:start + count])
        start += count
    result = [dict(row, split=assignment[row['subject_id']]) for row in rows]
    result.sort(key=lambda row: (row['subject_id'], row['age'], row['image_path']))
    validate_splits(result)
    return result
