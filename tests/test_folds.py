"""The official folds are used as published: disjoint, complete, unshuffled."""

import csv

import numpy as np

from audioscope.data import split_indices


def test_esc50_folds_match_published_column(esc50):
    with open(esc50.root / "meta" / "esc50.csv", newline="") as fh:
        published = {r["filename"]: int(r["fold"]) for r in csv.DictReader(fh)}
    assert len(esc50) == 2000
    assert esc50.num_classes == 50
    assert {f: int(k) for f, k in zip(esc50.filenames, esc50.folds)} == published


def test_esc50_folds_disjoint_and_cover_every_clip(esc50):
    folds = [set(esc50.fold_indices(k).tolist()) for k in range(1, 6)]
    for i in range(5):
        assert len(folds[i]) == 400
        for j in range(i + 1, 5):
            assert not folds[i] & folds[j]
    assert set().union(*folds) == set(range(2000))


def test_esc50_balanced_per_fold(esc50):
    for k in range(1, 6):
        counts = np.bincount(esc50.targets[esc50.fold_indices(k)], minlength=50)
        assert (counts == 8).all()


def test_esc50_source_recordings_stay_in_one_fold(esc50):
    # Within a class, every clip cut from one source recording is in one fold.
    seen = {}
    for g, t, f in zip(esc50.groups, esc50.targets, esc50.folds):
        assert seen.setdefault((g, t), f) == f
    # Across classes the published metadata has four source ids in two folds
    # (e.g. 131943: clock_tick in fold 2, clock_alarm in fold 3). Pin that fact
    # so a change in the metadata is noticed.
    by_src = {}
    for g, f in zip(esc50.groups, esc50.folds):
        by_src.setdefault(g, set()).add(int(f))
    split = sorted(g for g, fs in by_src.items() if len(fs) > 1)
    assert split == ["131943", "134049", "209698", "234879"]


def _check_split(cat, k, seed=0):
    tr, va, te = split_indices(cat, k, 0.1, seed=seed)
    assert not set(tr) & set(va) and not set(tr) & set(te) and not set(va) & set(te)
    assert set(tr) | set(va) | set(te) == set(range(len(cat)))
    assert (cat.folds[te] == k).all()
    assert (cat.folds[tr] != k).all() and (cat.folds[va] != k).all()
    assert not set(cat.groups[tr]) & set(cat.groups[va])
    return tr, va, te


def test_split_esc50_validation_from_training_folds_only(esc50):
    for k in range(1, 6):
        tr, va, _ = _check_split(esc50, k)
        assert 0.07 < len(va) / (len(tr) + len(va)) < 0.15
        assert set(esc50.targets[va]) == set(range(50))


def test_split_synthetic(synthetic):
    cat, _ = synthetic
    for k in range(1, 6):
        _check_split(cat, k, seed=k)


def test_split_is_deterministic(synthetic):
    cat, _ = synthetic
    a = split_indices(cat, 2, 0.2, seed=3)
    b = split_indices(cat, 2, 0.2, seed=3)
    for x, y in zip(a, b):
        assert np.array_equal(x, y)
