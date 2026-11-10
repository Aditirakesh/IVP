"""Participant-level cross-validation splits for CDL-MDDNet.

Strictly enforces subject-disjoint splits so no participant's epochs ever appear
in both training and evaluation sets.

Contracts (README.md & tests/test_no_subject_leakage.py):
    kfold_splits(subject_ids, n_folds=10, seed=0) -> yields (train_idx, test_idx)
    loso_splits(subject_ids)                       -> yields (train_idx, test_idx)
"""

from __future__ import annotations

from typing import Iterator, Optional, Tuple

import numpy as np
from sklearn.model_selection import StratifiedKFold


def _validate_subject_ids(subject_ids: np.ndarray) -> np.ndarray:
    if subject_ids is None:
        raise ValueError("subject_ids must not be None.")
    sids = np.asarray(subject_ids)
    if sids.ndim != 1 or sids.size == 0:
        raise ValueError(
            f"subject_ids must be a non-empty 1D array, got shape {sids.shape}."
        )
    if np.issubdtype(sids.dtype, np.floating) and np.isnan(sids).any():
        raise ValueError("subject_ids must not contain NaN values.")
    return sids


def _subject_labels(
    unique_subjects: np.ndarray,
    subject_ids: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:
    y_arr = np.asarray(y)
    if y_arr.ndim != 1 or len(y_arr) != len(subject_ids):
        raise ValueError(
            f"y must be a 1D array of length {len(subject_ids)}, got shape {y_arr.shape}."
        )
    labels = np.empty(len(unique_subjects), dtype=y_arr.dtype)
    for i, sid in enumerate(unique_subjects):
        subj_y = np.unique(y_arr[subject_ids == sid])
        if len(subj_y) != 1:
            raise ValueError(
                f"Subject '{sid}' has inconsistent labels {subj_y.tolist()}; "
                f"participant-level stratification requires one label per subject."
            )
        labels[i] = subj_y[0]
    return labels


def kfold_splits(
    subject_ids: np.ndarray,
    n_folds: int = 10,
    seed: int = 0,
    y: Optional[np.ndarray] = None,
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """Yield subject-disjoint K-fold (train_idx, test_idx) epoch index arrays.

    When `y` is provided and every class has at least `n_folds` subjects, performs
    participant-level StratifiedKFold so each fold preserves class balance.
    When `y` is omitted (e.g. leakage contract checks), partitions shuffled unique
    participants evenly into `n_folds` folds.
    """
    sids = _validate_subject_ids(subject_ids)
    if isinstance(n_folds, bool) or not isinstance(n_folds, (int, np.integer)):
        raise ValueError(f"n_folds must be an integer, got {type(n_folds).__name__}.")
    n_folds = int(n_folds)

    unique_subjects = np.unique(sids)
    n_subjects = len(unique_subjects)

    if n_folds < 2 or n_folds > n_subjects:
        raise ValueError(
            f"n_folds must be in [2, {n_subjects}] (number of unique subjects), got {n_folds}."
        )

    use_stratified = False
    subj_y: Optional[np.ndarray] = None
    if y is not None:
        subj_y = _subject_labels(unique_subjects, sids, y)
        _, counts = np.unique(subj_y, return_counts=True)
        if counts.min() >= n_folds:
            use_stratified = True

    if use_stratified and subj_y is not None:
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        fold_subject_groups = [
            unique_subjects[test_subj_pos]
            for _, test_subj_pos in skf.split(unique_subjects, subj_y)
        ]
    else:
        rng = np.random.default_rng(seed)
        shuffled_subjects = rng.permutation(unique_subjects)
        fold_subject_groups = list(np.array_split(shuffled_subjects, n_folds))

    for test_subjects in fold_subject_groups:
        test_mask = np.isin(sids, test_subjects)
        train_idx = np.where(~test_mask)[0]
        test_idx = np.where(test_mask)[0]
        yield train_idx, test_idx


def loso_splits(subject_ids: np.ndarray) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """Yield Leave-One-Subject-Out (train_idx, test_idx) epoch index arrays.

    For MODMA (53 participants), yields 53 folds in deterministic sorted order
    of unique subject IDs.
    """
    sids = _validate_subject_ids(subject_ids)
    unique_subjects = np.unique(sids)
    if len(unique_subjects) < 2:
        raise ValueError(
            f"LOSO requires at least 2 unique subjects, got {len(unique_subjects)}."
        )

    for held_out_sid in unique_subjects:
        test_mask = sids == held_out_sid
        train_idx = np.where(~test_mask)[0]
        test_idx = np.where(test_mask)[0]
        yield train_idx, test_idx


def make_val_split(
    train_idx: np.ndarray,
    subject_ids: np.ndarray,
    y: Optional[np.ndarray] = None,
    val_fraction: float = 0.15,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Carve a subject-disjoint validation set strictly from `train_idx`.

    Never accesses outer test participants. Used for early stopping and checkpoint
    selection.
    """
    sids = _validate_subject_ids(subject_ids)
    if not (0.0 < float(val_fraction) < 1.0):
        raise ValueError(f"val_fraction must be in (0, 1), got {val_fraction}.")

    raw_tr = np.asarray(train_idx)
    if raw_tr.ndim != 1 or raw_tr.size == 0:
        raise ValueError("train_idx must be a non-empty 1D index array.")
    if not np.issubdtype(raw_tr.dtype, np.integer):
        raise ValueError(f"train_idx must have an integer dtype, got {raw_tr.dtype}.")
    if raw_tr.min() < 0 or raw_tr.max() >= len(sids):
        raise ValueError(
            f"train_idx contains out-of-bounds indices for length {len(sids)}."
        )
    if len(np.unique(raw_tr)) != len(raw_tr):
        raise ValueError("train_idx must not contain duplicate indices.")

    tr_idx = np.sort(raw_tr.astype(np.int64, copy=False))
    tr_sids = sids[tr_idx]
    train_subjects = np.unique(tr_sids)
    if len(train_subjects) < 2:
        raise ValueError(
            f"Need at least 2 training subjects to form a subject-disjoint validation split, "
            f"got {len(train_subjects)}."
        )

    n_val = max(1, int(round(len(train_subjects) * val_fraction)))
    n_val = min(n_val, len(train_subjects) - 1)

    rng = np.random.default_rng(seed)
    val_subjects: Optional[np.ndarray] = None

    if y is not None:
        y_arr = np.asarray(y)
        if y_arr.ndim != 1 or len(y_arr) != len(sids):
            raise ValueError(
                f"y must be a 1D array of length {len(sids)}, got shape {y_arr.shape}."
            )
        tr_y = y_arr[tr_idx]
        subj_y = _subject_labels(train_subjects, tr_sids, tr_y)
        classes, counts = np.unique(subj_y, return_counts=True)
        if len(classes) >= 2 and n_val >= len(classes) and counts.min() >= 2:
            # Allocate validation subjects proportionally per class while leaving >= 1 per class in sub_train
            alloc = {c: 1 for c in classes}
            remaining = n_val - len(classes)
            ideal = {c: (cnt / len(train_subjects)) * n_val for c, cnt in zip(classes, counts)}
            while remaining > 0:
                candidates = [c for c, cnt in zip(classes, counts) if alloc[c] < cnt - 1]
                if not candidates:
                    break
                best_c = max(candidates, key=lambda c: ideal[c] - alloc[c])
                alloc[best_c] += 1
                remaining -= 1

            chosen = []
            for c in classes:
                c_subjs = train_subjects[subj_y == c]
                chosen.append(rng.choice(c_subjs, size=alloc[c], replace=False))
            val_subjects = np.concatenate(chosen)

    if val_subjects is None:
        val_subjects = rng.choice(train_subjects, size=n_val, replace=False)

    val_mask = np.isin(tr_sids, val_subjects)
    val_idx = tr_idx[val_mask]
    sub_train_idx = tr_idx[~val_mask]

    if sub_train_idx.size == 0 or val_idx.size == 0:
        raise RuntimeError("Failed to create non-empty sub_train_idx and val_idx.")
    return sub_train_idx, val_idx
