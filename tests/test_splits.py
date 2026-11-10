"""Comprehensive tests for participant-level data splitting (Member 3)."""

import numpy as np
import pytest

from src.training.splits import kfold_splits, loso_splits, make_val_split


def _modma_like_arrays(seed: int = 0):
    rng = np.random.default_rng(seed)
    # 53 MODMA participants: 24 MDD (1) and 29 HC (0) with variable epoch counts
    subj_labels = np.array([1] * 24 + [0] * 29, dtype=np.int64)
    epochs_per_subj = rng.integers(60, 85, size=53)
    subject_id = np.repeat(np.arange(53, dtype=np.int64), epochs_per_subj)
    y = np.repeat(subj_labels, epochs_per_subj)
    # Interleave epochs to confirm splits do not assume contiguous subject blocks
    perm = rng.permutation(len(subject_id))
    return subject_id[perm], y[perm]


class TestKFoldSplits:
    def test_53_subjects_coverage_and_no_leakage_without_y(self):
        subject_id, _ = _modma_like_arrays(seed=1)
        folds = list(kfold_splits(subject_id, n_folds=10, seed=0))
        assert len(folds) == 10

        seen_epochs = np.zeros(len(subject_id), dtype=int)
        seen_subjects = []
        for tr, te in folds:
            tr_subs = set(subject_id[tr].tolist())
            te_subs = set(subject_id[te].tolist())
            assert tr_subs.isdisjoint(te_subs)
            assert tr_subs | te_subs == set(range(53))
            seen_subjects.extend(te_subs)
            seen_epochs[te] += 1

        assert sorted(seen_subjects) == list(range(53))
        assert (seen_epochs == 1).all()

    def test_stratified_kfold_reproducibility_and_seed_effect(self):
        subject_id, y = _modma_like_arrays(seed=2)
        f1 = list(kfold_splits(subject_id, n_folds=10, seed=42, y=y))
        f2 = list(kfold_splits(subject_id, n_folds=10, seed=42, y=y))
        f3 = list(kfold_splits(subject_id, n_folds=10, seed=99, y=y))

        for (tr1, te1), (tr2, te2) in zip(f1, f2):
            assert np.array_equal(tr1, tr2)
            assert np.array_equal(te1, te2)

        # Different seeds should produce different fold assignments
        assert any(
            not np.array_equal(te1, te3) for (_, te1), (_, te3) in zip(f1, f3)
        )


class TestLosoSplits:
    def test_loso_53_folds_deterministic_order(self):
        subject_id, _ = _modma_like_arrays(seed=3)
        folds = list(loso_splits(subject_id))
        assert len(folds) == 53

        seen_epochs = np.zeros(len(subject_id), dtype=int)
        for k, (tr, te) in enumerate(folds):
            te_subs = np.unique(subject_id[te]).tolist()
            assert te_subs == [k]
            assert k not in set(subject_id[tr].tolist())
            # All remaining epochs of held-out subject must be in te
            assert len(te) == int((subject_id == k).sum())
            seen_epochs[te] += 1

        assert (seen_epochs == 1).all()


class TestMakeValSplit:
    def test_never_accesses_outer_test_participants_and_is_reproducible(self):
        subject_id, y = _modma_like_arrays(seed=4)
        folds = list(kfold_splits(subject_id, n_folds=10, seed=0, y=y))
        train_idx, test_idx = folds[0]
        outer_test_subs = set(subject_id[test_idx].tolist())

        # Corrupt outer test labels to prove make_val_split never inspects outer test entries
        y_poisoned = y.copy()
        y_poisoned[test_idx] = -999

        sub_tr_1, val_1 = make_val_split(
            train_idx, subject_id, y=y_poisoned, val_fraction=0.15, seed=7
        )
        sub_tr_2, val_2 = make_val_split(
            train_idx, subject_id, y=y_poisoned, val_fraction=0.15, seed=7
        )

        assert np.array_equal(sub_tr_1, sub_tr_2)
        assert np.array_equal(val_1, val_2)

        sub_tr_subs = set(subject_id[sub_tr_1].tolist())
        val_subs = set(subject_id[val_1].tolist())
        assert sub_tr_subs.isdisjoint(val_subs)
        assert sub_tr_subs.isdisjoint(outer_test_subs)
        assert val_subs.isdisjoint(outer_test_subs)
        assert np.array_equal(np.sort(np.concatenate([sub_tr_1, val_1])), np.sort(train_idx))
        assert set(np.unique(y[val_1]).tolist()) == {0, 1}


class TestInvalidInputs:
    def test_invalid_subject_ids_and_n_folds(self):
        with pytest.raises(ValueError):
            list(kfold_splits(np.array([])))
        with pytest.raises(ValueError):
            list(kfold_splits(np.zeros((2, 4), dtype=int)))
        with pytest.raises(ValueError):
            list(kfold_splits(np.array([0.0, np.nan, 1.0])))
        with pytest.raises(ValueError):
            list(kfold_splits(np.arange(10), n_folds=1))
        with pytest.raises(ValueError):
            list(kfold_splits(np.arange(5), n_folds=6))
        with pytest.raises(ValueError):
            list(kfold_splits(np.arange(10), n_folds=True))
        with pytest.raises(ValueError):
            list(loso_splits(np.array([0, 0, 0])))

    def test_inconsistent_or_mismatched_labels(self):
        sids = np.array([0, 0, 1, 1, 2, 2])
        with pytest.raises(ValueError):
            list(kfold_splits(sids, n_folds=2, y=np.array([0, 1, 0])))
        with pytest.raises(ValueError, match="inconsistent labels"):
            list(kfold_splits(sids, n_folds=2, y=np.array([0, 1, 0, 0, 1, 1])))

    def test_invalid_make_val_split_args(self):
        sids = np.array([0, 0, 1, 1, 2, 2])
        with pytest.raises(ValueError):
            make_val_split(np.array([]), sids)
        with pytest.raises(ValueError):
            make_val_split(np.array([0, 1]), sids)  # only 1 training subject
        with pytest.raises(ValueError):
            make_val_split(np.array([0, 2, 6]), sids)  # out of bounds
        with pytest.raises(ValueError):
            make_val_split(np.array([0, 0, 2]), sids)  # duplicate index
        with pytest.raises(ValueError):
            make_val_split(np.array([0, 1, 2, 3]), sids, val_fraction=0.0)
        with pytest.raises(ValueError):
            make_val_split(np.array([0, 1, 2, 3]), sids, val_fraction=1.0)

