"""No participant may sit in both train and test (deck slide 30).

splits.py is Member 3's. Contract: kfold_splits(subject_ids, n_folds, seed) and
loso_splits(subject_ids) yield (train_idx, test_idx), both epoch indices into
the per-epoch subject_ids array.

The checker is tested on its own first (a bad split must fail), then pointed
at the real splits. Real tests skip until src/training/splits.py exists.
"""
import numpy as np
import pytest

N_SUBJ = 53  # MODMA: 24 MDD + 29 healthy


def fakeSubjects(seed=0):
    rng = np.random.default_rng(seed)
    per = rng.integers(30, 50, size=N_SUBJ)  # epochs per subject differ
    return np.repeat(np.arange(N_SUBJ), per)


def check(sids, folds, want):
    folds = list(folds)
    assert len(folds) == want, f"{len(folds)} folds, expected {want}"
    seen = np.zeros(len(sids), dtype=int)
    for k, (tr, te) in enumerate(folds):
        assert len(tr) and len(te), f"fold {k} has an empty side"
        both = set(sids[tr]) & set(sids[te])
        assert not both, f"fold {k}: subjects in train and test: {sorted(both)}"
        seen[te] += 1
    assert (seen == 1).all(), "every epoch must be tested exactly once"


# --- reference splits, only used to prove the checker works ---

def epochKfold(sids, k=10, seed=0):
    # the mistake from paper [6]: shuffle epochs, ignore who they belong to
    idx = np.random.default_rng(seed).permutation(len(sids))
    for part in np.array_split(idx, k):
        yield np.setdiff1d(idx, part), part


def subjKfold(sids, k=10, seed=0):
    subs = np.random.default_rng(seed).permutation(np.unique(sids))
    for part in np.array_split(subs, k):
        te = np.isin(sids, part)
        yield np.where(~te)[0], np.where(te)[0]


def subjLoso(sids):
    for s in np.unique(sids):
        te = sids == s
        yield np.where(~te)[0], np.where(te)[0]


def test_checker_catches_epoch_split():
    sids = fakeSubjects()
    with pytest.raises(AssertionError):
        check(sids, epochKfold(sids), 10)


def test_checker_passes_subject_splits():
    sids = fakeSubjects()
    check(sids, subjKfold(sids), 10)
    check(sids, subjLoso(sids), N_SUBJ)


# --- the real thing ---

@pytest.mark.parametrize("seed", [0, 1, 2])
def test_real_kfold(seed):
    sp = pytest.importorskip("src.training.splits")
    sids = fakeSubjects(seed)
    check(sids, sp.kfold_splits(sids, 10, seed), 10)


def test_real_loso():
    sp = pytest.importorskip("src.training.splits")
    sids = fakeSubjects()
    check(sids, sp.loso_splits(sids), N_SUBJ)
