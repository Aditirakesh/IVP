"""Normalization must use training statistics only (deck slide 28).

normalize.py is Member 2's. Contract:
    stats = fit_normalizer(train_x)        # train_x is (N, C, T)
    out   = apply_normalizer(x, stats)
Skips until src/data/normalize.py exists.
"""
import numpy as np
import pytest


def test_test_set_not_renormalized():
    nm = pytest.importorskip("src.data.normalize")
    rng = np.random.default_rng(1)
    train = rng.normal(2.0, 3.0, size=(200, 8, 100)).astype(np.float32)
    test = rng.normal(10.0, 1.0, size=(50, 8, 100)).astype(np.float32)  # different offset on purpose

    st = nm.fit_normalizer(train)
    tr = nm.apply_normalizer(train, st)
    te = nm.apply_normalizer(test, st)

    assert abs(tr.mean()) < 0.05 and abs(tr.std() - 1) < 0.05
    # train stats put test near (10-2)/3 = 2.7. if it came out ~0 the
    # test set got normalized with its own mean, which is a leak
    assert te.mean() > 2
