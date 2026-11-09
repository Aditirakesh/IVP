"""Member 1's test. Fails if CDL-MDDNet goes over 100k parameters or breaks the contract."""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from models.cdl_mddnet import CDLMDDNet, count_parameters  # noqa: E402

LIMIT = 100_000


@pytest.mark.parametrize("n_channels", [128, 64, 32, 16])
def test_under_100k_parameters(n_channels):
    # worst case: adversarial head with 53 subjects (LOSO train fold uses 52)
    model = CDLMDDNet(n_channels=n_channels, n_subjects=53)
    assert count_parameters(model) < LIMIT


@pytest.mark.parametrize("n_channels", [128, 32, 16])
def test_forward_contract(n_channels):
    model = CDLMDDNet(n_channels=n_channels, n_samples=1000, n_classes=2).eval()
    out = model(torch.randn(3, n_channels, 1000))
    assert out.shape == (3, 2)


def test_train_forward_with_subject_head():
    model = CDLMDDNet(n_channels=128, n_subjects=48)
    cls, subj = model.forward_train(torch.randn(5, 128, 1000), lambd=0.5)
    assert cls.shape == (5, 2) and subj.shape == (5, 48)


def test_gradient_reversal_flips_sign():
    model = CDLMDDNet(n_channels=16, n_subjects=4).eval()  # dropout off so both passes match
    x = torch.randn(2, 16, 1000)
    _, subj = model.forward_train(x, lambd=1.0)
    subj.sum().backward()
    g_rev = model.spatial[0].weight.grad.clone()
    model.zero_grad()
    f = model.features(x)
    model.subject_head(f).sum().backward()
    g_plain = model.spatial[0].weight.grad
    assert torch.allclose(g_rev, -g_plain, atol=1e-5)
