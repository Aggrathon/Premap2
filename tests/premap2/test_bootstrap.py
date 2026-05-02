import torch

from premap2.bootstrap import bootstrap_select, bootstrap_split, confidence_interval
from tests.premap2.utils import temp_seed


def test_bootstrap_split():
    n = 100
    mask = torch.randint(0, 2, (n,)).to(torch.bool)
    with temp_seed(42):
        left, right = bootstrap_split(mask, None, 100, debug=True)
    assert ((left.exp() + right.exp() - 1.0).abs() < 1e-5).all().item()
    with temp_seed(42):
        leftp, rightp = bootstrap_split(mask, torch.ones(n), 100, debug=True)
    assert torch.allclose(left, leftp)
    assert torch.allclose(right, rightp)
    left, right = bootstrap_split(mask, torch.rand(n), 100, debug=True)
    assert ((left.exp() + right.exp() - 1.0).abs() < 1e-5).all().item()


def test_bootstrap_select():
    n = 100
    mask = torch.randint(0, 2, (n,)).to(torch.bool)
    left, right = bootstrap_select(mask, mask, None, 100, debug=True)
    assert torch.equal(left, right)
    with temp_seed(42):
        left, right = bootstrap_select(mask, ~mask, None, 100, debug=True)
    assert ((left.exp() + right.exp() - 1.0).abs() < 1e-5).all().item()
    with temp_seed(42):
        leftp, rightp = bootstrap_select(mask, ~mask, torch.ones(n), 100, debug=True)
    assert torch.allclose(left, leftp)
    assert torch.allclose(right, rightp)
    left, right = bootstrap_select(mask, ~mask, torch.rand(n), 100, debug=True)
    assert ((left.exp() + right.exp() - 1.0).abs() < 1e-5).all().item()


def test_confidence_interval():
    assert confidence_interval(torch.arange(101).float(), 0.9) == (5, 95)
    assert confidence_interval(torch.arange(11).float(), 0.7) == (1.5, 8.5)
