import pytest
import torch

from premap2.utils import (
    AppendableTensor,
    CycleTensor,
    assert_bounds,
    ess_exp,
    expand_patch,
    history_to_index,
    history_to_sign,
)
from tests.premap2.utils import is_close


def test_bounds():
    assert_bounds(torch.rand((3,)), 0.0, 1.0)
    assert_bounds(torch.rand((3,)), torch.zeros(()), torch.ones(()))
    assert_bounds(torch.rand((3,)), torch.zeros((3,)), torch.ones((3,)))
    assert_bounds([torch.rand((3,))], [0.0], [1.0])
    assert_bounds([torch.rand((3,))], [0.0], torch.ones((1,)))
    with pytest.raises(AssertionError):
        assert_bounds(torch.rand((3,)), 1.0, 0.0)


def test_ess():
    def ess(w: torch.Tensor) -> float:
        return (w.sum() ** 2 / (w**2).sum()).item()

    for w in (
        torch.ones(10) * 0.5,
        torch.rand(10),
        torch.rand(10),
        torch.rand(10),
    ):
        wl = torch.log(w)
        assert is_close(ess(w), ess_exp(wl))


def test_append():
    for x in (torch.rand(10, 5), torch.rand(10, 5, 3), torch.rand(10)):
        a = AppendableTensor()
        a.append(x[:2])
        assert torch.equal(x[:2], a.get())
        a = AppendableTensor(x[:3])
        a.append(x[3:4])
        a.append(x[4:9])
        a.append(x[9:10])
        assert torch.equal(x, a.get())


def test_cycle():
    x = torch.arange(12).reshape(4, 3)
    ct = CycleTensor(x[:1], 3)
    assert torch.equal(ct.get(), x[:1])
    assert len(ct) == 1
    ct.append(x[1:2])
    assert torch.equal(ct.get(), x[:2])
    ct.append(x[2:4])
    assert torch.equal(ct.get(), x[[3, 1, 2]])
    ct.append(x[3:])
    assert torch.equal(ct.get(), x[[3, 3, 2]])
    ct = CycleTensor(x, 3)
    assert torch.equal(ct.get(), x[-3:])
    assert len(ct) == 3


def test_expand_patch():
    X = torch.rand(10, 5)
    mask = torch.tensor([True, False, True, False, True])
    X[..., ~mask] = X[0, ~mask]
    Xm = X[..., mask]
    assert Xm.shape == (10, 3)
    assert torch.allclose(X, expand_patch(X, X[0], mask))
    assert torch.allclose(X, expand_patch(X, X[0], mask[None]))
    assert torch.allclose(X, expand_patch(X, X[:1], mask))
    assert torch.allclose(X, expand_patch(X, X[:1], mask[None]))
    X = torch.rand(10, 3, 5, 5)
    mask = torch.zeros(3, 5, 5, dtype=torch.bool)
    mask[..., 1:3, 2:4] = True
    X[..., ~mask] = X[0, ~mask]
    Xm = X[..., mask]
    assert Xm.shape == (10, 12)
    assert torch.allclose(X, expand_patch(X, X[0], mask))
    assert torch.allclose(X, expand_patch(X, X[0], mask[None]))
    assert torch.allclose(X, expand_patch(X, X[:1], mask))
    assert torch.allclose(X, expand_patch(X, X[:1], mask[None]))
    Xm = X[..., 1:2, 2:4]
    assert torch.allclose(X, expand_patch(X, X[0], (1, 2, 2, 2)))
    assert torch.allclose(X, expand_patch(X, X[:1], (1, 2, 2, 2)))


def test_history_to():
    history = [
        (torch.arange(0, 10), torch.arange(10, 20)),
        (torch.arange(30, 40), torch.arange(20, 30)),
    ]
    his = history_to_index(history_to_sign(history))
    for (a1, a2), (b1, b2) in zip(history, his):
        assert torch.equal(a1, b1)
        assert torch.equal(a2, b2)
