import torch

from premap2.constraints import LinearBounds, get_constraints, sparse_patches_to_matrix
from premap2.sampling import hit_and_run_generate
from premap2.utils import assert_polytope_contains, polytope_contains


def test_get_constraints():
    history = [([0, 1], [1])]
    lower = torch.zeros(1, 3)
    upper = torch.ones(1, 3)
    lA = torch.tensor([[1.0, 0.5, 0.2], [0.3, 1.0, 0.1]])
    uA = torch.tensor([[0.8, 0.4, 0.6], [0.1, 0.9, 0.7]])
    lb = -torch.ones(2)
    ub = torch.ones(2)
    X = torch.rand(100, 3) * (upper - lower) + lower
    y = X @ (lA * 0.5 + uA * 0.5).T + (lb + ub)[None] * 0.5
    assert torch.all(X @ lA.T + lb[None] <= y)
    assert torch.all(y <= X @ uA.T + ub[None])
    layers = [LinearBounds(lA, lb, uA, ub, y.min(0)[0] - 1e-6, y.max(0)[0] + 1e-6)]
    A, b = get_constraints(history, layers, lower, upper)
    ycon = polytope_contains(y, None, None, layers[0].lower, layers[0].upper)
    assert ycon.all().cpu().item()
    assert_polytope_contains(X, A, b, lower, upper)
    layers = [LinearBounds(lA, lb + 1, uA, ub, y.min(0)[0] + 1e-3, y.max(0)[0] - 1e-3)]
    A, b = get_constraints(history, layers, lower, upper)
    assert A is not None
    xcon = polytope_contains(X, A, b, lower, upper)
    assert xcon.any().item()
    hr = hit_and_run_generate(X[xcon], A, b, lower, upper)
    for _ in range(6):
        X = next(hr)
        assert_polytope_contains(X, A, b, lower, upper)


def test_sparse_to_matrix():
    from premap2.constraints import Patches

    img = torch.rand((1, 3, 10, 10))
    p = torch.rand(5, 1, 4, 4, 3, 3, 3)
    p = Patches(p, 2, 0, p.shape, output_shape=p.shape[:4])
    idx = [10, 33]
    a = p.to_matrix(img.shape)[0][idx]
    b = sparse_patches_to_matrix(p, idx, img.shape)[0]
    assert torch.allclose(a, b)
