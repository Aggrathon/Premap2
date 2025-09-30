import torch

from premap2.sampling import (
    LinearBounds,
    calc_samples,
    get_constraints,
    get_hit_and_run_samples,
    sparse_patches_to_matrix,
)
from premap2.utils import polytope_contains, split_contains, split_contains2
from tests.premap2.utils import model_conv, model_linear


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
    assert polytope_contains(X, A, b, lower, upper).all().cpu().item()
    layers = [LinearBounds(lA, lb, uA, ub, y.min(0)[0] + 1e4, y.max(0)[0] - 1e-4)]
    A, b = get_constraints(history, layers, lower, upper)
    assert A is not None
    ycon = polytope_contains(y, None, None, layers[0].lower, layers[0].upper)
    xcon = polytope_contains(X, A, b, lower, upper)
    assert torch.allclose(ycon, xcon)
    assert not ycon.all().cpu().item()
    X = get_hit_and_run_samples(X[xcon], A, b, lower, upper, samples=100)
    assert polytope_contains(X, A, b, lower, upper).all().cpu().item()


def test_sample():
    for model, x in [
        (model_linear(20, 15, 10, 5), torch.zeros(1, 20)),
        (model_linear(20, 15, 10, 5), torch.zeros(1, 20)),
        (model_conv(3, 4, 5, 4, 1), torch.zeros(1, 3, 5, 5)),
        (model_conv(3, 4, 5, 4, 1), torch.zeros(1, 3, 5, 5)),
    ]:
        s = calc_samples((x, x + 1), model, 100)
        assert len(s) >= 100
        xu = x + (torch.rand(x.shape) > 0.3).type(x.dtype)
        s = calc_samples((x, xu), model, 40)
        s2 = s.to("cpu:0")
        assert torch.equal(s.y, s2.y)
        split = [
            u.flatten().nonzero()[0].cpu().item()
            if u.flatten().count_nonzero() > 0
            else 0
            for u in s.unstable()
        ]
        val = [s.activations[i][0].flatten()[j] >= 0 for i, j in enumerate(split)]
        hist = [([j], [1 if v else -1]) for j, v in zip(split, val)]
        for i, (j, v) in enumerate(zip(split, val)):
            s = s.split(i, j)[1 - int(v)]
        assert split_contains2(hist, s.activations).all().cpu().item()
        s = calc_samples(s, model, 100, hist)
        s = calc_samples(s, model, 100, hist)
        s = calc_samples(s, model, 100, hist)
        s = calc_samples(s, model, 100, hist)
        s.activations = None
        s = calc_samples(s, model, 100, hist)
        s.A = x[:, s.mask] if s.mask is not None else x
        s.b = torch.zeros((1,))
        s = calc_samples(s, model, len(s) * 3, hist)


def test_split():
    lower = torch.tensor([[0.0, 0.0]])
    upper = torch.tensor([[1.0, 1.0]])
    model = model_linear(2, 3, 2)
    samples = calc_samples((lower, upper), model, num=10)
    layer_index = 0
    active_count = (samples.activations[layer_index] >= 0).count_nonzero(0)
    neuron_index = int(((active_count - 5).abs().argmin()).cpu().item())
    samples_a, samples_b = samples.split(layer_index, neuron_index)

    assert len(samples_a) + len(samples_b) == len(samples)
    assert (samples_a.activations[layer_index].flatten(1)[:, neuron_index] >= 0).all()
    assert (samples_b.activations[layer_index].flatten(1)[:, neuron_index] <= 0).all()

    # Ensure that the constraints are added correctly
    if len(samples_a) == 0 or len(samples_b) == 0:
        samples_a.constrain(layer_index, neuron_index, active=True)
        samples_b.constrain(layer_index, neuron_index, active=False)
    assert samples_a.constraints == [([], [neuron_index])]
    assert samples_b.constraints == [([neuron_index], [])]

    assert split_contains(samples_a.constraints, samples_a.activations).all()
    assert split_contains(samples_b.constraints, samples_b.activations).all()


def test_sparse_to_matrix():
    from premap2.sampling import Patches

    img = torch.rand((1, 3, 10, 10))
    p = torch.rand(5, 1, 4, 4, 3, 3, 3)
    p = Patches(p, 2, 0, p.shape, output_shape=p.shape[:4])
    idx = [10, 33]
    a = p.to_matrix(img.shape)[0][idx]
    b = sparse_patches_to_matrix(p, idx, img.shape)[0]
    assert torch.allclose(a, b)
