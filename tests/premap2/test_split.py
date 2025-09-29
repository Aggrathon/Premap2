import torch

from premap2.sampling import calc_samples
from premap2.splitting import (
    calc_priority,
    stabilize_on_samples,
)
from premap2.utils import WithActivations, split_contains2
from tests.premap2.utils import model_linear


def test_heuristic():
    model = WithActivations(model_linear(10, 15, 10, 5))
    X = torch.normal(0.0, 1.0, (100, 10))
    yc, act = model(X)
    unstable = [a[0] > 0 for a in act]
    upper = [a.max(0)[0] + 1.0 for a in act]
    lower = [a.min(0)[0] - 1.0 for a in act]
    lAs = [torch.ones((5, a.shape[1])) for a in act]
    uAs = [torch.ones((5, a.shape[1])) for a in act]
    coefs = dict(
        balance_coef=1.0,
        soft_coef=1.0,
        lower_coef=1.0,
        width_coef=1.0,
        loose_coef=1.0,
        bound_coef=1.0,
        gap_coef=1.0,
        area_coef=1.0,
        under_coef=1.0,
        extra_coef=1.0,
        stable_coef=1.0,
        pure_coef=1.0,
    )
    pri = calc_priority(X, yc, act, lAs, uAs, lower, upper, unstable, **coefs)
    assert len(pri) == len(act)
    for p, a in zip(pri, act):
        assert p.shape == a.shape[1:]
    pri = calc_priority(X, yc, act, lAs, None, lower, upper, unstable, **coefs)
    pri = calc_priority(X, yc, act, None, uAs, lower, upper, unstable, **coefs)
    coefs = {k: 0.0 for k in coefs}
    calc_priority(X, yc, act, lAs, uAs, lower, upper, unstable, **coefs)


def test_sample_stability():
    lower = torch.tensor([[0.0, 0.0]])
    upper = torch.tensor([[1.0, 1.0]])
    model = model_linear(2, 3, 2)
    samples = calc_samples((lower, upper), model, num=10)
    assert samples.activations is not None
    lower = [
        torch.minimum(act.min(0)[0][None], -act.new_ones(1))
        for act in samples.activations
    ]
    upper = [
        torch.maximum(act.max(0)[0][None], act.new_ones(1))
        for act in samples.activations
    ]
    history = [([], []) for _ in lower]
    stabilize_on_samples([samples], [history], lower, upper)
    if sum(len(i) for i, _ in history) == 0:
        return test_sample_stability()
    assert split_contains2(history, samples.activations).all()
