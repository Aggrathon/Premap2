import torch

from premap2.splitting import calc_priority
from premap2.utils import WithActivations
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
