import copy
from itertools import product

import torch

from premap2.sampling import calc_samples
from premap2.splitting import (
    calc_priority,
    split_node_batch,
    stabilize_on_samples,
)
from premap2.utils import WithActivations, split_contains2
from tests.premap2.utils import model_linear


class Mock:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def add_domain(self, *args):
        pass


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
    model = model_linear(2, 5, 5, 2)
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
    stabilize_on_samples(
        [samples],
        [history],
        copy.deepcopy(lower),
        copy.deepcopy(upper),
        store_splits=True,
    )
    if sum(len(i) for i, _ in history) == 0:
        return test_sample_stability()
    assert split_contains2(history, samples.activations).all()

    history = [([], []) for _ in lower]
    stabilize_on_samples(
        [samples],
        [history],
        lower,
        upper,
        domains=[Mock(history=history)],  # type: ignore
        domain_list=Mock(),  # type: ignore
        readd_splits=True,
    )
    assert sum(len(i) for i, _ in history) > 0


def test_split_node():
    lower = torch.tensor([[0.0, 0.0]])
    upper = torch.tensor([[1.0, 1.0]])
    model = model_linear(2, 5, 5, 2)
    samples = calc_samples((lower, upper), model, num=10)
    lower = [act.min(0)[0] - 0.1 for act in samples.activations]
    upper = [act.max(0)[0] + 0.1 for act in samples.activations]
    samples.priority = [torch.ones_like(a[0]) for a in samples.activations]
    history = [([], []) for _ in range(3)]
    for under, sign in product([True, False], [1.0, -1.0]):
        split_node_batch(
            Mock(x=Mock(ptb=Mock(eps=0.0, norm=1.0), data=torch.ones((2,)))),  # type: ignore
            Mock(),  # type: ignore
            under,
            [v[None] for v in lower],
            [v[None] for v in upper],
            {},
            [None],
            [None],
            [
                Mock(
                    preimg_A=torch.zeros((1, 2)),
                    preimg_b=sign * torch.ones((1,)),
                    history=history,
                    volume=1.0,
                    lower_all=lower,
                    upper_all=upper,
                    preimg_vol=1.0,
                )  # type: ignore
            ],
            torch.ones((1, 1)),
            torch.zeros((1,)),
            [history],
            [samples],
            [(1, 1)],
            False,
        )
