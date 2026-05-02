import torch

from premap2.domains import Domain
from premap2.sampling import calc_samples
from premap2.splitting import (
    calc_priority,
    split_node_batch,
    stabilize_on_samples,
)
from premap2.utils import WithActivations, assert_history_contains
from tests.premap2.utils import model_linear, temp_seed


class Mock:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def add(self, *args):
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
    pri = calc_priority(yc, act, lAs, uAs, lower, upper, unstable, k=8, **coefs)
    assert len(pri) == 8
    for _, l, i in pri:
        assert l < len(act)
        assert i < act[l][0].numel()
    pri = calc_priority(yc, act, lAs, None, lower, upper, unstable, **coefs)
    pri = calc_priority(yc, act, None, uAs, lower, upper, unstable, **coefs)
    coefs = {k: 0.0 for k in coefs}
    calc_priority(yc, act, lAs, uAs, lower, upper, unstable, **coefs)


def test_sample_stability():
    lower = torch.tensor([[0.0, 0.0]])
    upper = torch.tensor([[1.0, 1.0]])
    model = model_linear(2, 5, 5, 5, 2)
    samples = calc_samples(Domain(lower, upper), model, num=10)
    assert samples.activations is not None
    samples.lower_all = [
        torch.minimum(act.min(0)[0][None], -act.new_ones(1))
        for act in samples.activations
    ]
    samples.upper_all = [
        torch.maximum(act.max(0)[0][None], act.new_ones(1))
        for act in samples.activations
    ]
    stabilize_on_samples(
        [samples],
        Mock(),  # type: ignore
        add_empty=True,
    )
    if sum(len(a) + len(b) for a, b in samples.history) == 0:
        return test_sample_stability()
    assert_history_contains(samples.activations, samples.history)


def test_split_node():
    lower = torch.tensor([[0.0, 0.0]])
    upper = torch.tensor([[1.0, 1.0]])
    with temp_seed(42):
        model = model_linear(2, 5, 5, 2)
        samples = calc_samples(Domain(lower, upper), model, num=10)
    samples.lower_all = [act.min(0)[0] - 0.1 for act in samples.activations]
    samples.upper_all = [act.max(0)[0] + 0.1 for act in samples.activations]
    samples.preimg_A = torch.zeros((1, 2))
    samples.preimg_vol = 1.0
    output = torch.eye(2)[:1]
    yo = samples.y @ output.T
    for under in [True, False]:
        samples.preimg_b = (yo.max() + 0.1) if under else (yo.min() - 0.1)
        split_node_batch(
            Mock(x=Mock(ptb=Mock(eps=0.0, norm=1.0), data=torch.ones((2,)))),  # type: ignore
            Mock(output=output, threshold=torch.zeros(1), num_samples=100),  # type: ignore
            under,
            [samples],
            [(1.0, 1, 2)],
            False,
        )
