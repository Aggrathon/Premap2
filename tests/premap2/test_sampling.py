import torch

from premap2.sampling import (
    Domain,
    box_sample,
    calc_samples,
    fill_box_samples,
    fill_rejection_samples,
)
from premap2.utils import WithActivations, assert_history_contains, ess_exp
from tests.premap2.utils import assert_deep_equal, is_close, model_conv, model_linear


def test_box():
    X = box_sample(torch.zeros(1, 2), torch.ones((1, 2)), 1)
    assert X[0, 0] != X[0, 1]


def test_sample():
    for model, x, use_mask in [
        (model_linear(20, 15, 10, 5), torch.zeros(1, 20), True),
        (model_linear(20, 15, 10, 5), torch.zeros(1, 20), False),
        (model_conv(3, 4, 5, 4, 1), torch.zeros(1, 3, 5, 5), True),
        (model_conv(3, 4, 5, 4, 1), torch.zeros(1, 3, 5, 5), False),
    ]:
        if use_mask:
            lu = (x, x + (torch.rand(*x.shape) * 0.8).round())
            mask = (lu[0] < lu[1])[0]
        else:
            lu = (x, x + 1)
            mask = None
        s = calc_samples(Domain(*lu), model, 100, mask)
        assert len(s) == 500
        s.activations = None
        s.X = s.X[-50:]
        s.y = s.y[-50:]
        s = calc_samples(Domain(*lu), model, 100, mask)
        s2 = s.to("cpu:0")
        assert torch.equal(s.y, s2.y)
        split = [
            u.flatten().nonzero()[0].cpu().item()
            if u.flatten().count_nonzero() > 0
            else 0
            for u in s.unstable()
        ]
        val = [s.activations[i][0].flatten()[j] >= 0 for i, j in enumerate(split)]
        for i, (j, v) in enumerate(zip(split, val)):
            s = s.split(i, j)[1 - int(v)]
        assert_history_contains(s.activations, s.history)
        s = calc_samples(s, model, 100, mask)
        s = calc_samples(s, model, 140, mask)
        s.activations = None
        s.X = s.X[-50:]
        s.y = s.y[-50:]
        s = calc_samples(s, model, 100, mask)
        s = calc_samples(s, model, 140, mask)
        s.poly_A = x[:, mask] if use_mask else x
        s.poly_b = torch.zeros((1,))
        s.activations = None
        s.X = s.X[-50:]
        s.y = s.y[-50:]
        s = calc_samples(s, model, 100, mask)
        s = calc_samples(s, model, 140, mask)
        s.activations = None
        s.X = s.X[-50:]
        s.y = s.y[-50:]
        log_prob = lambda X: -torch.ones(X.shape[0])  # noqa: E731
        s.log_prob = log_prob(s.X)
        calc_samples(s, model, 100, mask, log_prob)
        calc_samples(s, model, 140, mask, log_prob)
        assert (s.log_prob == -1).all()
        assert is_close(ess_exp(s.log_prob), len(s), 0.01)
        s.activations = None
        s.X = s.X[-50:]
        s.y = s.y[-50:]
        log_prob = lambda X: -X.flatten(1).mean(1)  # noqa: E731
        s.log_prob = log_prob(s.X)
        assert ess_exp(s.log_prob) < len(s)
        calc_samples(s, model, 100, mask, log_prob)
        calc_samples(s, model, 140, mask, log_prob)


def test_split():
    lower = torch.tensor([[0.0, 0.0]])
    upper = torch.tensor([[1.0, 1.0]])
    model = model_linear(2, 3, 2)
    samples = calc_samples(Domain(lower, upper), model, num=10)
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
    assert samples_a.constraints == [([neuron_index], [])]
    assert samples_b.constraints == [([], [neuron_index])]

    assert_history_contains(samples_a.activations, samples_a.constraints)
    assert_history_contains(samples_b.activations, samples_b.constraints)


def test_equal():
    for model, x in [
        (model_linear(20, 15, 10, 5), torch.zeros(1, 20)),
        (model_conv(3, 4, 5, 4, 5), torch.zeros(1, 3, 5, 5)),
    ]:
        y, a = WithActivations(model)(x)
        s1 = Domain(
            X=x,
            y=y,
            lower_in=x - 1.0,
            upper_in=x + 1.0,
            log_prob=torch.ones(1),
            activations=a,
        )
        s1.poly_A = x.expand((5, *x.shape[1:]))
        s1.poly_b = torch.zeros(5)
        s1.lower_As = s1.upper_As = a
        s2 = s1.to("cpu:0")
        assert_deep_equal(s1, s2)
        for s in (s1, s2):
            s.constrain(0, 0, True)
            s.constrain(1, 0, False)
            s.stabilize(2, 2, True)
            s.stabilize(1, 2, False)
        assert_deep_equal(s1, s2)
        s1 = s1.split(0, 1)[int(a[0].ravel()[1] < 0)]
        assert len(s1) > 0
        s2 = s2.split(0, 1)[int(a[0].ravel()[1] < 0)]
        assert_deep_equal(s2, s1.to("cpu:0"))


def test_fill_box_samples():
    model = WithActivations(model_linear(20, 15, 10, 5))
    sm = Domain(lower_in=torch.zeros((1, 20)), upper_in=torch.ones((1, 20)))  # type: ignore
    sm = fill_box_samples(sm, model, num=10)
    assert len(sm) >= 10
    sm = fill_box_samples(sm, model, num=20)
    assert len(sm) >= 20
    log_prob = lambda X: -X.flatten(1).mean(1)  # noqa: E731
    sm.log_prob = log_prob(sm.X)
    sm = fill_box_samples(sm, model, num=40, log_prob=log_prob)
    assert len(sm) >= 40


def test_rejection_hit_and_run():
    model = WithActivations(torch.nn.ReLU())
    sm = Domain(
        -100 * torch.ones(1, 2),
        100 * torch.ones(1, 2),
        torch.zeros(1, 2),
        torch.zeros(1, 2),
    )
    sm.poly_A = torch.cat((torch.eye(2), -torch.eye(2)))
    sm.poly_b = torch.Tensor([0.0, 0.0, 1.0, 1.0])
    sm = fill_rejection_samples(sm, model, 200, hit_batch=10)
    assert 211 > len(sm) > 200
    assert torch.all(sm.X >= 0.0)
    assert torch.all(sm.X <= 1.0)


def test_zero_prob():
    model = WithActivations(torch.nn.ReLU())
    log_prob = lambda X: torch.where(  # noqa: E731
        (X >= 0.0).all(1), torch.Tensor([0.0]), torch.Tensor([-torch.inf])
    )
    sm = Domain(
        -torch.ones(1, 2),
        torch.ones(1, 2),
        torch.zeros(1, 2),
        torch.zeros(1, 2),
    )
    sm = fill_box_samples(sm, model, 100, log_prob=log_prob)
    assert len(sm) >= 100
    assert torch.all(sm.X >= 0.0)
    assert torch.all(sm.X <= 1.0)
    sm = Domain(
        -100 * torch.ones(1, 2),
        100 * torch.ones(1, 2),
        torch.zeros(1, 2),
        torch.zeros(1, 2),
    )
    sm.poly_A = torch.cat((torch.eye(2), -torch.eye(2)))
    sm.poly_b = torch.Tensor([1.0, 1.0, 1.0, 1.0])
    sm = fill_rejection_samples(sm, model, 100, log_prob=log_prob)
    assert len(sm) >= 100
    assert torch.all(sm.X >= 0.0)
    assert torch.all(sm.X <= 1.0)
    sm.X = None
    log_prob = lambda x: torch.as_tensor(-torch.inf).expand(x.size(0))  # noqa: E731
    sm = fill_box_samples(sm, model, 100, log_prob=log_prob)
    assert len(sm) == 0
    sm.X = torch.zeros(1, 2)
    sm.y = sm.activations = sm.log_prob = None
    sm = fill_rejection_samples(sm, model, 100, log_prob=log_prob)
    assert len(sm) == 1
