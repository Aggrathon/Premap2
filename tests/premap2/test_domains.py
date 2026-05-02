from copy import copy

import numpy as np
import torch

from premap2.coverage import update_volumes
from premap2.domains import Domain, DomainList, sizeof_tensors
from premap2.sampling import calc_samples
from premap2.utils import WithActivations

from .utils import minimize, model_linear


def test_domains():
    # Test that the priority queue works as expected
    model = model_linear(100, 50, 10, 5)
    domain = calc_samples(Domain(torch.zeros(1, 100), torch.ones(1, 100)), model, 100)
    domain.preimg_A = torch.ones(5, 100)
    domain.preimg_b = torch.ones(5)
    domain.to("cpu:0")
    size = sizeof_tensors(domain)
    domains = DomainList(
        model,
        torch.zeros(1),
        torch.zeros(1),
        True,
        lower_in=torch.zeros(1, 100),
        upper_in=torch.ones(1, 100),
        num_samples=100,
        device_size=size * 2 + 2,
        reduce_size=size * 3 + 3,
        keep=2,
        store_size=size * 4 + 4,
        interval=1,
    )
    assert not domains.get_batch(10)
    for i in [1.0, -np.inf, 2.0, 0.0, -1.0, 5.0, -np.inf, 3.0, -2.0, 2.7, 2.5]:
        d = copy(domain)
        d.priority = i
        assert d.X is not None
        assert d.activations is not None
        domains.add(d)  # type: ignore
        if i == -np.inf:
            assert d.X is None
        elif i < 0:
            assert d.activations is None
    assert domains.pop().priority == 5.0
    for d in domains.pop().split(0, 1):
        d.preimg_vol = d.volume
        domains.add(d)
    assert domains._cache is not None
    domains.pop(-1)
    assert len(domains) == len(list(domains.iter()))
    while len(domains):
        domains.pop()
    assert domain.X is not None
    domain.minimize_memory()
    domain.priority = -np.inf
    domain.minimize_final()
    domains.save({"model": {"name": "tmp"}, "preimage": {"log_prob": None}})
    domains.save(
        {"model": {"name": model}, "preimage": {"log_prob": lambda x: x[:, 0]}}
    )


def test_split_ci():
    lower = torch.zeros(10)
    upper = torch.ones(10)
    model = model_linear(10, 5, 3, 2)
    X = torch.rand((1000, 10))
    y, act = WithActivations(model)(X)
    balance = (y[:, 0] > y[:, 1]).count_nonzero()
    d = Domain(lower, upper, X=X, y=y, activations=act, confidence=True)
    th = torch.zeros(1, 1)

    for under in (True, False):
        c = torch.tensor([[-1.0, 1.0]] if (balance < 500) == under else [[1.0, -1.0]])
        domains = DomainList(model, th, c, under, num_samples=500)
        domains.add(d)

        def loss(X: torch.Tensor, y: torch.Tensor, ab: torch.Tensor) -> torch.Tensor:
            pred = X @ ab[:, :-1].T + ab[:, -1:]
            targ = y @ c.T
            if under:
                off = targ - 0.1 - pred
            else:
                off = pred - targ - 0.1
            return torch.where(off < 0.0, 100.0 * off**2, off).mean()

        def approx(d: Domain):
            if d.preimg_A is None or d.preimg_b is None:
                ab = torch.zeros((1, 11))
            else:
                ab = torch.cat((d.preimg_A.view(1, 10), d.preimg_b.view(1, 1)), 1)
            ab = minimize(ab, lambda ab: loss(d.X, d.y, ab), 2)  # type: ignore
            d.preimg_A, d.preimg_b = ab[:, :-1], ab[:, -1:]

        for i in range(10):
            if domains.finished():
                break
            d = domains.pop()
            layer = i % len(d.activations)
            act = d.activations[layer]
            index = ((act >= 0.0).float().mean(0).flatten() - 0.5).abs().argmin().item()
            da, di = d.split(layer, index, debug=True)
            approx(da)
            update_volumes(
                da, domains.output, domains.threshold, domains.under, debug=True
            )
            approx(di)
            update_volumes(
                di, domains.output, domains.threshold, domains.under, debug=True
            )
            domains.add(da)
            domains.add(di)
        if i == 0:
            return test_split_ci()
        res = torch.load(domains.save({}, confidence=0.95, debug=True))
        assert np.isfinite(res["preimage_ci"]).all()
        assert np.isfinite(res["approx_ci"]).all()
        if res["preimage_vol"] > 0.0:
            assert np.isfinite(res["ratio_ci"]).all()
