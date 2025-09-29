import copy
from warnings import warn

import torch

from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm
from premap2.tighten_bounds import (
    NewBounds,
    tighten_backwards,
    tighten_bounds,
    tighten_bounds_back,
)
from premap2.utils import WithActivations

from .utils import get_intermediate_bounds, model_conv, model_linear


def create_worst_case(
    lower: torch.Tensor,
    upper: torch.Tensor,
    lA: torch.Tensor,
    uA: torch.Tensor,
    X: torch.Tensor | None = None,
) -> torch.Tensor:
    X_lower = torch.where(lA > 0, lower, upper)
    X_upper = torch.where(uA > 0, upper, lower)
    Xs = [] if X is None else [X]
    for i in range(lower.shape[-1]):
        for Xn in (X_lower, X_upper):
            Xn = Xn.clone()
            Xn[..., i] = lower[..., i]
            Xs.append(Xn.clone())
            Xn[..., i] = upper[..., i]
            Xs.append(Xn)
    return torch.cat(Xs, 0)


def test_tighten():
    for _ in range(40):
        model = model_linear(20, 15, 10, 5)
        x0 = torch.zeros(1, 20)
        lirpa = BoundedModule(model, x0)
        model = WithActivations(model)
        ptb = PerturbationLpNorm(x_L=x0 - 1.0, x_U=x0 + 1.0)
        x = BoundedTensor(x0, ptb)
        C = torch.eye(5)[:, None]
        _, _, A = lirpa.compute_bounds(
            (x,),
            C=C,
            bound_lower=True,
            bound_upper=True,
            return_A=True,
            need_A_only=True,
            needed_A_dict={lirpa.output_name[0]: [lirpa.input_name[0]]},
        )
        bounds = get_intermediate_bounds(lirpa)
        X = torch.rand(1000, 20) * 2 - 1

        A = A[lirpa.output_name[0]][lirpa.input_name[0]]
        lA, lbias = A["lA"][:, 0], A["lbias"][:, 0]
        uA, ubias = A["uA"][:, 0], A["ubias"][:, 0]
        X = create_worst_case(ptb.x_L, ptb.x_U, lA, uA, X)
        y, act = model(X)
        ly = torch.einsum("oi,ni->no", lA, X) + lbias[None]
        uy = torch.einsum("oi,ni->no", uA, X) + ubias[None]
        assert (y >= ly).all().item()
        assert (y <= uy).all().item()

        for a, (lb, ub) in zip(act, bounds.values()):
            assert (lb - 1e-6 < a).all().item()
            assert (ub + 1e-6 > a).all().item()
            assert (lb < ub).all().item()
        old_bounds = copy.deepcopy(bounds)
        bounds = tighten_bounds(lirpa, x, bounds, forward=True)
        for a, (lb, ub), (olb, oub) in zip(act, bounds.values(), old_bounds.values()):
            assert (lb < ub).all().item()
            assert (lb - 1e-6 < a).all().item()
            assert (ub + 1e-6 > a).all().item()
            assert (lb > olb - 1e-6).all().item()
            assert (ub < oub + 1e-6).all().item()

        key = list(bounds)[-1]
        i = (bounds[key][1][0] - bounds[key][0][0]).argmax()
        bounds[key][0][:, i] = act[-1][:, i].mean()
        mask = (act[-1] > bounds[key][0]).all(1)
        if not mask.any().item() or mask.all().item():
            continue
        nb = NewBounds(active=(len(bounds) - 1, 0, i))
        bounds = tighten_bounds(lirpa, x, bounds, nb, forward=True)
        key2 = list(bounds)[-2]
        tightened = (
            (bounds[key2][0] > old_bounds[key2][0]).any().item()  #
            or (bounds[key2][1] < old_bounds[key2][1]).any().item()
        )
        for a, (lb, ub) in zip(act, bounds.values()):
            assert (lb < ub).all().item()
            assert (lb - 1e-6 < a[mask]).all().item()
            assert (ub + 1e-6 > a[mask]).all().item()
        _, _, A = lirpa.compute_bounds(
            (x,),
            C=C,
            method="backward",
            return_A=True,
            bound_lower=True,
            bound_upper=True,
            needed_A_dict={lirpa.output_name[0]: [lirpa.input_name[0]]},
        )

        A = A[lirpa.output_name[0]][lirpa.input_name[0]]
        lA, lbias = A["lA"][:, 0], A["lbias"][:, 0]
        uA, ubias = A["uA"][:, 0], A["ubias"][:, 0]
        X2 = create_worst_case(ptb.x_L, ptb.x_U, lA, uA)
        y2, act2 = model(X2)
        mask2 = (act2[-1] > bounds[key][0]).all(1)
        ly2 = torch.einsum("oi,ni->no", lA, X2[mask2]) + lbias[None]
        uy2 = torch.einsum("oi,ni->no", uA, X2[mask2]) + ubias[None]
        assert (y2[mask2] >= ly2).all().item()
        assert (y2[mask2] <= uy2).all().item()
        for a, (lb, ub) in zip(act2, bounds.values()):
            assert (lb - 1e-6 < a[mask2]).all().item()
            assert (ub + 1e-6 > a[mask2]).all().item()

        ptb2 = PerturbationLpNorm(x_L=-torch.ones(2, 20), x_U=torch.ones(2, 20))
        x2 = BoundedTensor(torch.zeros(2, 20), ptb2)
        new_bounds = {
            k: (torch.cat((lb, lb - 0.1)), torch.cat((ub, ub + 0.1)))
            for k, (lb, ub) in bounds.items()
        }
        new_bounds = tighten_bounds(lirpa, x2, new_bounds, forward=True)
        for a, (lb, ub), (olb, oub) in zip(
            act, new_bounds.values(), old_bounds.values()
        ):
            assert (lb[:1] - 1e-6 < a)[mask].all().item()
            assert (ub[:1] + 1e-6 > a)[mask].all().item()
            assert (lb < ub).all().item()
            assert (lb > olb - 1e-6).all().item()
            assert (ub < oub + 1e-6).all().item()

        nb.add_active(len(bounds) - 1, 0, torch.arange(10))
        tighten_bounds(lirpa, x2, new_bounds, nb, forward=True)
        if tightened:
            return
    warn("test_tighten: Could not find a case that tightened the previous bounds.")


def test_tighten_conv():
    model = model_conv(3, 4, 5, 4, 1)
    x0 = torch.zeros(2, 3, 5, 5)
    lirpa = BoundedModule(model, x0[:1])
    ptb = PerturbationLpNorm(x_L=x0[:1], x_U=x0[:1] + 1.0)
    x = BoundedTensor(x0[:1], ptb)
    lirpa.compute_bounds((x,), method="backward")
    bounds = get_intermediate_bounds(lirpa)
    bounds[lirpa.input_name[0]] = (x0, x0 + 1)
    old_bounds = copy.deepcopy(bounds)
    key = list(bounds)[-1]
    bounds[key][0][0, 0] = (bounds[key][0][0, 0] + bounds[key][1][0, 0]) * 0.5
    bounds[key][0][1, 0] = (bounds[key][0][1, 0] + bounds[key][1][1, 0]) * 0.5
    nb = NewBounds(active=(len(bounds) - 2, 0, 0), inactive=(len(bounds) - 2, 0, 0))
    bounds = tighten_bounds(lirpa, x, bounds, nb, forward=True)
    bounds = tighten_bounds(lirpa, x, bounds, None, forward=True)
    for k, (ol, ou) in old_bounds.items():
        (nl, nu) = bounds[k]
        assert (nl > ol - 1e-6).all().item()
        assert (nu < ou + 1e-6).all().item()
        assert (nl < nu).all().item()


def test_back():
    w = 2
    for _ in range(10):
        lA = torch.normal(-0.1, 1.0, (2, 3, 5))
        uA = lA
        lbias = torch.zeros(2, 3) - 0.2
        ubias = torch.zeros(2, 3) + 0.2
        lower = -torch.ones(2, 5) * w
        upper = torch.ones(2, 5) * w
        center = ((upper + lower) / 2.0)[..., None]
        diff = ((upper - lower) / 2.0)[..., None]
        lower_next = lA.matmul(center)[..., 0] - lA.abs().matmul(diff)[..., 0] + lbias
        upper_next = uA.matmul(center)[..., 0] + uA.abs().matmul(diff)[..., 0] + ubias
        X = lower[None] + torch.rand(1000, 2, 5) * (upper[None] - lower[None])
        X = create_worst_case(
            lower[None], upper[None], lA.moveaxis(0, 1), uA.moveaxis(0, 1), X
        )
        ly = torch.einsum("nbi,boi->nbo", X, lA) + lbias[None]
        uy = torch.einsum("nbi,boi->nbo", X, uA) + ubias[None]
        assert (ly <= uy).all().item()
        assert (ly >= lower_next[None]).all().item()
        assert (uy <= upper_next[None]).all().item()

        i = lower_next[0].argmin().item()
        lower_next[0, i] = (lower_next[0, i] + 3 * upper_next[0, i]) / 4.0
        upper_next[1, 2] = (3 * lower_next[1, 2] + upper_next[1, 2]) / 4.0
        lower_alt = lower.clone()
        upper_alt = upper.clone()
        tighten_bounds_back(
            lA,
            lbias,
            uA,
            ubias,
            lower,
            upper,
            lower_next,
            upper_next,
            [(0, [i], []), (1, [], [2])],
        )
        assert (lower <= upper).all().item()
        mask1 = ((ly >= lower_next[None]) & (uy <= upper_next[None])).all(-1)
        mask2 = ((X >= lower[None]) & (X <= upper[None])).all(-1)
        assert (mask1 <= mask2).all().item()
        lower_alt[:1], upper_alt[:1] = tighten_backwards(
            uA[None, 0, i],
            ubias[0, i] - lower_next[0, i],
            lower_alt[:1],
            upper_alt[:1],
        )
        lower_alt[1:], upper_alt[1:] = tighten_backwards(
            -lA[None, 1, 2],
            -lbias[1, 2] + upper_next[1, 2],
            lower_alt[1:],
            upper_alt[1:],
        )
        assert torch.allclose(lower, lower_alt)
        assert torch.allclose(upper, upper_alt)
        if not ((lower[:1] > -w).any().item() or (upper[:1] < w).any().item()):
            continue
        if not ((lower[1:] > -w).any().item() or (upper[1:] < w).any().item()):
            continue
        return
    assert False, "No tightening found"


def test_back_2d():
    lower = -torch.ones(1, 2)
    upper = torch.ones(1, 2)
    upper_next = torch.ones(1, 1)
    lower_next = -2 * torch.ones(1, 1)
    lA = torch.ones(1, 1, 2)
    lbias = torch.zeros(1, 1)
    uA = torch.zeros(1, 1, 2)
    uA[0, 0, 0] = 1.0
    ubias = torch.zeros(1, 1)
    batches = [(0, [0], [0])]
    tighten_bounds_back(
        lA, lbias, uA, ubias, lower, upper, lower_next, upper_next, batches
    )
    assert torch.all(lower == -1).cpu().item()
    assert torch.all(upper == 1).cpu().item()
    lower_next[0, 0] = -0.5
    upper_next[0, 0] = 1.0
    tighten_bounds_back(
        lA, lbias, uA, ubias, lower, upper, lower_next, upper_next, batches
    )
    assert torch.all(upper == 1).cpu().item()
    assert torch.allclose(lower, torch.tensor([[-0.5, -1.0]]))
    upper_next[0, 0] = 0.0
    tighten_bounds_back(
        lA, lbias, uA, ubias, lower, upper, lower_next, upper_next, batches
    )
    assert torch.allclose(upper, torch.tensor([[1.0, 0.5]]))
    assert torch.allclose(lower, torch.tensor([[-0.5, -1.0]]))
