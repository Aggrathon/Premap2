from typing import Callable, Iterator
from warnings import warn

import torch

from premap2.domains import Domain
from premap2.raycast import raycast_batch
from premap2.tighten_bounds import tighten_backwards
from premap2.utils import (
    IS_TEST_OR_DEBUG,
    AppendableTensor,
    CycleTensor,
    WithActivations,
    assert_history_contains,
    assert_polytope_contains,
    ess_exp,
    expand_patch,
    history_contains,
    polytope_contains,
)


@torch.no_grad()
def calc_samples(
    domain: Domain,
    model: torch.nn.Module | WithActivations,
    num: int = 2_000,
    mask: torch.Tensor | None = None,
    log_prob: None | Callable[[torch.Tensor], torch.Tensor] = None,
    *,
    debug: bool = IS_TEST_OR_DEBUG,
) -> Domain:
    """Generate uniform samples from the domain and compute their activations.

    Args:
        x: Previous samples or bounds for the input.
        model: Model to compute activations.
        num: Number of samples.
        log_prob: Function for weighting the samples.
        debug: Enable additional asserts.

    Returns:
        samples: Generated samples.
    """
    wodel = model if isinstance(model, WithActivations) else WithActivations(model)
    if domain.X is not None:
        assert domain.lower_in is not None
        if debug:
            assert (domain.X >= domain.lower_in).all().cpu().item()
            assert (domain.X <= domain.upper_in).all().cpu().item()
        if domain.activations is None:
            domain.y, domain.activations = wodel(domain.X)
        if log_prob is not None and domain.log_prob is None:
            domain.log_prob = log_prob(domain.X)
    size = len(domain)
    if size > num * 8 // 10:
        pass  # Enough samples
    elif not domain.history:
        # Without constraints we can just sample the bounding box
        domain = fill_box_samples(domain, wodel, num * 5, mask, log_prob)
    else:
        _tighten_bounds(domain, mask, debug)
        domain = fill_rejection_samples(domain, wodel, num, mask, log_prob, debug=debug)
        domain.selection.clear()
    if domain.log_prob is not None:
        ess = ess_exp(domain.log_prob)
        if ess < len(domain) / 100:
            warn(
                "Effective Sample Size is less than 1/100 of the number of samples"
                f" ({ess / len(domain) * 100:.2g}%)."
                " This means that the sample probabilities are heavily skewed. "
                "Consider changing the `log_prob` function."
            )
    return domain


def _tighten_bounds(
    d: Domain, mask: torch.Tensor | None, debug: bool = IS_TEST_OR_DEBUG
):
    if d.poly_A is None:
        return
    assert d.poly_b is not None and d.lower_in is not None and d.upper_in is not None
    if mask is not None:
        ml, mu = d.lower_in[:, mask], d.upper_in[:, mask]
        ml, mu = tighten_backwards(d.poly_A, d.poly_b, ml, mu, debug=debug)
        lower, upper = d.lower_in.clone(), d.upper_in.clone()
        lower[:, mask], upper[:, mask] = ml, mu
        if debug and d.X is not None:
            X = d.X[:, mask]
            assert_polytope_contains(X, d.poly_A, d.poly_b, ml, mu)
    else:
        lower, upper = tighten_backwards(
            d.poly_A, d.poly_b, d.lower_in, d.upper_in, debug
        )
        if debug and d.X is not None:
            assert_polytope_contains(d.X, d.poly_A, d.poly_b, lower, upper)
    d.lower_in, d.upper_in = lower, upper


def box_sample(
    lower: torch.Tensor, upper: torch.Tensor, num: int = 10_000
) -> torch.Tensor:
    """Generate uniform samples from a bounding box.

    Args:
        lower: Lower bound.
        upper: Upper bound.
        num: Number of samples.

    Returns:
        X: samples.
    """
    X = torch.rand(num, *lower.shape[1:], dtype=lower.dtype, device=upper.device)
    return lower + X * (upper - lower)


def fill_box_samples(
    sm: Domain,
    model: WithActivations,
    num: int,
    mask: torch.Tensor | None = None,
    log_prob: None | Callable[[torch.Tensor], torch.Tensor] = None,
    refill: bool = True,
) -> Domain:
    """Generate uniform samples from a bounding box.

    Args:
        sm: Samples to fill with new samples.
        upper: Upper bound.
        model: `torch.nn.Module` wrapped in a `WithActivations`.
        num: Number of samples.
        log_prob: Function for weighting the samples.
        refill: Run multiple times if the `log_prob` discards samples.

    Returns:
        The `sm` object with additional samples.
    """
    assert sm.lower_in is not None and sm.upper_in is not None
    if mask is None:
        X = box_sample(sm.lower_in, sm.upper_in, num)
    else:
        X = box_sample(sm.lower_in[:, mask], sm.upper_in[:, mask], num)
        X = expand_patch(X, sm.lower_in, mask)
    if log_prob is not None:
        w = log_prob(X)
        if torch.any(w.isneginf()):
            inside = ~w.isneginf()
            X, w = X[inside], w[inside]
            if w.size(0) == 0 and not refill:
                return sm
        sm.log_prob = w if sm.log_prob is None else torch.cat((sm.log_prob, w))
    if sm.X is None:
        sm.X = X
        sm.y, sm.activations = model(sm.X)
    else:
        sm.X = torch.cat((sm.X, X))
        if sm.activations is None or sm.y is None:
            sm.y, sm.activations = model(sm.X)
        else:
            y, act = model(X)
            sm.y = torch.cat((sm.y, y))
            sm.activations = [torch.cat(a) for a in zip(sm.activations, act)]
    if refill and len(sm) < num:
        if len(sm) < num // 8:
            warn(
                f"The `log_prob` function discarded {(1.0 - len(sm) / num) * 100:.1f}% of the initial samples."
                " This will make sampling difficult, consider changing the `log_prob` function."
            )
        for _ in range(20):
            fill_box_samples(sm, model, num, mask, log_prob, refill=False)
            if len(sm) >= num:
                break
    return sm


def hit_and_run_generate(
    X: torch.Tensor,
    A: torch.Tensor,
    b: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    batch: int = 128,
    steps: int = 5,
) -> Iterator[torch.Tensor]:
    """Generate uniform samples from a polytope.

    Args:
        X: Previous samples
        A: Polytope constraints coefficients.
        b: Polytope constraints biases.
        lower: Lower bound.
        upper: Upper bound.
        batch: Batch size.
        steps: Thinning steps.

    Returns:
        Generator of batches of uniform samples from the polytope.
    """
    assert len(X) > 0, "Hit-and-Run sampling requires initial samples"
    buffer = CycleTensor(X, batch * 5)
    x = buffer.get()[-batch:].clone()
    shape = [batch] + [1] * (len(x.shape) - 1)
    counter = 0
    while counter < 100:
        if x.shape[0] < batch:
            x = torch.cat((x, buffer.get_random(batch - x.shape[0])))
        for _ in range(steps):
            dir = torch.normal(0.0, 1.0, x.shape, dtype=x.dtype, device=x.device)
            # Reflection when x is at the bounding box to avoid sampling issues
            reflect = ((dir >= 0) | (x > lower)) & ((dir <= 0) | (x < upper))
            dir = torch.where(reflect, dir, -dir)
            # Handle locked dimensions
            dir = torch.where(lower == upper, 0.0, dir)
            # Normalize
            dir = dir / dir.abs().flatten(1).max(1)[0].view(shape)
            back, forward = raycast_batch(
                x, dir, A[None], b[None], lower[None], upper[None], verify=False
            )
            sample = torch.rand(forward.shape, device=x.device, dtype=x.dtype)
            x = x + dir * (sample * (forward - back) + back).view(shape)
        x = x[polytope_contains(x, A, b, lower, upper)]
        if len(x):
            yield x
            counter = 0
            buffer.append(x)
        else:
            counter += 1
    raise Exception("Hit and run sampling failed to produce any samples")


def fill_rejection_samples(
    sm: Domain,
    model: WithActivations | torch.nn.Module,
    num: int,
    mask: torch.Tensor | None = None,
    log_prob: None | Callable[[torch.Tensor], torch.Tensor] = None,
    max_iter: int = 20,
    hit_batch: int = 256,
    *,
    debug: bool = IS_TEST_OR_DEBUG,
) -> Domain:
    """Rejection sampling for a ReLU split domain.

    Args:
        sm: Samples.
        model: The model to compute the activations.
        num: The (minimum) number of samples to return.
        log_prob: Function for weighting the samples.
        max_iter: The number of batches to try.
        debug: Activate additional asserts. Defaults to False unless debugging.

    Returns:
        X: The sampled points.
        y: The labels of the sampled points.
        activations: The activations of the sampled points.
    """
    reserve = num + hit_batch
    Xs = AppendableTensor(sm.X, reserve)
    ys = AppendableTensor(sm.y, reserve)
    model = model if isinstance(model, WithActivations) else WithActivations(model)
    if sm.activations is None:
        acts = [AppendableTensor(reserve=reserve) for _ in range(model.layers)]
    else:
        acts = [AppendableTensor(a, reserve) for a in sm.activations]
        sm.activations = None
    if log_prob is not None:
        ws = AppendableTensor(sm.log_prob, reserve)
    assert sm.lower_in is not None and sm.upper_in is not None
    if mask is not None:
        lower, upper = sm.lower_in[:, mask], sm.upper_in[:, mask]
    else:
        lower, upper = sm.lower_in, sm.upper_in

    hit_and_run = False
    attempts = 0
    while attempts < max_iter:
        if hit_and_run:
            if isinstance(hit_and_run, bool):
                assert sm.poly_A is not None and sm.poly_b is not None
                if mask is None:
                    hit_and_run = hit_and_run_generate(
                        Xs.get(), sm.poly_A, sm.poly_b, lower, upper, hit_batch
                    )
                else:
                    hit_and_run = hit_and_run_generate(
                        Xs.get()[:, mask],
                        sm.poly_A,
                        sm.poly_b,
                        lower,
                        upper,
                        hit_batch,
                    )
            X = next(hit_and_run)
            attempts += 4 * hit_batch / num
        else:
            X = box_sample(lower, upper, num)
            X = X[polytope_contains(X, sm.poly_A, sm.poly_b)]
            hit_and_run = X.shape[0] < num // 100
            attempts += 1
        if X is None or X.shape[0] == 0:
            continue
        if mask is not None:
            X = expand_patch(X, sm.lower_in, mask)
        y, act = model(X)
        inside = history_contains(sm.history, act)
        X = X[inside]
        if X.shape[0] == 0:
            del X, y, act, inside
            continue
        if X.shape[0] < y.shape[0]:
            y = y[inside]
            act = [a[inside] for a in act]
        if log_prob is not None:
            w = log_prob(X)
            if torch.any(w.isneginf()):
                inside = ~w.isneginf()
                X, y, w = X[inside], y[inside], w[inside]
                act = [a[inside] for a in act]
            ws.append(w)
        Xs.append(X)
        ys.append(y)
        for a1, a2 in zip(acts, act):
            a1.append(a2)
        del X, y, act, inside
        if len(Xs) >= num * (1 + (hit_and_run is False)):
            break
    if len(Xs) == len(sm):
        warn("Failed to sample even a single additional point")
    sm.X = Xs.get()
    sm.y = ys.get()
    sm.activations = [act.get() for act in acts]
    if log_prob is not None:
        sm.log_prob = ws.get()
    if debug:
        assert_history_contains(sm.activations, sm.history)
    return sm
