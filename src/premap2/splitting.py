from copy import copy

import numpy as np
import torch

from premap2.coverage import update_volumes
from premap2.domains import Domain, DomainList
from premap2.tighten_bounds import NewBounds, tighten_bounds
from premap2.utils import (
    IS_TEST_OR_DEBUG,
    assert_bounds,
    assert_history_contains,
    history_to_sign,
)

try:
    from auto_LiRPA.bound_general import BoundedTensor, PerturbationLpNorm
    from premap.preimage_beta_crown_solver_relu_split import LiRPAConvNet
except ImportError:
    pass


@torch.no_grad()
def select_node_batch(
    domains: list[Domain],
    output: torch.Tensor,
    num_samples: int = 2_000,
    heuristics: list[tuple[str, float]] | None = None,
    *,
    debug: bool = IS_TEST_OR_DEBUG,
    **coefs,
) -> list[tuple[float, int, int]]:
    """Select which neuron to split for each domain in the batch.

    Args:
        domains: batch of domains.
        output: Output specification as a linear tensor.
        num_samples: Number of samples to use for heuristic calculations.
        heuristics: Key-value pairs of weights for the heuristics.
        debug: Enable additional asserts.
        **coefs: Alternative way of specifying weights for the heuristics.

    Returns:
        Top neuron (priority, layer, index) for each domain.
    """
    if heuristics is not None:
        coefs = coefs | {key: value for key, value in heuristics}
    if coefs:
        coefs = {
            key if "_coef" in key else key + "_coef": value
            for key, value in coefs.items()
        }
    for d in domains:
        mask = d.unstable(False)
        if d.selection:
            d.selection = [
                (v, l, j) for v, l, j in d.selection if mask[l].view(-1)[j].item()
            ]
        if not d.selection or len(d.selection) == 1:
            # `len(d.selection) == 1` is used as a proxy for final split
            assert d.X is not None and d.y is not None and d.activations is not None
            act = d.activations
            d.selection = calc_priority(
                torch.einsum("o...,n...->no", output, d.y[:num_samples]),
                [a[:num_samples] for a in act] if len(d.X) > num_samples else act,
                d.lower_As,
                d.upper_As,
                d.lower_all,
                d.upper_all,
                mask,
                **coefs,
                debug=debug,
            )
    return [d.selection.pop() for d in domains]  # type: ignore


def calc_priority(
    yc: torch.Tensor,
    activations: list[torch.Tensor],
    lAs: list[torch.Tensor] | None,
    uAs: list[torch.Tensor] | None,
    lower: list[torch.Tensor],
    upper: list[torch.Tensor],
    unstable: list[torch.Tensor],
    k: int = 3,
    *,
    balance_coef: float = 0.0,
    soft_coef: float = 0.0,
    lower_coef: float = 0.0,
    gap_coef: float = 0.25,
    width_coef: float = 0.0,
    loose_coef: float = 0.0,
    bound_coef: float = 0.0,
    area_coef: float = 0.75,
    under_coef: float = 0.5,
    extra_coef: float = 1.0,
    stable_coef: float = 1.0,
    pure_coef: float = 0.0,
    debug: bool = IS_TEST_OR_DEBUG,
) -> list[tuple[float, int, int]]:
    """Calculate priorities for the neurons.

    Args:
        yc: Post specification output.
        activations: Activation values for the samples.
        lAs: Lower linear bounds.
        uAs: Upper linear bounds.
        lower: Lower bounds
        upper: Upper bounds.
        unstable: Masks for unstable neurons.
        k: Number of top neurons to return.
        balance_coef: Coefficient for the `balance` heuristic.
        soft_coef: Coefficient for the `soft` heuristic.
        lower_coef: Coefficient for the `lower` heuristic.
        width_coef: Coefficient for the `width` heuristic.
        loose_coef: Coefficient for the `loose` heuristic.
        bound_coef: Coefficient for the `bound` heuristic.
        gap_coef: Coefficient for the `gap` heuristic.
        area_coef: Coefficient for the `area` heuristic.
        under_coef: Coefficient for the `under` heuristic.
        extra_coef: Coefficient for the `extra` heuristic.
        stable_coef: Coefficient for the `stable` heuristic.
        pure_coef: Coefficient for the `pure` heuristic.
        debug: Enable additional asserts.

    Returns:
        Top neurons (priority, layer, index).
    """
    n = yc.shape[0]
    eps = torch.finfo(yc.dtype).eps * 2
    priority = [torch.zeros_like(a[0]) for a in activations]
    if balance_coef > 0.0:  # Balance the split
        for pri, act in zip(priority, activations):
            pri += balance_coef * (1.0 - ((act >= 0).sum(0) * (2 / n) - 1.0).abs())
    if soft_coef > 0.0:  # Soft balance the split
        for pri, act in zip(priority, activations):
            pri += soft_coef * (1.0 - (torch.sigmoid(act).mean(0) * 2.0 - 1.0).abs())
    if area_coef > 0.0:  # Area (lb<x<0) that will get constrained after a split
        if lAs and uAs:
            areas = [
                (lA.abs() + uA.abs()).sum(0) * lb**2 * u
                for lA, uA, lb, u in zip(lAs, uAs, lower, unstable)
            ]
        else:
            areas = [
                A.abs().sum(0) * lb**2 * u
                for A, lb, u in zip(lAs or uAs, lower, unstable)
            ]
        area_norm = area_coef / (max(a.max() for a in areas) + eps)
        for pri, area in zip(priority, areas):
            pri += area * area_norm
    if lower_coef > 0.0:  # Lowest bound
        min_lower = max((lb * u).min() for lb, u in zip(lower, unstable)) + eps
        for pri, lb in zip(priority, lower):
            pri += torch.relu(-lb) * (lower_coef / min_lower)
    if width_coef > 0.0:  # Widest bound
        max_width = max(
            ((ub - lb) * u).max() for lb, ub, u in zip(lower, upper, unstable)
        )
        for pri, lb, ub in zip(priority, lower, upper):
            pri += (ub - lb) * (width_coef / (max_width + eps))
    if extra_coef > 0.0:  # Average (sample) distance from bound to zero when x<0
        dists = []
        if lAs and uAs:
            for lA, uA, act, u in zip(lAs, uAs, activations, unstable):
                dists.append(_heuristic_extra2(lA, uA, act, u, eps))
        else:
            for A, act, u in zip(lAs or uAs, activations, unstable):
                dists.append(_heuristic_extra(A, act, u, eps))
        dist_norm = extra_coef / (max((d.max() for d in dists)) + eps)
        for pri, d in zip(priority, dists):
            pri += d * dist_norm
    if under_coef > 0.0:  # Maximum distance from bound to zero when x<0
        if lAs and uAs:
            unders = [
                ((lA * lb[None]).abs().sum(0) + (uA * lb[None]).abs().sum(0)) * u
                for lA, uA, lb, u in zip(lAs, uAs, lower, unstable)
            ]
        else:
            unders = [
                (A * lb[None]).abs().sum(0) * u
                for A, lb, u in zip(lAs or uAs, lower, unstable)
            ]
        under_norm = under_coef / (max((m.max() for m in unders)) + eps)
        for pri, minf in zip(priority, unders):
            pri += minf * under_norm
    if gap_coef > 0.0:  # Distance to local bound when x=0
        gaps = [
            (-ub * lb) / (ub - lb + eps) * u
            for lb, ub, u in zip(lower, upper, unstable)
        ]
        gap_norm = gap_coef / max((m.max() for m in gaps)) + eps
        for pri, gap in zip(priority, gaps):
            pri += gap * gap_norm
    if bound_coef > 0.0:  # Size difference between bound and sample minmax
        for pri, lb, ub, act in zip(priority, lower, upper, activations):
            bound_gap = ub - lb + eps * 2
            pri += bound_coef * (1.0 - (act.max(0)[0] - act.min(0)[0]) / bound_gap)
    if loose_coef > 0.0:  # Difference between bound and sample minmax
        loose = [
            (ub - lb + act.min(0)[0] - act.max(0)[0]) * u
            for lb, ub, act, u in zip(lower, upper, activations, unstable)
        ]
        loose_norm = loose_coef / (max(m.max() for m in loose) + eps)
        for pri, lo in zip(priority, loose):
            pri += lo * loose_norm
    if stable_coef > 0.0:  # Is the activation stable for the samples
        for pri, act in zip(priority, activations):
            pri += stable_coef * ((act.min(0)[0] >= 0) | (act.max(0)[0] <= 0))
    if pure_coef > 0.0:  # Would the split result in purer preimages
        preimg = (yc.flatten(1) >= 0).all(1)
        for pri, act in zip(priority, activations):
            shape = (-1,) + tuple(1 for _ in act.shape[1:])
            snum = (act >= 0).sum(0)
            lp = (preimg.view(shape) & (act >= 0)).sum(0) / (snum + eps)
            rp = (preimg.view(shape) & (act < 0)).sum(0) / (n - snum + eps)
            pri += pure_coef * (2.0 * torch.maximum((lp - 0.5).abs(), (rp - 0.5).abs()))
    for p, m in zip(priority, unstable):
        p[~m] = -np.inf
    if debug:
        for p, m in zip(priority, unstable):
            assert p.numel() == m.numel()
    top = [
        (v, l, i)
        for l, p in enumerate(priority)
        for vals, inds in (p.view(-1).topk(min(k, p.numel()), sorted=False),)
        for v, i in zip(vals.cpu().numpy(), inds.cpu().numpy())
        if v > -np.inf
    ]
    top = sorted(top)[-k:] if top else [(-np.inf, -1, -1)]
    if debug:
        v, l, i = top[-1]
        for p in priority:
            assert (p <= v).all().item()
        assert (priority[l].view(-1)[i] == v).all().item()
    return top


@torch.jit.script  # type: ignore
def _heuristic_extra(
    A: torch.Tensor, act: torch.Tensor, unstable: torch.Tensor, eps: float
) -> torch.Tensor:
    """JIT this function to (hopefully) fuse the calculations, saving CUDA memory."""
    a = torch.relu(-act)
    m = unstable / ((a <= 0).sum(0) + eps)
    return torch.einsum("b...,n...,...->...", A.abs(), a, m)


@torch.jit.script  # type: ignore
def _heuristic_extra2(
    lA: torch.Tensor, uA: torch.Tensor, act: torch.Tensor, uns: torch.Tensor, eps: float
) -> torch.Tensor:
    """JIT this function to (hopefully) fuse the calculations, saving CUDA memory."""
    a = torch.relu(-act)
    m = uns / ((a <= 0).sum(0) + eps)
    return torch.einsum("b...,n...,...->...", lA.abs() + uA.abs(), a, m)


@torch.no_grad()
def split_node_batch(
    net: "LiRPAConvNet",
    domains: DomainList,
    under: bool,
    selected: list[Domain],
    branching_decision: list[tuple[float, int, int]],
    tighten: bool = True,
    debug: bool = IS_TEST_OR_DEBUG,
):
    """Split the domain on a neuron (batched).

    Any item in `selected_domains` that can be shortcutted is returned to `domains`.

    Args:
        net: LiRPA wrapped network.
        domains: List of domains.
        selected: Domains to split.
        branching_decision: Branching decisons.
        tighten: Tighten the bounds after splitting.
        debug: Activate additional asserts. Defaults to False unless debugging.

    Returns:
        orig_lbs: Split and filtered lower bounds.
        orig_ubs: Split and filtered upper bounds.
        slopes: Split and filtered `slopes`.
        betas: Split and filtered `betas`.
        intermediate_betas: Split and filtered `intermediate_betas`.
        selected_domains: Split and filtered `selected_domains`.
        cs: Output specification as a linear tensor.
        rhs: Right hand side of the output constraint.
        history: Filtered `history`.
        split_history: Split and filtered `history`.
        samples: Split and filtered `samples`.
        branching_decision: Filtered `branching_decision`.
    """
    split_str = ""
    left_domains: list[Domain] = []
    right_domains: list[Domain] = []
    output = domains.output  # type: ignore
    splits = []
    old: list[Domain] = []
    for domain, (priority, layer, index) in zip(selected, branching_decision):
        # Check for fully explored branches (priority of branching decision < 0)
        if priority == -np.inf or domain.priority == -np.inf:
            domain.priority = -np.inf
            domains.add(domain)
            continue

        # Check for shortcuts (domains where we don't need further processing after a split)
        d1, d2 = domain.split(layer, index, debug=debug)
        len1 = len(d1)
        len2 = len(d2)
        split_str += f"({layer}, {index}: {len1} | {len2}) "

        if len1 == 0:
            domains.add(d2)
            if not under:  # Save domain not guaranteed to be outside preimage
                d1.priority = -np.inf
                d1.preimg_vol = d1.approx_vol = 0.0
                domains.add(d1)
            continue
        if len2 == 0:
            domains.add(d1)
            if not under:  # Save domain not guaranteed to be outside preimage
                d2.priority = -np.inf
                d2.preimg_vol = d2.approx_vol = 0.0
                domains.add(d2)
            continue

        update_volumes(d1, domains.output, domains.threshold, under, debug=debug)
        update_volumes(d2, domains.output, domains.threshold, under, debug=debug)

        # For the final split of a branch we must not skip the optimisation
        if domain.selection and (d1.priority == -np.inf or d2.priority == -np.inf):
            domains.add(d1)
            domains.add(d2)
        else:
            left_domains.append(d1)
            right_domains.append(d2)
            splits.append((layer, index))
            old.append(domain)

    if len(split_str) > 101:
        split_str = split_str[:97] + "..."
    print("Splits decision:", split_str[:100])

    selected = left_domains + right_domains
    if debug:
        assert all(len(s) > 0 for s in selected)
        for d in selected:
            assert_history_contains(d.activations, d.history)

    # Tighten the bounds for the splits
    lower_all = [torch.stack(t) for t in zip(*(d.lower_all for d in selected))]
    upper_all = [torch.stack(t) for t in zip(*(d.upper_all for d in selected))]
    history = [history_to_sign(d.history, len(lower_all) - 1) for d in old]
    betas = [d.beta for d in old if d.beta] or None
    alphas = {}
    intermediate_betas = [d.beta_im for d in old if d.beta_im] or None
    cs = output[None]
    rhs = domains.threshold[None]
    if len(selected) > 0:
        alphas = {
            k: {kk: torch.cat([d.alpha[k][kk] for d in old], 2) for kk in v.keys()}
            for k, v in old[0].alpha.items()
        }
        cs = torch.cat((cs,) * len(old))
        rhs = torch.cat((rhs,) * len(old))
        del old
        if tighten:
            lower_all, upper_all = get_updated_bounds(
                net=net,
                lower_all=lower_all,
                upper_all=upper_all,
                alphas=alphas,
                betas=betas,
                history=history,
                cs=cs,
                threshold=rhs,
                domains=selected,
                splits=splits,
                debug=debug,
            )
        net.x = _make_bounded_tensor(net.x, selected)

    return (
        lower_all,
        upper_all,
        alphas,
        betas,
        intermediate_betas,
        selected,
        cs,
        rhs,
        history,
        selected,
        splits,
        [s.get_sample(domains.num_samples // 2) for s in selected],
    )


@torch.no_grad()
def get_updated_bounds(
    net: "LiRPAConvNet",
    lower_all: list[torch.Tensor],
    upper_all: list[torch.Tensor],
    alphas: dict[str, dict[str, torch.Tensor]],
    betas: list[list[torch.Tensor]] | None,
    history: list[list[tuple[torch.Tensor, torch.Tensor]]],
    cs: torch.Tensor,
    threshold: torch.Tensor,
    domains: list[Domain],
    splits: list[tuple[int, int]],
    debug: bool = IS_TEST_OR_DEBUG,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """
    Get the updated bounds after splitting a layer.

    Args:
        net: Alpha beta CROWN model.
        lower_all: Lower bounds for the pre-relu layers and the output.
        upper_all: Upper bounds for the pre-relu layers and the output.
        alphas: Alphas of the CROWN bounds.
        betas: Betas of the CROWN bounds.
        history: History of the split decisions.
        cs: Output transformation.
        threshold: Threshold of the split decision.
        domains: Samples from the domain.
        splits: Split decisions.
        debug: Run additional asserts.

    Returns:
        lower_all: Lower bounds for the pre-relu layers and the output.
        upper_all: Upper bounds for the pre-relu layers and the output.
    """
    nb = NewBounds()
    count = 0
    for batch, domain in enumerate(domains):
        if domain.stabilized:
            count -= 1
            for (layer, active), index in domain.stabilized.items():
                count += index.numel()
                if active:
                    nb.add_active(layer, batch, index)
                else:
                    nb.add_inactive(layer, batch, index)
        elif batch < len(splits):
            layer, index = splits[batch]
            nb.add_split(layer, batch, index, len(domains))
    if count < len(domains) + 2:
        # NOTE: Tightening bounds is mostly useful for branches with future splits.
        # This is a really cheap count that skips alot of leaf branches.
        # (Non-leaf branches will accumulate splits until this exit is bypassed.)
        return lower_all, upper_all
    for domain in domains:
        if domain.stabilized:
            domain.stabilized.clear()
    lower = torch.cat([d.lower_in for d in domains])  # type: ignore
    upper = torch.cat([d.upper_in for d in domains])  # type: ignore
    net.x = _make_bounded_tensor(net.x, domains, lower, upper)
    # NOTE: Running LiRPAConvNet.update_bounds_parallel first to make sure the state
    #   is correctly restored from the domain (slopes, betas, etc.).
    net.update_bounds_parallel(
        pre_lb_all=lower_all,
        pre_ub_all=upper_all,
        split=splits,
        slopes=alphas,
        betas=betas,
        history=history,
        samples=domains,
        fix_intermediate_layer_bounds=True,
        shortcut=True,
        cs=cs,
        decision_thresh=threshold,
        bound_lower=True,
        bound_upper=False,
    )
    bounds = {
        k.name: (l, u)
        for r, l, u in zip(net.net.relus, lower_all, upper_all)
        for k in r.inputs
    }
    bounds[net.net.input_name[0]] = (lower, upper)
    bounds = tighten_bounds(net.net, net.x, bounds, nb)
    del bounds[net.net.input_name[0]]
    for lb, ub, s in zip(lower, upper, domains):
        s.lower_in = lb[None]
        s.upper_in = ub[None]
        if debug:
            assert s.X is not None
            eps = torch.finfo(s.X.dtype).eps * 2
            assert (s.X >= s.lower_in - eps).all().cpu().item()
            assert (s.X <= s.upper_in + eps).all().cpu().item()
    lower_all = [l for (l, _) in bounds.values()] + [lower_all[-1]]
    upper_all = [u for (_, u) in bounds.values()] + [upper_all[-1]]
    if debug:
        assert all((ub >= lb).all().item() for lb, ub in zip(lower_all, upper_all))
        for i, (lbo, ubo) in enumerate(zip(lower_all[:-1], upper_all)):
            for j, s in enumerate(domains):
                assert_bounds(s.activations[i], lbo[None, j], ubo[None, j])  # type: ignore
    return lower_all, upper_all


def _make_bounded_tensor(
    x: "BoundedTensor",
    domains: list[Domain],
    lower: torch.Tensor | None = None,
    upper: torch.Tensor | None = None,
) -> "BoundedTensor":
    if lower is None:
        lower = torch.cat([s.lower_in for s in domains])  # type: ignore
    if upper is None:
        upper = torch.cat([s.upper_in for s in domains])  # type: ignore
    ptb = PerturbationLpNorm(x.ptb.eps, x.ptb.norm, lower, upper)
    if len(domains) > x.data.shape[0]:
        data = x.data[:1].expand(len(domains), *x.data.shape[1:])
    else:
        data = x.data[: len(domains)]
    return BoundedTensor(data, ptb)


@torch.no_grad()
def stabilize_on_samples(
    batch: list[Domain],
    domains: DomainList,
    add_empty: bool = False,
    *,
    debug: bool = IS_TEST_OR_DEBUG,
):
    """Stabilize unstable intermediate bounds if no sample crosses zero.
    This reduces the search space by adding constraints that avoid both impossible and rare subdomains.

    Args:
        batch: Batch of `Domain`:s.
        domains: Domain list.
        add_empty: Create domains for empty branches. Set to `True` when doing over-approximations.
        debug: Run additional asserts. Defaults to IS_TEST_OR_DEBUG.
    """
    for domain in batch:
        assert domain.activations is not None
        for layer, (act, lb, ub) in enumerate(
            zip(domain.activations, domain.lower_all, domain.upper_all)
        ):
            lbz = (act.min(0, True)[0] >= 0.0) & (lb < 0.0)
            ubz = (act.max(0, True)[0] <= 0.0) & (ub > 0.0)
            lbz = torch.nonzero(lbz.view(-1)).detach()
            ubz = torch.nonzero(ubz.view(-1)).detach()
            if lbz.numel() > 0:
                domain.stabilize(layer, lbz.view(-1), True)
                if add_empty:
                    _add_stabilized(domains, domain, layer, lbz.numel(), True)
            if ubz.numel() > 0:
                domain.stabilize(layer, ubz.view(-1), False)
                if add_empty:
                    _add_stabilized(domains, domain, layer, ubz.numel(), False)
        if debug:
            assert_history_contains(domain.activations, domain.history)
            assert_bounds(domain.activations, domain.lower_all, domain.upper_all)


def _add_stabilized(
    domains: DomainList,
    domain: Domain,
    layer: int,
    num_prev: int,
    active: bool,
):
    domain = copy(domain)
    domain.history = copy(domain.history)
    domain.log_volume = -np.inf
    domain.preimg_vol = domain.approx_vol = 0.0
    domain.priority = -np.inf
    domain.minimize_final()
    domain.stabilize(layer, 0, not active)
    his_act, his_ina = domain.history[layer]
    num_hist = len(his_act) if active else len(his_ina)
    for i in range(num_hist - num_prev, num_hist):
        other = copy(domain)
        other.history = copy(domain.history)
        if active:
            his_ina[-1] = his_act[i]
            other.history[layer] = (his_act[:i], his_ina.clone())
        else:
            his_act[-1] = his_ina[i]
            other.history[layer] = (his_act.clone(), his_ina[:i])
        domains.add(other)
