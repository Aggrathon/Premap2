import copy
from itertools import repeat

import numpy as np
import torch

from auto_LiRPA.bound_general import BoundedTensor, PerturbationLpNorm
from premap2.sampling import Samples
from premap2.tighten_bounds import NewBounds, tighten_bounds
from premap2.utils import IS_TEST_OR_DEBUG, assert_bounds, assert_contains_his

try:
    from premap.branching_domains import ReLUDomain, SortedReLUDomainList
    from premap.preimage_beta_crown_solver_relu_split import LiRPAConvNet
except ImportError:
    pass


def select_node_batch(
    samples: list[Samples],
    masks: list[torch.Tensor],
    lower: list[torch.Tensor],
    upper: list[torch.Tensor],
    cs: torch.Tensor,
    heuristics: list[tuple[str, float]] | None = None,
    **coefs,
) -> list[tuple[int, int]]:
    """Select which neuron to split for each domain in the batch.

    Args:
        samples: Samples per domain.
        masks: Batched mask of active neurons per layer.
        lower: Batched lower bound per layer.
        upper: Batched upper bound per layer.
        cs: Output specification as a linear tensor.
        heuristics: Key-value pairs of weights for the heuristics.
        **coefs: Alternative way of specifying weights for the heuristics.

    Returns:
        Tuple of (`layer`, `neuron`) for each domain.
    """
    if heuristics is not None:
        coefs = coefs | {key: value for key, value in heuristics}
    if coefs:
        coefs = {
            key if "_coef" in key else key + "_coef": value
            for key, value in coefs.items()
        }
    with torch.no_grad():
        for i, s in enumerate(samples):
            if s.priority is None:
                lbs = [lb[i] for lb in lower]
                ubs = [ub[i] for ub in upper]
                ms = [m[i].view(lb.shape[1:]) for m, lb in zip(masks, lower)]
                acts = [a[: s.num] for a in s.activations]
                out = torch.einsum("o...,n...->no", cs[i], s.y[: s.num])
                s.priority = calc_priority(
                    s.X[: s.num], out, acts, s.lAs, s.uAs, lbs, ubs, ms, **coefs
                )
            for p, m in zip(s.priority, masks):
                p.view(-1)[~m[i]] = -1.0
        decision = [
            max(
                (
                    (i, score.view(-1).argmax().cpu().item())
                    for i, score in enumerate(sample.priority)
                ),
                key=lambda k: sample.priority[k[0]].view(-1)[k[1]],
                default=(0, 0),
            )
            for sample in samples
        ]
    return decision


def calc_priority(
    X: torch.Tensor,
    yc: torch.Tensor,
    activations: list[torch.Tensor],
    lAs: list[torch.Tensor] | None,
    uAs: list[torch.Tensor] | None,
    lower: list[torch.Tensor],
    upper: list[torch.Tensor],
    unstable: list[torch.Tensor],
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
) -> list[torch.Tensor]:
    """Calculate priorities for the neurons.

    Args:
        X: Samples.
        yc: Post specification output.
        activations: Activation values for the samples.
        lAs: Lower linear bounds.
        uAs: Upper linear bounds.
        lower: Lower bounds
        upper: Upper bounds.
        unstable: Masks for unstable neurons.
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

    Returns:
        Neuron priorities per layer.
    """
    n, d = X.shape[0], X.shape[1:].numel()
    eps = torch.finfo(X.dtype).eps * 2
    priority = [torch.zeros_like(a[0]) for a in activations]
    if balance_coef > 0.0:  # Balance the split
        for pri, act in zip(priority, activations):
            pri += balance_coef * (
                1.0 - ((act >= 0).count_nonzero(0) * (2 / n) - 1.0).abs()
            )
    if soft_coef > 0.0:  # Soft balance the split
        for pri, act in zip(priority, activations):
            pri += soft_coef * (1.0 - (torch.sigmoid(act).mean(0) * 2.0 - 1.0).abs())
    if area_coef > 0.0:  # Area (lb<x<0) that will get constrained after a split
        if lAs is not None and uAs is not None:
            areas = [
                (lA.abs() + uA.abs()).sum(0) * lb**2 * u
                for lA, uA, lb, u in zip(lAs, uAs, lower, unstable)
            ]
        elif lAs is not None:
            areas = [
                lA.abs().sum(0) * lb**2 * u for lA, lb, u in zip(lAs, lower, unstable)
            ]
        elif uAs is not None:
            areas = [
                uA.abs().sum(0) * lb**2 * u for uA, lb, u in zip(uAs, lower, unstable)
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
        if lAs is not None and uAs is not None:
            for lA, uA, act, u in zip(lAs, uAs, activations, unstable):
                a = torch.relu(-act)
                d = (
                    torch.einsum("b...,n...->nb...", lA, a).abs()
                    + torch.einsum("b...,n...->nb...", uA, a).abs()
                ).sum((0, 1)) * u
                dists.append(d / (a.count_nonzero(0) + eps))
        elif lAs is not None:
            for lA, act, u in zip(lAs, activations, unstable):
                a = torch.relu(-act)
                d = torch.einsum("b...,n...->nb...", lA, a).abs().sum((0, 1)) * u
                dists.append(d / (a.count_nonzero(0) + eps))
        elif uAs is not None:
            for uA, act, u in zip(uAs, activations, unstable):
                a = torch.relu(-act)
                d = torch.einsum("b...,n...->nb...", uA, a).abs().sum((0, 1)) * u
                dists.append(d / (a.count_nonzero(0) + eps))
        dist_norm = extra_coef / (max((d.max() for d in dists)) + eps)
        for pri, d in zip(priority, dists):
            pri += d * dist_norm
    if under_coef > 0.0:  # Maximum distance from bound to zero when x<0
        if lAs is not None and uAs is not None:
            unders = [
                ((lA * lb[None]).abs().sum(0) + (uA * lb[None]).abs().sum(0)) * u
                for lA, uA, lb, u in zip(lAs, uAs, lower, unstable)
            ]
        elif lAs is not None:
            unders = [
                (lA * lb[None]).abs().sum(0) * u
                for lA, lb, u in zip(lAs, lower, unstable)
            ]
        elif uAs is not None:
            unders = [
                (uA * lb[None]).abs().sum(0) * u
                for uA, lb, u in zip(uAs, lower, unstable)
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
        preimg = (yc >= 0).all(1)
        for pri, act in zip(priority, activations):
            shape = (-1,) + tuple(1 for _ in act.shape[1:])
            snum = (act >= 0).count_nonzero(0)
            lp = (preimg.view(shape) & (act >= 0)).count_nonzero(0) / (snum + eps)
            rp = (preimg.view(shape) & (act < 0)).count_nonzero(0) / (n - snum + eps)
            pri += pure_coef * (
                torch.maximum((2.0 * lp - 1.0).abs(), (2.0 * rp - 1.0).abs())
            )
    return priority


def split_node_batch(
    net: "LiRPAConvNet",
    domains: "SortedReLUDomainList",
    under: bool,
    orig_lbs: list[torch.Tensor],
    orig_ubs: list[torch.Tensor],
    slopes: dict[str, dict[str, torch.Tensor]],
    betas: list[torch.Tensor | None],
    intermediate_betas: list[dict[str, torch.Tensor]] | list[None],
    selected_domains: list["ReLUDomain"],
    cs: torch.Tensor,
    rhs: torch.Tensor,
    history: list[list[tuple[list[int], list[float]]]],
    samples: list[Samples],
    branching_decision: list[tuple[int, int]],
    tighten: bool = True,
    debug: bool = False,
) -> tuple[object, ...]:
    """Split the domain on a neuron (batched).

    Any item in `selected_domains` that can be shortcutted is returned to `domains`.

    Args:
        net: LiRPA wrapped network.
        domains: List of domains.
        under: Under or over approximation.
        orig_lbs: Lower bounds.
        orig_ubs: Upper bounds.
        slopes: Alpha-CROWN slopes.
        betas: Beta-CROWN parameters.
        intermediate_betas: Betas for intermediate layers.
        selected_domains: Domains to split.
        cs: Output specification as a linear tensor.
        rhs: Right hand side of the output constraint.
        history: Split history for the `selected_domains`.
        samples: Samples.
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
    to_remove = []
    split_str = ""
    left_history = []
    right_history = []
    left_samples = []
    right_samples = []
    for i, (domain, sample, (layer, index)) in enumerate(
        zip(selected_domains, samples, branching_decision)
    ):
        # Check for fully explored branches (priority of branching decision < 0)
        if sample.priority[layer].view(-1)[index].cpu().item() < 0:
            domain.priority = -np.inf
            domains.add_domain(domain)
            to_remove.append(i)
        else:  # Check for shortcuts (domains where we don't need further processing after a split)
            s1, s2 = sample.split(layer, index)
            len1 = len(s1)
            len2 = len(s2)
            split_str += f"({layer}, {index}: {len1} | {len2}) "

            if len1 == 0:
                to_remove.append(i)
                d1, d2 = _split_domain(domain, s1, s2, layer, index, under, skip1=under)
                domains.add_domain(d2)
                if not under:  # Save domain not guaranteed to be outside preimage
                    d1.priority = -np.inf
                    domains.add_domain(d1)
                continue
            elif len2 == 0:
                to_remove.append(i)
                d1, d2 = _split_domain(domain, s1, s2, layer, index, under, skip2=under)
                domains.add_domain(d1)
                if not under:  # Save domain not guaranteed to be outside preimage
                    d2.priority = -np.inf
                    domains.add_domain(d2)
                continue

            preimg1 = torch.einsum("o...,n...->no", cs[i], s1.y) >= 0
            preimg1 = preimg1.all(-1).count_nonzero().item()
            preimg2 = torch.einsum("o...,n...->no", cs[i], s2.y) >= 0
            preimg2 = preimg2.all(-1).count_nonzero().item()
            A, b = domain.preimg_A, domain.preimg_b
            approx1 = torch.einsum("o...,n...->no", A, s1.X) + b.view(1, -1) >= 0
            approx1 = approx1.all(-1).count_nonzero().item()
            approx2 = torch.einsum("o...,n...->no", A, s2.X) + b.view(1, -1) >= 0
            approx2 = approx2.all(-1).count_nonzero().item()

            if preimg1 == 0 if under else preimg1 == len1:
                to_remove.append(i)
                d1, d2 = _split_domain(domain, s1, s2, layer, index, under, skip1=under)
                d2.preimg_vol = preimg2 / len2 * d2.volume
                d2.approx_vol = approx2 / len2 * d2.volume
                d2.priority = abs(d2.preimg_vol - d2.approx_vol)
                domains.add_domain(d2)
                if not under:  # Save domain with full preimage
                    d1.preimg_vol = d1.approx_vol = d1.volume
                    d1.priority = -np.inf
                    domains.add_domain(d1)
                continue
            elif preimg2 == 0 if under else preimg2 == len2:
                to_remove.append(i)
                d1, d2 = _split_domain(domain, s1, s2, layer, index, under, skip2=under)
                d1.preimg_vol = preimg1 / len1 * d1.volume
                d1.approx_vol = approx1 / len1 * d1.volume
                d1.priority = abs(d1.preimg_vol - d1.approx_vol)
                domains.add_domain(d1)
                if not under:  # Save domain with full preimage
                    d2.preimg_vol = d2.approx_vol = d2.volume
                    d2.priority = -np.inf
                    domains.add_domain(d2)
                continue

            if approx1 == preimg1 if under else approx1 == 0:
                to_remove.append(i)
                d1, d2 = _split_domain(domain, s1, s2, layer, index, under)
                d1.approx_vol = d1.preimg_vol = preimg1 / len1 * d1.volume
                d1.priority = -np.inf
                domains.add_domain(d1)
                d2.approx_vol = approx2 / len2 * d2.volume
                d2.preimg_vol = preimg2 / len2 * d2.volume
                d2.priority = abs(d2.preimg_vol - d2.approx_vol)
                domains.add_domain(d2)
            elif approx2 == preimg2 if under else approx2 == 0:
                to_remove.append(i)
                d1, d2 = _split_domain(domain, s1, s2, layer, index, under)
                d2.approx_vol = d2.preimg_vol = preimg2 / len2 * d2.volume
                d2.priority = -np.inf
                domains.add_domain(d2)
                d1.approx_vol = approx1 / len1 * d1.volume
                d1.preimg_vol = preimg1 / len1 * d1.volume
                d1.priority = abs(d1.preimg_vol - d1.approx_vol)
                domains.add_domain(d1)
            else:
                h1, h2 = _split_history(domain.history, layer, index)
                left_history.append(h1)
                left_samples.append(s1)
                right_history.append(h2)
                right_samples.append(s2)

    if len(split_str) > 101:
        split_str = split_str[:97] + "..."
    print("Splits decision:", split_str[:100])

    for index in reversed(to_remove):
        domain = selected_domains.pop(index)
        for lst in (samples, history, branching_decision, betas, intermediate_betas):
            lst.pop(index)
        for lst in (orig_lbs, orig_ubs):
            for i, v in enumerate(lst):
                lst[i] = torch.cat((v[:index], v[index + 1 :]))
        cs = torch.cat((cs[:index], cs[index + 1 :]))
        rhs = torch.cat((rhs[:index], rhs[index + 1 :]))
        for k1, v1 in slopes.items():
            for k2, v2 in v1.items():
                slopes[k1][k2] = torch.cat((v2[:, :, :index], v2[:, :, index + 1 :]), 2)

    samples = left_samples + right_samples
    if debug:
        assert all(len(s) > 0 for s in samples)
        for hist, sample in zip(left_history + right_history, samples):
            assert_contains_his(sample.activations, hist)

    # Tighten the bounds for the splits
    if len(samples) > 0:
        with torch.no_grad():
            orig_lbs, orig_ubs = get_updated_bounds(
                net=net,
                lower_all=orig_lbs,
                upper_all=orig_ubs,
                alphas=slopes,
                betas=betas,
                history=history,
                cs=cs,
                threshold=rhs,
                samples=samples,
                splits=branching_decision,
                split_only=not tighten,
                debug=debug,
            )
        net.x = _make_bounded_tensor(net.x, samples)

    return (
        orig_lbs,
        orig_ubs,
        slopes,
        betas,
        intermediate_betas,
        selected_domains,
        cs,
        rhs,
        history,
        left_history + right_history,
        samples,
        branching_decision,
    )


def _split_history(
    history: list[tuple[list[int], list[float]]], layer: int, index: int
) -> tuple[list[tuple[list[int], list[float]]], list[tuple[list[int], list[float]]]]:
    history1 = copy.copy(history)
    history2 = copy.copy(history)
    hist_ind, hist_sign = history[layer]
    history1[layer] = (hist_ind + [index], hist_sign + [+1.0])
    history2[layer] = (hist_ind + [index], hist_sign + [-1.0])
    return history1, history2


@torch.no_grad()
def _split_domain(
    domain: "ReLUDomain",
    samples1: Samples,
    samples2: Samples,
    layer: int,
    index: int,
    under: bool,
    skip1: bool = False,
    skip2: bool = False,
) -> tuple["ReLUDomain", "ReLUDomain"]:
    if skip1:
        domain_above = None
        domain_below = domain
    elif skip2:
        domain_above = domain
        domain_below = None
    else:
        domain_above = copy.copy(domain)
        domain_below = domain
    num = len(samples1) + len(samples2)
    for domain, sign, samples in (
        (domain_above, +1.0, samples1),
        (domain_below, -1.0, samples2),
    ):
        if domain is None:
            continue
        domain.valid = True
        domain.history = copy.copy(domain.history)
        hist_ind, hist_sign = domain.history[layer]
        domain.history[layer] = (hist_ind + [index], hist_sign + [sign])
        domain.samples = samples
        if len(domain.samples) != num:
            scale = len(domain.samples) / num
            domain.volume *= scale
            domain.preimg_vol *= scale
            domain.approx_vol = 0.0 if under else domain.volume
        domain.samples.stabilize(layer, index, sign > 0)
        if sign > 0:
            domain.lower_all = copy.copy(domain.lower_all)
            domain.lower_all[layer] = domain.lower_all[layer].clone()
            domain.lower_all[layer].view(-1)[index] = 0.0
        else:
            domain.upper_all = copy.copy(domain.upper_all)
            domain.upper_all[layer] = domain.upper_all[layer].clone()
            domain.upper_all[layer].view(-1)[index] = 0.0
    return domain_above, domain_below


def get_updated_bounds(
    net: "LiRPAConvNet",
    lower_all: list[torch.Tensor],
    upper_all: list[torch.Tensor],
    alphas: dict[str, dict[str, torch.Tensor]],
    betas: list[torch.Tensor | None],
    history: list[list[tuple[list[int], list[float]]]],
    cs: torch.Tensor,
    threshold: torch.Tensor,
    samples: list[Samples],
    splits: list[tuple[int, int]],
    split_only: bool = False,
    debug: bool = False,
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
        samples: Samples from the domain.
        splits: Split decisions.
        split_only: Return after splitting without updating any other bounds.
        debug: Run additional asserts.

    Returns:
        lower_all: Lower bounds for the pre-relu layers and the output.
        upper_all: Upper bounds for the pre-relu layers and the output.
    """
    for i in range(len(lower_all)):
        lower_all[i] = torch.cat((lower_all[i], lower_all[i]))
        upper_all[i] = torch.cat((upper_all[i], upper_all[i]))
    for j, (l, i) in enumerate(splits):
        lower_all[l].view(len(samples), -1)[j, i] = 0.0
        upper_all[l].view(len(samples), -1)[j + len(splits), i] = 0.0
    if split_only:
        return lower_all, upper_all
    nb = NewBounds()
    for batch, (layer, index) in enumerate(splits):
        nb.add_split(layer, batch, index, len(samples))
    count = 0
    for batch, sample in enumerate(samples):
        for (layer, active), index in sample.stabilized.items():
            count += index.numel()
            if active:
                nb.add_active(layer, batch, index)
            else:
                nb.add_inactive(layer, batch, index)
    if count < len(samples) + 2:
        # NOTE: Tightening bounds is mostly useful for branches with future splits.
        # This is a really cheap count that skips alot of leaf branches.
        # (Non-leaf branches will accumulate splits until this exit is bypassed.)
        return lower_all, upper_all
    for sample in samples:
        sample.stabilized.clear()
    lower = torch.cat([s.lower for s in samples])
    upper = torch.cat([s.upper for s in samples])
    net.x = _make_bounded_tensor(net.x, samples, lower, upper)
    # NOTE: Running LiRPAConvNet.update_bounds_parallel first to make sure the state
    #   is correctly restored from the domain (slopes, betas, etc.).
    net.update_bounds_parallel(
        pre_lb_all=lower_all,
        pre_ub_all=upper_all,
        split=splits,
        slopes=alphas,
        betas=betas,
        history=history,
        samples=samples,
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
    for lb, ub, s in zip(lower, upper, samples):
        s.lower = lb[None]
        s.upper = ub[None]
        if debug:
            eps = torch.finfo(s.X.dtype).eps * 2
            assert (s.X >= s.lower - eps).all().cpu().item()
            assert (s.X <= s.upper + eps).all().cpu().item()
    lower_all = [l for (l, _) in bounds.values()] + [lower_all[-1]]
    upper_all = [u for (_, u) in bounds.values()] + [upper_all[-1]]
    if debug:
        assert all(
            (ub >= lb).all().cpu().item() for lb, ub in zip(lower_all, upper_all)
        )
        for i, (lbo, ubo) in enumerate(zip(lower_all[:-1], upper_all)):
            for j, s in enumerate(samples):
                assert_bounds(s.activations[i], lbo[None, j], ubo[None, j])
    return lower_all, upper_all


def _make_bounded_tensor(
    x: BoundedTensor,
    samples: list[Samples],
    lower: torch.Tensor | None = None,
    upper: torch.Tensor | None = None,
) -> BoundedTensor:
    if lower is None:
        lower = torch.cat([s.lower for s in samples])
    if upper is None:
        upper = torch.cat([s.upper for s in samples])
    ptb = PerturbationLpNorm(x.ptb.eps, x.ptb.norm, lower, upper)
    if len(samples) > x.data.shape[0]:
        data = x.data[:1].expand(len(samples), *x.data.shape[1:])
    else:
        data = x.data[: len(samples)]
    return BoundedTensor(data, ptb)


@torch.no_grad()
def stabilize_on_samples(
    samples: list[Samples],
    history: list[list[tuple[list[int], list[float]]]],
    lower: list[torch.Tensor],
    upper: list[torch.Tensor],
    domains: list["ReLUDomain"] | None = None,
    domain_list: "SortedReLUDomainList | None" = None,
    store_splits: bool = False,
    readd_splits: bool = False,
    debug: bool = IS_TEST_OR_DEBUG,
):
    """Stabilize unstable intermediate bounds if no sample crosses zero.
    This reduces the search space by adding constraints that avoid both impossible and rare subdomains.

    Args:
        samples: Batch of `Samples`.
        history: History that will be modified in-place.
        lower: Lower bounds.
        upper: Upper bounds.
        domains: Batch of `ReLUDomain`.
        domain_list: Domains not in the batch.
        store_splits: Store the stabilized bounds in `Samples.stabilized`.
        readd_splits: Create domains for empty branches. Set to `True` when doing over-approximations.
        debug: Run additional asserts. Defaults to IS_TEST_OR_DEBUG.
    """
    for i, lb in enumerate(lower):
        lower[i] = lb.contiguous()
    for i, ub in enumerate(upper):
        upper[i] = ub.contiguous()
    assert not readd_splits or (domains is not None and domain_list is not None)
    for i, (s, d, his) in enumerate(
        zip(samples, domains if domains else repeat(None), history)
    ):
        assert (acts := s.activations) is not None
        for layer, (act, (ind, sgn), lb, ub) in enumerate(zip(acts, his, lower, upper)):
            lbz = (act.min(0, True)[0] >= 0.0) & (lb[i] < 0.0)
            ubz = (act.max(0, True)[0] <= 0.0) & (ub[i] > 0.0)
            lbz = torch.nonzero(lbz.view(-1)).detach()
            ubz = torch.nonzero(ubz.view(-1)).detach()
            if lbz.numel() == 0 and ubz.numel() == 0:
                continue
            if lbz.numel() > 0:
                lb[i].view(-1)[lbz] = 0.0
                sgn = sgn + [1.0] * lbz.numel()
                ind = ind + lbz.view(-1).tolist()
                if store_splits:
                    s.stabilize(layer, lbz.view(-1).cpu(), True)
                if readd_splits:
                    _add_stabilized(domain_list, d, layer, lbz, True)
            if ubz.numel() > 0:
                ub[i].view(-1)[ubz] = 0.0
                sgn = sgn + [-1.0] * ubz.numel()
                ind = ind + ubz.view(-1).tolist()
                if store_splits:
                    s.stabilize(layer, ubz.view(-1).cpu(), False)
                if readd_splits:
                    _add_stabilized(domain_list, d, layer, ubz, False)
            history[i][layer] = (ind, sgn)
        if debug:
            assert_contains_his(
                acts, his, (l[None, i] for l in lower), (u[None, i] for u in upper)
            )
        # Update domain in case of later shortcut
        if d is not None:
            d.history = history[i]
            d.samples = s
            d.lower_all = [lb[i, None] for lb in lower]
            d.upper_all = [ub[i, None] for ub in upper]


def _add_stabilized(
    domains: "SortedReLUDomainList",
    domain: "ReLUDomain",
    layer: int,
    index: torch.LongTensor,
    above: bool,
):
    domain.history = copy.copy(domain.history)
    his_ind, his_sgn = (copy.copy(l) for l in domain.history[layer])
    domain.history[layer] = (his_ind, his_sgn)
    for i in index.ravel().detach().cpu().numpy():
        domain2 = copy.copy(domain)
        domain2.history = copy.copy(domain.history)
        domain2.history[layer] = (his_ind + [i], his_sgn + [-1.0 if above else +1.0])
        domain2.volume = 0.0
        domain2.preimg_vol = 0.0
        domain2.approx_vol = 0.0
        domain2.priority = -np.inf
        domains.add_domain(domain2)
        his_ind.append(i)
        his_sgn.append(+1.0 if above else -1.0)
