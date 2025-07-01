import copy

import numpy as np
import torch

from auto_LiRPA.bound_general import BoundedTensor, PerturbationLpNorm
from premap2.sampling import Samples
from premap2.tighten_bounds import tighten_bounds

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
    width_coef: float = 0.0,
    loose_coef: float = 0.0,
    bound_coef: float = 0.0,
    gap_coef: float = 0.0,
    area_coef: float = 1.0,
    under_coef: float = 0.75,
    extra_coef: float = 1.0,
    stable_coef: float = 0.0,
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

    def remove_domain(index: int, add: bool = True):
        # Shortcut branches that don't need exploring
        nonlocal cs, rhs
        batch = len(samples)
        domain = selected_domains.pop(index)
        if add:
            domain.samples = samples[index]
            domains.add_domain(domain)
        for lst in (samples, history, branching_decision, betas, intermediate_betas):
            lst.pop(index)
        for lst in (orig_lbs, orig_ubs):
            for i, v in enumerate(lst):
                if len(v) == batch * 2:
                    v = (
                        v[:index],
                        v[index + 1 : batch + index],
                        v[batch + index + 1 :],
                    )
                    lst[i] = torch.cat(v)
                else:
                    lst[i] = torch.cat((v[:index], v[index + 1 :]))
        cs = torch.cat((cs[:index], cs[index + 1 :]))
        rhs = torch.cat((rhs[:index], rhs[index + 1 :]))
        for k1, v1 in slopes.items():
            for k2, v2 in v1.items():
                slopes[k1][k2] = torch.cat((v2[:, :, :index], v2[:, :, index + 1 :]), 2)

    b = 0
    # Check for fully explored branches (priority of branching decision < 0)
    while b < len(branching_decision):
        layer, index = branching_decision[b]
        if samples[b].priority[layer].view(-1)[index].cpu().item() < 0:
            selected_domains[b].priority = -np.inf
            remove_domain(b)
        else:
            b += 1

    samples = [s.split(lay, ind) for s, (lay, ind) in zip(samples, branching_decision)]
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
                samples=[s[i] for i in range(2) for s in samples],
                splits=branching_decision,
                split_only=not tighten,
                debug=debug,
            )

        split_str = " ".join(str(b) for b in branching_decision)
        if len(split_str) > 100:
            split_str = split_str[:97] + "..."
        print("Splits decision:", split_str)
    if debug:
        assert all((ub >= lb).all().cpu().item() for lb, ub in zip(orig_lbs, orig_ubs))

    # Check and shortcut one-sided splits
    def readd_domain(b: int, above: bool = True, final: bool = False):
        s1, s2 = samples[b]
        domain = selected_domains[b]
        layer, index = branching_decision[b]
        if final:
            domain = copy.copy(domain)
            domain.priority = -np.nan
        domain.valid = True
        domain.history = copy.copy(domain.history)
        hist_ind, hist_sign = domain.history[layer]
        if above:
            domain.samples = s1
            domain.history[layer] = (hist_ind + [index], hist_sign + [+1.0])
            domain.lower_all = [lb[None, b] for lb in orig_lbs]
            domain.upper_all = [ub[None, b] for ub in orig_ubs]
            domain.volume = domain.volume * (len(s1) / (len(s1) + len(s2)))
            domains.add_domain(domain)
        else:
            domain.samples = s2
            domain.history[layer] = (hist_ind + [index], hist_sign + [-1.0])
            domain.lower_all = [lb[None, b + len(samples)] for lb in orig_lbs]
            domain.upper_all = [ub[None, b + len(samples)] for ub in orig_ubs]
            domain.volume = domain.volume * (len(s2) / (len(s1) + len(s2)))
            domains.add_domain(domain)

    b = 0
    split_str = ""
    pruned = False
    while b < len(samples):
        s1, s2 = samples[b]
        len1 = len(s1)
        len2 = len(s2)
        split_str += f"({len1} | {len2}) "
        remove1 = len1 == 0
        remove2 = len2 == 0
        # NOTE: An empty mask means either an invalid domain or extremely tiny volume.
        # In case of invalid domain, pruning is the correct course of action.
        # In case of tiny volume, ideally we should try more samples.
        # However, that might be difficult and might not change the result in any meaningful way.
        if under and not remove1 and not remove2:
            preimg1 = (torch.einsum("o...,n...->no", cs[b], s1.y) >= 0).all(-1)
            preimg1 = preimg1.count_nonzero().cpu().item()
            preimg2 = (torch.einsum("o...,n...->no", cs[b], s2.y) >= 0).all(-1)
            preimg2 = preimg2.count_nonzero().cpu().item()
            remove1 = preimg1 == 0
            remove2 = preimg2 == 0
        if remove1 or remove2:
            if not remove1:
                readd_domain(b, True, False)
            if not remove2:
                readd_domain(b, False, False)
            remove_domain(b, False)
            pruned = True
        else:
            b += 1
    if len(split_str) > 101:
        print("Splits preimage:", split_str[:97] + "...")
    elif len(split_str) > 0:
        print("Splits preimage:", split_str[:100])
    if pruned:
        print("Shortcut subdomains without preimage samples.")
    samples = [s[i] for i in range(2) for s in samples]
    if debug:
        assert all(len(s) > 0 for s in samples)

    # Create histories for the splits
    left_history = []
    right_history = []
    for hist, (layer, index) in zip(history, branching_decision):
        left = copy.copy(hist)
        left[layer] = (left[layer][0] + [index], left[layer][1] + [+1.0])
        left_history.append(left)
        right = copy.copy(hist)
        right[layer] = (right[layer][0] + [index], right[layer][1] + [-1.0])
        right_history.append(right)

    if debug:
        for hist, sample in zip(left_history + right_history, samples):
            for (i, s), a in zip(hist, sample.activations):
                if i:
                    a = a.flatten(1)[:, i] >= 0
                    s = torch.tensor([s], device=a.device) >= 0
                    assert (a == s).all().cpu().item()
    if len(samples) > 0:
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
        domain: The domain to split.
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
    split_dict = {
        "decision": [[bd] for bd in splits],
        "coeffs": [[1.0] for i in range(len(splits))],
    }
    lower = torch.cat([s.lower for s in samples])
    upper = torch.cat([s.upper for s in samples])
    net.x = _make_bounded_tensor(net.x, samples, lower, upper)
    # NOTE: Running LiRPAConvNet.update_bounds_parallel first to make sure the state
    #   is correctly restored from the domain (slopes, betas, etc.).
    net.update_bounds_parallel(
        pre_lb_all=lower_all,
        pre_ub_all=upper_all,
        split=split_dict,
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
    splits2 = [(i, [], [], [], []) for i in range(len(net.net.relus))]
    for batch, (layer, index) in enumerate(splits):
        splits2[layer][1].append(batch)
        splits2[layer][2].append(index)
        splits2[layer][3].append(batch + len(splits))
        splits2[layer][4].append(index)
    splits2 = [s for s in splits2 if len(s[1]) + len(s[3]) > 0]
    bounds = tighten_bounds(net.net, net.x, bounds, splits2)
    del bounds[net.net.input_name[0]]
    for lb, ub, s in zip(lower, upper, samples):
        s.lower = lb[None]
        s.upper = ub[None]
        if debug:
            assert (s.X >= s.lower).all().cpu().item()
            assert (s.X <= s.upper).all().cpu().item()
    lower_all = [l for (l, _) in bounds.values()] + [lower_all[-1]]
    upper_all = [u for (_, u) in bounds.values()] + [upper_all[-1]]
    # if debug:
    eps = torch.finfo(lower_all[0].dtype).eps * 3
    for i, (lbo, ubo) in enumerate(zip(lower_all[:-1], upper_all)):
        for j, s in enumerate(samples):
            if s.activations[i].shape[0] > 0:
                # These asserts might fail due to `tighten_bounds_forward`, which is just a wrapper around `lirpa.compute_bounds`.
                # (The asserts succeed if inserted just before and only disabling `tighten_bounds_forward` avoids this issue).
                # I don't understand how LiRPA could fail (only observed with the CNN).
                # assert (s.activations[i] >= lbo[None, j] - eps).all().cpu().item()
                # assert (s.activations[i] <= ubo[None, j] + eps).all().cpu().item()
                # These where:s should in theory not do anything, see above.
                amax = s.activations[i].max(0)[0]
                ubo[j] = torch.where(amax > ubo[j], amax + eps, ubo[j])
                amin = s.activations[i].min(0)[0]
                lbo[j] = torch.where(amin < lbo[j], amin - eps, lbo[j])

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
