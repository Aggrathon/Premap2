import torch

from auto_LiRPA import BoundedModule, BoundedTensor
from premap2.utils import IS_TEST_OR_DEBUG, is_int


def tighten_bounds(
    lirpa: BoundedModule,
    x: BoundedTensor,
    bounds: dict[str, tuple[torch.Tensor, torch.Tensor]],
    split: None
    | tuple[int, int]
    | list[tuple[int, list[int], list[int], list[int], list[int]]] = None,
    debug: bool = IS_TEST_OR_DEBUG,
    **kwargs,
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """Calculate tighter bounds afer a split (batched).

    Args:
        lirpa: LiRPA wrapped network.
        x: Input specification.
        bounds: Lower and upper bounds for each layer.
        split: Just performed splits.
        debug: Activate additional asserts. Defaults to False unless debugging.

    Returns:
        Tightened `bounds`.
    """
    if split is not None:
        if len(split) == 2 and is_int(split[0]) and is_int(split[1]):
            layer, index = split
            layer = min(layer, len(lirpa.relus) + layer)
            lower = torch.arange(x.shape[0] // 2)
            upper = torch.arange(x.shape[0] // 2, x.shape[0])
            split = [(layer, lower, index, upper, index)]
        for layer, lb, li, ub, ui in sorted(split):
            tighten_bounds_backward(
                lirpa, x, bounds, layer, lb, li, ub, ui, debug=debug, **kwargs
            )
    return tighten_bounds_forward(lirpa, x, bounds, **kwargs, debug=debug)


def tighten_bounds_forward(
    lirpa: BoundedModule,
    x: BoundedTensor,
    bounds: dict[str, tuple[torch.Tensor, torch.Tensor]],
    debug: bool = IS_TEST_OR_DEBUG,
    **kwargs,
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """Use LiRPA to tighten the bounds of following layers (batched).

    Args:
        lirpa: LiRPA wrapped network.
        x: Input specification.
        bounds: Lower and upper bounds for each layer.
        debug: Activate additional asserts. Defaults to False unless debugging.

    Returns:
        Tightened `bounds`.
    """
    lirpa.compute_bounds(
        x=(x,),
        final_node_name=lirpa.relus[-1].inputs[0].name,
        **kwargs,
        bound_upper=True,
        bound_lower=True,
        reference_bounds=bounds,
    )
    for key, (lbo, ubo) in bounds.items():
        if getattr(lirpa[key], "lower", None) is not None:
            if debug:
                eps = torch.finfo(ubo.dtype).eps * 2
                assert (lirpa[key].lower <= ubo + eps).all().cpu().item()
                assert (lirpa[key].upper >= lbo - eps).all().cpu().item()
            lbo[:] = torch.clamp(lirpa[key].lower.detach(), lbo, ubo)
            ubo[:] = torch.clamp(lirpa[key].upper.detach(), lbo, ubo)
    return bounds


def tighten_bounds_backward(
    lirpa: BoundedModule,
    x: BoundedTensor,
    bounds: dict[str, tuple[torch.Tensor, torch.Tensor]],
    layer: int,
    lower_b: list[int],
    lower_i: list[int],
    upper_b: list[int],
    upper_i: list[int],
    debug: bool = IS_TEST_OR_DEBUG,
    **kwargs,
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """Tighten the bounds of previous layers (batched).
    This function runs LiRPA to get linear bounds.

    Args:
        lirpa: LiRPA wrapped network.
        x: Input specification.
        bounds: Lower and upper bounds per layer.
        layer: Layer to tighten bounds for.
        lower_b: Batch indices of lower bound updates.
        lower_i: Indices of lower bound updates.
        upper_b: Batch indices of upper bound updates.
        upper_i: Indices of upper bound updates.
        debug: Activate additional asserts. Defaults to False unless debugging.

    Returns:
        Tightened `bounds`.
    """
    final = lirpa.relus[layer].inputs[0].name
    prev = [r.name for r in lirpa.relus[:layer]]
    if lirpa.input_name[0] in bounds:
        prev.append(lirpa.input_name[0])
    if len(prev) == 0:
        return bounds
    c = x.new_zeros(x.shape[0], *lirpa.relus[layer].output_shape[1:])
    c.view(c.shape[0], -1)[lower_b, lower_i] = 1.0
    c.view(c.shape[0], -1)[upper_b, upper_i] = 1.0
    _, _, A2 = lirpa.compute_bounds(
        x=(x,),
        C=c[:, None],
        final_node_name=final,
        **kwargs,
        bound_upper=True,
        bound_lower=True,
        return_A=True,
        need_A_only=True,
        needed_A_dict={final: prev},
        intermediate_layer_bounds=bounds,
    )

    lb_curr, ub_curr = bounds[final]
    for p in prev:
        if p in bounds:
            lb_prev, ub_prev = bounds[p]
        else:
            lb_prev, ub_prev = bounds[lirpa[p].inputs[0].name]
        tighten_bounds_back(
            A2[final][p]["lA"],
            A2[final][p]["lbias"],
            A2[final][p]["uA"],
            A2[final][p]["ubias"],
            lb_prev,
            ub_prev,
            lb_curr,
            ub_curr,
            lower_b,
            lower_i,
            upper_b,
            upper_i,
            debug=debug,
        )
    return bounds


def tighten_bounds_back(
    lA: torch.Tensor,
    lbias: torch.Tensor,
    uA: torch.Tensor,
    ubias: torch.Tensor,
    lb_prev: torch.Tensor,
    ub_prev: torch.Tensor,
    lb_curr: torch.Tensor,
    ub_curr: torch.Tensor,
    lower_b: list[int],
    lower_i: list[int],
    upper_b: list[int],
    upper_i: list[int],
    debug: bool = IS_TEST_OR_DEBUG,
):
    """Tighten the bounds of a previous layer (batched).

    Args:
        lA: Linear lower bound coefficients.
        lbias: Linear lower bound bias.
        uA: Linear upper bound coefficients.
        ubias: Linear upper bound bias.
        lb_prev: Previous layer's lower bounds.
        ub_prev: Previous layer's upper bounds.
        lb_curr: Current layer's lower bounds.
        ub_curr: Current layer's upper bounds.
        lower_b: Batch indices of lower bound updates.
        lower_i: Indices of lower bound updates.
        upper_b: Batch indices of upper bound updates.
        upper_i: Indices of upper bound updates.
        debug: Activate additional asserts. Defaults to False unless debugging.
    """
    eps = torch.finfo(lA.dtype).eps * 3
    center = (ub_prev + lb_prev) / 2.0
    diff = (ub_prev - lb_prev) / 2.0
    flat1 = (ub_curr.shape[0], -1)
    flat2 = (-1,) + tuple([1] * len(lb_prev.shape[1:]))
    lb_curr = lb_curr.reshape(flat1)
    ub_curr = ub_curr.reshape(flat1)
    lbias = lbias.reshape(flat1)
    ubias = ubias.reshape(flat1)
    uA = uA.reshape(*flat1, *lb_prev.shape[1:])
    lA = lA.reshape(*flat1, *lb_prev.shape[1:])
    # To propagate bounds backward we use a worst case scenario mindset.
    # The idea for one variable is as follows:
    #   1. Set all other variables, x_j, to the value that causes the maximum/minimum value.
    #   2. Calculate the potential new bounds value of the variable x_i:
    #       uA_i * x_i + ubias + c >= l => xmin_i = (l - ubias - c) / uA_i | uA_i > 0, c = max(sum(uA_j * x_j | j != i))
    #       uA_i * x_i + ubias + c >= l => xmax_i = (l - ubias - c) / uA_i | uA_i < 0, c = max(sum(uA_j * x_j | j != i))
    #       lA_i * x_i + lbias + c <= u => xmax_i = (u - lbias - c) / lA_i | lA_i > 0, c = min(sum(lA_j * x_j | j != i))
    #       lA_i * x_i + lbias + c <= u => xmin_i = (u - lbias - c) / lA_i | lA_i < 0, c = min(sum(lA_j * x_j | j != i))
    # Note the worst case assumption 'uA @ x + ubias >= l' and 'lA @ x + lbias <= u'.
    # We calculate the bounds individually for each variable, but in parallel using vectorization.

    if len(lower_b) > 0:
        A = uA[lower_b, lower_i] if uA.shape[1] > 1 else uA[lower_b, 0]
        bias = ubias[lower_b, lower_i] if ubias.shape[1] > 1 else ubias[lower_b, 0]
        ci = A * center[lower_b] + A.abs() * diff[lower_b]
        c = ci.flatten(1).sum(1)
        ln = ((lb_curr[lower_b, lower_i] - bias - c).reshape(flat2) + ci) / A
        lb_new = torch.where((A > eps), ln, lb_prev[lower_b])
        ub_new = torch.where((A < -eps), ln, ub_prev[lower_b])
        if debug:
            assert (lb_new <= ub_prev[lower_b] + eps).all().cpu().item()
            assert (ub_new >= lb_prev[lower_b] - eps).all().cpu().item()
        lb_prev[lower_b] = torch.maximum(lb_new, lb_prev[lower_b])
        ub_prev[lower_b] = torch.minimum(ub_new, ub_prev[lower_b])

    if len(upper_b) > 0:
        A = lA[upper_b, upper_i] if lA.shape[1] > 1 else lA[upper_b, 0]
        bias = lbias[upper_b, upper_i] if lbias.shape[1] > 1 else lbias[upper_b, 0]
        ci = A * center[upper_b] - A.abs() * diff[upper_b]
        c = ci.flatten(1).sum(1)
        un = ((ub_curr[upper_b, upper_i] - bias - c).reshape(flat2) + ci) / A
        lb_new = torch.where((A < -eps), un, lb_prev[upper_b])
        ub_new = torch.where((A > eps), un, ub_prev[upper_b])
        if debug:
            assert (lb_new <= ub_prev[upper_b] + eps).all().cpu().item()
            assert (ub_new >= lb_prev[upper_b] - eps).all().cpu().item()
        lb_prev[upper_b] = torch.maximum(lb_new, lb_prev[upper_b])
        ub_prev[upper_b] = torch.minimum(ub_new, ub_prev[upper_b])


def tighten_backwards(
    A: torch.Tensor,
    bias: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    debug: bool = IS_TEST_OR_DEBUG,
):
    """Thigten bounds of a previous layer (batched).

    Args:
        A: Linear bound coefficients.
        bias: Linear bound biases.
        lower: Lower bounds of the previous layer.
        upper: Upper bounds of the previous layer.
        debug: Activate additional asserts. Defaults to False unless debugging.
    """
    eps = torch.finfo(lower.dtype).eps * 3
    center = (upper + lower) / 2.0
    diff = (upper - lower) / 2.0
    shape = (-1,) + tuple([1] * len(lower.shape[1:]))

    A = A.reshape(-1, *lower.shape[1:])
    bias = bias.flatten()
    ci = A * center + A.abs() * diff
    ln = (ci - (bias + ci.flatten(1).sum(1)).reshape(shape)) / A
    lb_new = torch.where((A > eps), ln, lower).max(0)[0]
    ub_new = torch.where((A < -eps), ln, upper).min(0)[0]
    if debug:
        assert (lb_new <= upper + eps).all().cpu().item()
        assert (ub_new >= lower - eps).all().cpu().item()
    lower[:] = torch.maximum(lb_new, lower)
    upper[:] = torch.minimum(ub_new, upper)
