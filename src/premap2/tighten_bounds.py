from collections import defaultdict
from typing import Any, Iterable

import torch

from premap2.utils import IS_TEST_OR_DEBUG

try:
    from auto_LiRPA import BoundedModule, BoundedTensor
except ImportError:
    pass


class NewBounds:
    def __init__(self, *, active: None | Any = None, inactive: None | Any = None):
        self.spec = defaultdict(
            lambda: defaultdict(lambda: (torch.LongTensor(), torch.LongTensor()))
        )
        if active is not None:
            self.add_active(*active)
        if inactive is not None:
            self.add_inactive(*inactive)

    @torch.no_grad()
    def add_active(self, layer: int, batch: int, index: int | torch.LongTensor):
        if isinstance(index, torch.Tensor):
            index = torch.atleast_1d(index)
        else:
            index = torch.LongTensor([index])
        if index.numel() > 0:
            act, ina = self.spec[layer][batch]
            self.spec[layer][batch] = (torch.cat((act, index)), ina)

    @torch.no_grad()
    def add_inactive(self, layer: int, batch: int, index: int | torch.LongTensor):
        if isinstance(index, torch.Tensor):
            index = torch.atleast_1d(index)
        else:
            index = torch.LongTensor([index])
        if index.numel() > 0:
            act, ina = self.spec[layer][batch]
            self.spec[layer][batch] = (act, torch.cat((ina, index)))

    def add_split(
        self, layer: int, batch: int, index: int | torch.LongTensor, batches: int
    ):
        self.add_active(layer, batch, index)
        self.add_inactive(layer, batch + batches // 2, index)

    def iter(
        self,
    ) -> Iterable[tuple[int, list[tuple[int, torch.LongTensor, torch.LongTensor]]]]:
        for layer in sorted(self.spec.keys()):
            yield layer, [(b, act, ina) for b, (act, ina) in self.spec[layer].items()]


def tighten_bounds(
    lirpa: "BoundedModule",
    x: "BoundedTensor",
    bounds: dict[str, tuple[torch.Tensor, torch.Tensor]],
    new_bounds: None | tuple[int, int] | NewBounds = None,
    debug: bool = IS_TEST_OR_DEBUG,
    **kwargs,
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """Calculate tighter bounds afer a split (batched).

    Args:
        lirpa: LiRPA wrapped network.
        x: Input specification.
        bounds: Lower and upper bounds for each layer.
        new_bounds: New bounds that should be propagated backwards.
        debug: Activate additional asserts. Defaults to False unless debugging.

    Returns:
        Tightened `bounds`.
    """
    start_layer = len(lirpa.relus)
    if new_bounds is not None:
        if not isinstance(new_bounds, NewBounds):
            layer, index = new_bounds
            new_bounds = NewBounds()
            new_bounds.add_split(layer, 0, index, x.shape[0])
        for layer, batches in new_bounds.iter():
            layer = tighten_bounds_backward(
                lirpa, x, bounds, layer, batches, debug=debug, **kwargs
            )
            start_layer = min(start_layer, layer)
    return tighten_bounds_forward(lirpa, x, bounds, start_layer, **kwargs, debug=debug)


def tighten_bounds_forward(
    lirpa: "BoundedModule",
    x: "BoundedTensor",
    bounds: dict[str, tuple[torch.Tensor, torch.Tensor]],
    start_layer: int = -1,
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
    start_layer = -1
    if start_layer + 1 >= len(lirpa.relus):
        return bounds
    fix_relus = [lirpa.input_name[0]] if lirpa.input_name[0] in bounds else []
    if start_layer >= 0:
        fix_relus.extend(n for r in lirpa.relus[: start_layer + 1] for n in r.inputs)
    lirpa.compute_bounds(
        x=(x,),
        final_node_name=lirpa.relus[-1].inputs[0].name,
        **kwargs,
        bound_upper=True,
        bound_lower=True,
        reference_bounds=bounds,
        intermediate_layer_bounds={k: bounds[k] for k in fix_relus},
    )
    for key, (lbo, ubo) in bounds.items():
        if key not in fix_relus and getattr(lirpa[key], "lower", None) is not None:
            if debug:
                eps = torch.finfo(ubo.dtype).eps * 2
                assert (lirpa[key].lower <= ubo + eps).all().cpu().item()
                assert (lirpa[key].upper >= lbo - eps).all().cpu().item()
            bounds[key] = (
                torch.clamp(lirpa[key].lower.detach(), lbo, ubo).contiguous(),
                torch.clamp(lirpa[key].upper.detach(), lbo, ubo).contiguous(),
            )
    return bounds


def tighten_bounds_backward(
    lirpa: "BoundedModule",
    x: "BoundedTensor",
    bounds: dict[str, tuple[torch.Tensor, torch.Tensor]],
    layer: int,
    batches: list[tuple[int, torch.LongTensor, torch.LongTensor]],
    debug: bool = IS_TEST_OR_DEBUG,
    **kwargs,
) -> int:
    """Tighten the bounds of previous layers (batched).
    This function runs LiRPA to get linear bounds.

    Args:
        lirpa: LiRPA wrapped network.
        x: Input specification.
        bounds: Lower and upper bounds per layer.
        layer: Layer to tighten bounds for.
        batches: List of neurons with new bounds: `(batch_index, active_indices, inactive_indices)`.
        debug: Activate additional asserts. Defaults to False unless debugging.

    Returns:
        The lowest modified layer.
    """
    final = lirpa.relus[layer].inputs[0].name
    prev = [r.name for r in lirpa.relus[:layer]]
    if lirpa.input_name[0] in bounds:
        prev.append(lirpa.input_name[0])
    if len(prev) == 0:
        return layer
    num_out = max(ai.numel() + ii.numel() for _, ai, ii in batches)
    c = x.new_zeros(x.shape[0], num_out, *lirpa.relus[layer].output_shape[1:])
    for b, ai, ii in batches:
        na, ni = ai.numel(), ii.numel()
        c.view(c.shape[0], c.shape[1], -1)[b, range(na), ai] = 1.0
        c.view(c.shape[0], c.shape[1], -1)[b, range(na, na + ni), ii] = 1.0
    if debug:
        assert c.sum().item() == sum(ai.numel() + ii.numel() for _, ai, ii in batches)
    _, _, A2 = lirpa.compute_bounds(
        x=(x,),
        C=c,
        final_node_name=final,
        **kwargs,
        bound_upper=True,
        bound_lower=True,
        return_A=True,
        need_A_only=True,
        needed_A_dict={final: prev},
        intermediate_layer_bounds=bounds,
    )

    with torch.no_grad():
        lb_curr, ub_curr = bounds[final]
        for i, p in enumerate(prev):
            if p in bounds:
                lb_prev, ub_prev = bounds[p]
            else:
                lb_prev, ub_prev = bounds[lirpa[p].inputs[0].name]
            if tighten_bounds_back(
                A2[final][p]["lA"],
                A2[final][p]["lbias"],
                A2[final][p]["uA"],
                A2[final][p]["ubias"],
                lb_prev,
                ub_prev,
                lb_curr,
                ub_curr,
                batches,
                sparse_c=True,
                debug=debug,
            ):
                layer = min(layer, i - 1)
    return layer


def tighten_bounds_back(
    lA: torch.Tensor,
    lbias: torch.Tensor,
    uA: torch.Tensor,
    ubias: torch.Tensor,
    lb_prev: torch.Tensor,
    ub_prev: torch.Tensor,
    lb_curr: torch.Tensor,
    ub_curr: torch.Tensor,
    batches: list[tuple[int, torch.LongTensor, torch.LongTensor]],
    sparse_c: bool = False,
    debug: bool = IS_TEST_OR_DEBUG,
) -> bool:
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
        batches: List of neurons with new bounds: `(batch_index, active_indices, inactive_indices)`.
        sparse_c: The linear models only have outputs for the indices in the batches.
        debug: Activate additional asserts. Defaults to False unless debugging.
    """
    eps = torch.finfo(lA.dtype).eps * 2
    epsl = torch.finfo(lA.dtype).eps ** 0.5
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

    tighter = False
    for b, active, inactive in batches:
        na, ni = len(active), len(inactive)
        if na > 0:
            A = uA[b, :na] if sparse_c else uA[b, active]
            bias = ubias[b, :na] if sparse_c else ubias[b, active]
            ci = A * center[b, None] + A.abs() * diff[b, None]
            c = ci.flatten(1).sum(1)
            ln = ((lb_curr[b, active] - bias - c).reshape(flat2) + ci) / A
            lb_new = torch.where((A > epsl), ln, lb_prev[b, None]).max(0)[0]
            ub_new = torch.where((A < -epsl), ln, ub_prev[b, None]).min(0)[0]
            if debug:
                assert (lb_new <= ub_prev[b] + eps).all().cpu().item()
                assert (ub_new >= lb_prev[b] - eps).all().cpu().item()
            if torch.any(lb_new > lb_prev[b]).item():
                tighter = True
                lb_prev[b] = torch.maximum(lb_new, lb_prev[b])
            if torch.any(ub_new < ub_prev[b]).item():
                tighter = True
                ub_prev[b] = torch.minimum(ub_new, ub_prev[b])

        if ni > 0:
            A = lA[b, na : na + ni] if sparse_c else lA[b, inactive]
            bias = lbias[b, na : na + ni] if sparse_c else lbias[b, inactive]
            ci = A * center[b, None] - A.abs() * diff[b, None]
            c = ci.flatten(1).sum(1)
            un = ((ub_curr[b, inactive] - bias - c).reshape(flat2) + ci) / A
            lb_new = torch.where((A < -epsl), un, lb_prev[b, None]).max(0)[0]
            ub_new = torch.where((A > epsl), un, ub_prev[b, None]).min(0)[0]
            if debug:
                assert (lb_new <= ub_prev[b] + eps).all().cpu().item()
                assert (ub_new >= lb_prev[b] - eps).all().cpu().item()
            if torch.any(lb_new > lb_prev[b]).item():
                tighter = True
                lb_prev[b] = torch.maximum(lb_new, lb_prev[b])
            if torch.any(ub_new < ub_prev[b]).item():
                tighter = True
                ub_prev[b] = torch.minimum(ub_new, ub_prev[b])
    return tighter


def tighten_backwards(
    A: torch.Tensor,
    bias: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    debug: bool = IS_TEST_OR_DEBUG,
) -> tuple[torch.Tensor, torch.Tensor]:
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
    return torch.maximum(lb_new, lower), torch.minimum(ub_new, upper)
