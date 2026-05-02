from dataclasses import dataclass

import numpy as np
import torch

from premap2.domains import Domain

try:
    from auto_LiRPA.patches import Patches
    from premap.preimage_beta_crown_solver_relu_split import LiRPAConvNet
except ImportError:
    pass


@dataclass
class LinearBounds:
    lA: "torch.Tensor | Patches"
    lb: torch.Tensor
    uA: "torch.Tensor | Patches"
    ub: torch.Tensor
    lower: torch.Tensor
    upper: torch.Tensor


def calc_constraints(
    net: "LiRPAConvNet",
    domains: list[Domain],
    lower: list[torch.Tensor],
    upper: list[torch.Tensor],
    mask: torch.Tensor | None = None,
    num_samples: int = 20_000,
):
    """Calculate constraints to store in the `samples` (used to calculate priorities and sampling polytopes).

    Args:
        net: LiRPA wrapped network.
        domains: batch of `Domains`.
        lower: Lower bounds.
        upper: Upper bounds.
        num_samples: Number of samples (same as in calc_samples).
    """
    # This function assumes it is called directly after ´net.get_lower_bound´
    for i, d in enumerate(domains):
        if net.net.relus[0].inputs[0].lA is not None:
            d.lower_As = [
                relu.inputs[0].lA.detach()[:, i].contiguous() for relu in net.net.relus
            ]
        if net.net.relus[0].inputs[0].uA is not None:
            d.upper_As = [
                relu.inputs[0].uA.detach()[:, i].contiguous() for relu in net.net.relus
            ]
    limit = num_samples * 8 // 10
    to_update = [
        i
        for i, d in enumerate(domains)
        if d.constraints and len(d) < limit and d.priority != -np.inf
    ]
    if not to_update:
        return
    Abs = net.get_intermediate_constraints(range(len(net.net.relus)))
    with torch.no_grad():
        for i in to_update:
            d = domains[i]
            d.poly_A = d.poly_b = None
            ab = [
                LinearBounds(
                    _patch_index(input["lA"], i),
                    input["lbias"][i],
                    _patch_index(input["uA"], i),
                    input["ubias"][i],
                    lb[i],
                    ub[i],
                )
                for layer, lb, ub in zip(Abs, lower, upper)
                for input in layer.values()
            ]
            assert d.lower_in is not None and d.upper_in is not None
            d.poly_A, d.poly_b = get_constraints(
                d.constraints, ab, d.lower_in, d.upper_in, mask
            )


def _patch_index(value: torch.Tensor | Patches, index: int) -> torch.Tensor | Patches:
    if isinstance(value, torch.Tensor):
        return value[index]
    else:
        shape = (1, *value.output_shape[1:])  # type: ignore
        patch = value.patches[:, index, None].contiguous()  # type: ignore
        return value.create_similar(patch, output_shape=shape)  # type: ignore


def get_constraints(
    history: list[tuple[list[int], list[int]]],
    layers: list[LinearBounds],
    lower: torch.Tensor,
    upper: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor] | tuple[None, None]:
    """Extract input constraints from linear bounds.

    Args:
        history: Split history.
        layers: Linear bounds.
        lower: Input lower bounds.
        upper: Input upper bounds.

    Returns:
       Constraint coefficients and biases.
    """
    As, bs = [], []
    for (below, above), Ab in zip(history, layers):
        if above:
            A, b, c = Ab.uA, Ab.ub.flatten(), Ab.lower.flatten()
            if isinstance(A, Patches):
                As.append(sparse_patches_to_matrix(A, above, lower.shape)[0])  # type: ignore
            else:
                As.append(A[above])  # type: ignore
            bs.append(b[above] - c[above].to(b.device))
        if below:
            A, b, c = Ab.lA, Ab.lb.flatten(), Ab.upper.flatten()
            if isinstance(A, Patches):
                As.append(-sparse_patches_to_matrix(A, below, lower.shape)[0])  # type: ignore
            else:
                As.append(-A[below])  # type: ignore
            bs.append(c[below].to(b.device) - b[below])
    if As:
        A, b = torch.cat(As), torch.cat(bs)
        mid = (lower + upper) * 0.5
        dif = (upper - lower) * 0.5
        eps = torch.finfo(lower.dtype).eps * 3
        filter = (A * mid - A.abs() * dif).flatten(1).sum(1) + b < -eps
        if not filter.all().item():
            # Some constraint boundaries might lie completely outside the bounding box
            A, b = A[filter], b[filter]
        if A.shape[0] > 0:
            if mask is not None:
                b = b + (A * lower)[:, ~mask].sum(1)
                A = A[:, mask]
            return A.contiguous(), b.contiguous()
    return None, None


def sparse_patches_to_matrix(
    patch: "Patches", indices: list[int], shape: torch.Size
) -> torch.Tensor:
    c, _, w, h, *_ = patch.shape  # type: ignore
    c, w, h = np.unravel_index(indices, (c, w, h))
    c, w, h = torch.tensor(c), torch.tensor(w), torch.tensor(h)
    patches = patch.patches[c, :, w, h, ...]  # type: ignore
    return patch.create_similar(patches, unstable_idx=(c, w, h)).to_matrix(shape)
