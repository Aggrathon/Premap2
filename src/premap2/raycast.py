import numpy as np
import torch

from premap2.utils import IS_TEST_OR_DEBUG


def _verify(
    x: torch.Tensor,
    A: torch.Tensor | None = None,
    b: torch.Tensor | None = None,
    lower: torch.Tensor | None = None,
    upper: torch.Tensor | None = None,
    batch: bool = False,
    epsilon: float = 0.0,
):
    if A is not None:
        assert b is not None
        if batch:
            v = torch.einsum("b...,bn...->bn", x, A) + b.view(A.shape[:2])
        else:
            v = A.flatten(1) @ x.view(-1, 1) + b.view(-1, 1)
        assert torch.all(v >= -epsilon).cpu().item()
    if upper is not None:
        assert torch.all(x <= upper + epsilon).cpu().item()
    if lower is not None:
        assert torch.all(x >= lower - epsilon).cpu().item()


def raycast(
    x: torch.Tensor,
    d: torch.Tensor,
    A: torch.Tensor | None = None,
    b: torch.Tensor | None = None,
    lower: torch.Tensor | None = None,
    upper: torch.Tensor | None = None,
    epsilon: float = 1e-5,
    verify: bool = IS_TEST_OR_DEBUG,
) -> tuple[float, float]:
    """Raycast from the point `x` in direction `d`.
    The ray is extends in both directions, and the distance to the bounds are returned.

    Args:
        x: Ray starting point, shape: [dims...].
        d: Ray direction and step magnitude, shape: [dims...].
        A: Optional linear constraints, coefficients `Ax+b>=0`, shape: [num, dims...].
        b: Optional linear constraint, constants: `Ax+b>=0`, shape: [num].
        lower: Optional bounding box where `x >= lower`, shape: [dims...].
        upper: Optional bounding box, where `x <= upper`, shape: [dims...].
        verify: Verify that the distances are within bounds.

    Returns:
        (start, end): Distance to the bounds in units of `d`.
            The intersections are at `start * d` and `end * d`.
            If `start >= end` then `x` is outside the bounds.
    """
    assert x.shape == d.shape
    forward = np.inf
    back = -np.inf
    mask_pos = d > 0
    mask_neg = d < 0
    any_mask_pos = mask_pos.any().cpu().item()
    any_mask_neg = mask_neg.any().cpu().item()
    if upper is not None:
        bu = (upper - x) / d
        if any_mask_pos:
            forward = min(forward, bu[mask_pos].min().cpu().item())
        if any_mask_neg:
            back = max(back, bu[mask_neg].max().cpu().item())
    if lower is not None:
        bl = (lower - x) / d
        if any_mask_pos:
            back = max(back, bl[mask_pos].max().cpu().item())
        if any_mask_neg:
            forward = min(forward, bl[mask_neg].min().cpu().item())
    if A is not None:
        assert b is not None
        dots = A.flatten(1) @ d.view(-1, 1)
        offs = A.flatten(1) @ x.view(-1, 1) + b.view(-1, 1)
        dists = -offs / dots
        mask_pos = dots > 0
        if mask_pos.any().cpu().item():
            back = max(back, dists[mask_pos].max().cpu().item() + epsilon)
        mask_neg = dots < 0
        if mask_neg.any().cpu().item():
            forward = min(forward, dists[mask_neg].min().cpu().item() - epsilon)
    if verify:
        xf = x + d * (forward if np.isfinite(forward) else 1e10)
        _verify(xf, A, b, lower, upper)
        xb = x + d * (back if np.isfinite(back) else -1e10)
        _verify(xb, A, b, lower, upper)
    return back, forward


@torch.jit.script  # type: ignore
def _min_where(
    value: torch.Tensor, mask: torch.Tensor, dim: int = -1
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    inf = value.new_zeros([]) + np.inf
    val, ind = torch.where(mask, value, inf).min(dim=dim)
    mask = torch.isfinite(val)
    return val[mask], mask, ind[mask]


@torch.jit.script  # type: ignore
def _max_where(
    value: torch.Tensor, mask: torch.Tensor, dim: int = -1
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    inf = value.new_zeros([]) - np.inf
    val, ind = torch.where(mask, value, inf).max(dim=dim)
    mask = torch.isfinite(val)
    return val[mask], mask, ind[mask]


def raycast_batch(
    x: torch.Tensor,
    d: torch.Tensor,
    A: torch.Tensor | None = None,
    b: torch.Tensor | None = None,
    lower: torch.Tensor | None = None,
    upper: torch.Tensor | None = None,
    check_numerical: bool = IS_TEST_OR_DEBUG,
    verify: bool = IS_TEST_OR_DEBUG,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Raycast from points `x` in directions `d`.
    The rays extend in both directions for each point, and the distance to the bounds are returned.

    This function tries to correct numerical issues

    Args:
        x: Batch of ray starting points, shape: [batch or 1, dims...].
        d: Batch of ray direction and step magnitude, shape: [batch or 1, dims...].
        A: Optional linear constraints, coefficients `Ax+b>=0`, shape: [batch or 1, num, dims...].
        b: Optional linear constraint, constants: `Ax+b>=0`, shape: [batch or 1, num].
        lower: Optional bounding box where `x >= lower`, shape: [batch or 1, dims...].
        upper: Optional bounding box, where `x <= upper`, shape: [batch or 1, dims...].
        verify: Verify that the distances are within bounds.

    Returns:
        (start, end): Distances to the bounds in units of `d` for each ray.
            The intersections are at `start * d` and `end * d` for each batch element.
            If `start >= end` then the corresponding point is outside the bounds.
    """
    epsilon = torch.finfo(x.dtype).eps * 3.0
    n = max(x.shape[0], d.shape[0])
    assert x.shape[1:] == d.shape[1:]
    x, d = x.flatten(1).expand(n, -1), d.flatten(1).expand(n, -1)
    forward = x.new_full((n,), np.inf)
    back = x.new_full((n,), -np.inf)
    mask_pos = d > epsilon
    mask_neg = d < -epsilon
    any_mask_pos = mask_pos.any().cpu().item()
    any_mask_neg = mask_neg.any().cpu().item()
    if upper is not None:
        upper = upper.flatten(1).expand(n, -1)
        assert x.shape == upper.shape
        bu = (upper - x) / d
        if any_mask_pos:
            fo, mask, ind = _min_where(bu, mask_pos, -1)
            if check_numerical:
                vals = upper[mask, ind] - x[mask, ind] - fo * d[mask, ind]
                prec = -torch.relu(-vals) * 4 / d[mask, ind]
                forward[mask] = torch.minimum(forward[mask], fo + prec)
            else:
                forward[mask] = torch.minimum(forward[mask], fo)
        if any_mask_neg:
            ba, mask, ind = _max_where(bu, mask_neg, -1)
            if check_numerical:
                vals = upper[mask, ind] - x[mask, ind] - ba * d[mask, ind]
                prec = -torch.relu(-vals) * 4 / d[mask, ind]
                back[mask] = torch.maximum(back[mask], ba + prec)
            else:
                back[mask] = torch.maximum(back[mask], ba)
    if lower is not None:
        lower = lower.flatten(1).expand(n, -1)
        assert x.shape == lower.shape
        bl = (lower - x) / d
        if any_mask_pos:
            ba, mask, ind = _max_where(bl, mask_pos, -1)
            if check_numerical:
                vals = x[mask, ind] + ba * d[mask, ind] - lower[mask, ind]
                prec = torch.relu(-vals) * 4 / d[mask, ind]
                back[mask] = torch.maximum(back[mask], ba + prec)
            else:
                back[mask] = torch.maximum(back[mask], ba)
        if any_mask_neg:
            fo, mask, ind = _min_where(bl, mask_neg, -1)
            if check_numerical:
                vals = x[mask, ind] + fo * d[mask, ind] - lower[mask, ind]
                prec = torch.relu(-vals) * 4 / d[mask, ind]
                forward[mask] = torch.minimum(forward[mask], fo + prec)
            else:
                forward[mask] = torch.minimum(forward[mask], fo)
    if A is not None:
        assert b is not None
        b = b.reshape(A.shape[:2]).expand(n, -1)
        A = A.flatten(2).expand(n, -1, -1)
        assert x.shape == A[:, 0].shape
        dots = torch.einsum("b...,bn...->bn", d, A)  # [batch, num]
        offs = torch.einsum("b...,bn...->bn", x, A) + b  # [bat, num]
        dists = -offs / dots
        mask_pos = dots > epsilon
        if mask_pos.any().cpu().item():
            ba, mask, ind = _max_where(dists, mask_pos, -1)
            if not check_numerical:
                back[mask] = torch.maximum(back[mask], ba)
            elif (ba > back[mask]).any().cpu().item():
                ba = torch.maximum(ba, back[mask])
                vals = torch.einsum(
                    "b...,bn...->bn", (x[mask] + d[mask] * ba[:, None]), A[mask]
                )
                prec = torch.relu(-vals - b[mask]) * 16 / dots[mask]
                back[mask] = torch.maximum(back[mask], ba + prec.max(1)[0])
        mask_neg = dots < -epsilon
        if mask_neg.any().cpu().item():
            fo, mask, ind = _min_where(dists, mask_neg, -1)
            if not check_numerical:
                forward[mask] = torch.minimum(forward[mask], fo)
            elif (fo < forward[mask]).any().cpu().item():
                fo = torch.minimum(fo, forward[mask])
                vals = torch.einsum(
                    "b...,bn...->bn", (x[mask] + d[mask] * fo[:, None]), A[mask]
                )
                prec = torch.relu(-vals - b[mask]) * 16 / dots[mask]
                forward[mask] = torch.minimum(forward[mask], fo + prec.min(1)[0])
    if verify:
        xf = x + d * torch.where(torch.isfinite(forward), forward, 1e10)[:, None]
        _verify(xf, A, b, lower, upper, True, 0.0 if check_numerical else epsilon)
        xb = x + d * torch.where(torch.isfinite(back), back, -1e10)[:, None]
        _verify(xb, A, b, lower, upper, True, 0.0 if check_numerical else epsilon)
    return back, forward
