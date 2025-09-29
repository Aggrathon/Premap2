import numbers
import sys
from itertools import repeat
from pathlib import Path

import numpy as np
import torch


def is_test_or_debug() -> bool:
    """Try to detect debugging or pytest:ing"""
    return (
        hasattr(sys, "gettrace") and (sys.gettrace() is not None)
    ) or "pytest" in sys.modules


IS_TEST_OR_DEBUG: bool = is_test_or_debug()


def is_int(x) -> bool:
    """Check if x is an scalar integer (including torch and numpy)."""
    if isinstance(x, torch.Tensor) and x.numel() == 1:
        x = x.detach().cpu().item()
    if isinstance(x, np.ndarray) and x.size == 1:
        x = x.flat[0]
    return isinstance(x, (int, np.integer))


class WithActivations:
    def __init__(self, model: torch.nn.Module):
        """Module wrapper that returns predictions and activations of pre relu layers."""
        self.acts = None
        self.handles = [
            m.register_forward_pre_hook(self._add_act)
            for m in model.modules()
            if isinstance(m, torch.nn.ReLU)
        ]
        assert len(self.handles) > 0, "No relu layers found"
        self.model = model

    def _add_act(self, m: torch.nn.Module, input: tuple[torch.Tensor, ...]):
        if self.acts is not None:
            self.acts.append(input[0].detach())

    def __call__(self, X: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Call the model.

        Args:
            X: Input tensor.

        Returns:
            y: Model output.
            activations: Activations of the pre relu layers.
        """
        self.acts = []
        y = self.model(X)
        acts = self.acts
        self.acts = None
        return y, acts

    def __del__(self):
        for handle in self.handles:
            handle.remove()


def split_contains(
    history: list[tuple[torch.LongTensor, torch.LongTensor]],
    activations: list[torch.Tensor],
    epsilon: float | None = None,
) -> torch.Tensor | slice:
    """Return a mask for all items within the split.
    This version takes the history as index_below&index_above.

    Args:
        history: Split history (layers[(below, above)]).
        activations: Activations of the pre relu layers.
        epsilon: Numerical tolerance.

    Returns:
        Mask of items inside the split.
    """
    mask = True
    for (below, above), a in zip(history, activations):
        eps = torch.finfo(a.dtype).eps if epsilon is None else epsilon
        if len(above):
            mask = (a.flatten(1)[:, above].min(1)[0] > -eps) & mask
        if len(below):
            mask = (a.flatten(1)[:, below].max(1)[0] < eps) & mask
    if isinstance(mask, bool):
        return slice(None)
    return mask


def history_to_index(
    history: list[tuple[list[int], list[float]]], sort: bool = False
) -> list[tuple[torch.LongTensor, torch.LongTensor]]:
    """Converts the history from index&sign to index_below&index_above."""
    array = sorted if sort else list
    return [
        (
            torch.LongTensor(array(i for i, s in zip(*h) if s < 0)),
            torch.LongTensor(array(i for i, s in zip(*h) if s >= 0)),
        )
        for h in history
    ]


def split_contains2(
    history: list[tuple[list[int], list[float]]],
    activations: list[torch.Tensor],
    epsilon: float | None = None,
) -> torch.Tensor | slice:
    """Return a mask for all items within the split.
    This version takes the history as index&sign.

    Args:
        history: Split history (layers[(index, sign)]).
        activations: Activations of the pre relu layers.
        epsilon: Numerical tolerance.

    Returns:
        Mask of items inside the split.
    """
    return split_contains(history_to_index(history), activations, epsilon)


def polytope_contains(
    X: torch.Tensor,
    A: torch.Tensor | None = None,
    b: torch.Tensor | None = None,
    lower: torch.Tensor | None = None,
    upper: torch.Tensor | None = None,
    epsilon: float | None = None,
) -> torch.Tensor | slice:
    """Return a mask for all items within the polytope.

    Args:
        X: Samples.
        A: Polytope constraints coefficients.
        b: Polytope constraints bias.
        lower: Lower bounds.
        upper: Upper bounds.
        epsilon: Numerical tolerance.

    Returns:
        Mask of items inside the polytope.
    """
    epsilon = torch.finfo(X.dtype).eps if epsilon is None else epsilon
    if A is not None:
        assert b is not None
        check = torch.all(torch.einsum("n...,b...->nb", X, A) + b[None] >= -epsilon, 1)
    else:
        check = True
    if lower is not None:
        check &= (X > lower - epsilon).flatten(1).all(1)
    if upper is not None:
        check &= (X < upper + epsilon).flatten(1).all(1)
    if check is True:
        return slice(None)
    return check


def assert_bounds(
    X: torch.Tensor | list[torch.Tensor],
    lower: torch.Tensor | float | list[torch.Tensor] | list[float],
    upper: torch.Tensor | float | list[torch.Tensor] | list[float],
    epsilon: float | None = None,
):
    if (
        isinstance(X, torch.Tensor)
        and isinstance(lower, (torch.Tensor, numbers.Real))
        and isinstance(upper, (torch.Tensor, numbers.Real))
    ):
        # Large GMMs are non-deterministic, so we need a suprisingly large epsilon
        epsilon = torch.finfo(X.dtype).eps ** 0.5 * 0.5 if epsilon is None else epsilon
        assert (X > lower - epsilon).all()
        assert (X < upper + epsilon).all()
    else:
        for x, lb, ub in zip(X, lower, upper):
            assert_bounds(x, lb, ub, epsilon)


def assert_contains_hii(
    activations: list[torch.Tensor],
    history: list[tuple[torch.LongTensor, torch.LongTensor]],
    lower: list[torch.Tensor] | None = None,
    upper: list[torch.Tensor] | None = None,
    epsilon: float | None = None,
):
    if lower is None:
        lower = repeat(None)  # type: ignore
    if upper is None:
        upper = repeat(None)  # type: ignore
    for act, (below, above), lb, ub in zip(activations, history, lower, upper):
        # Large GMMs are non-deterministic, so we need a suprisingly large epsilon
        eps = torch.finfo(act.dtype).eps ** 0.5 * 0.5 if epsilon is None else epsilon
        if lb is not None:
            assert (act > lb - eps).all()
        if ub is not None:
            assert (act < ub + eps).all()
        if len(below):
            assert (act.flatten(1)[:, below] < eps).all()
        if len(above):
            assert (act.flatten(1)[:, above] > -eps).all()


def assert_contains_his(
    activations: list[torch.Tensor],
    history: list[tuple[list[int], list[float]]],
    lower: list[torch.Tensor] | None = None,
    upper: list[torch.Tensor] | None = None,
    epsilon: float | None = None,
):
    assert_contains_hii(activations, history_to_index(history), lower, upper, epsilon)


def results_contains(
    X: torch.Tensor, results: Path | dict[str, object], model: torch.nn.Module, **kwargs
) -> torch.Tensor:
    _, activations = WithActivations(model)(X)
    if not isinstance(results, dict):
        results = torch.load(results, **kwargs)
    contains = torch.zeros(X.shape[0], dtype=torch.bool)
    for A, b, _, _, hist in results["domains"]:  # type: ignore
        contp = polytope_contains(X, A, b)
        if isinstance(contp, slice):
            conts = split_contains(history_to_index(hist), activations)
            if not isinstance(conts, slice):
                contains |= conts
        elif contp.any():
            conts = split_contains(history_to_index(hist), activations)
            if isinstance(conts, slice):
                contains |= contp
            else:
                contains |= contp & conts
    return contains
