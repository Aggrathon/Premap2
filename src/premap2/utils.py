import sys

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
    history: list[tuple[list[int], list[int]]], activations: list[torch.Tensor]
) -> torch.Tensor | slice:
    """Return a mask for all items within the split.
    This version takes the history as index_below&index_above.

    Args:
        history: Split history (layers[(below, above)]).
        activations: Activations of the pre relu layers.

    Returns:
        Mask of items inside the split.
    """
    mask = True
    for (below, above), a in zip(history, activations):
        if above:
            mask = (a.flatten(1)[:, above] >= 0).all(1) & mask
        if below:
            mask = (a.flatten(1)[:, below] <= 0).all(1) & mask
    if isinstance(mask, bool):
        return slice(None)
    return mask


def history_to_index(
    history: list[tuple[list[int], list[float]]],
) -> list[tuple[list[int], list[int]]]:
    """Converts the history from index&sign to index_below&index_above."""
    return [
        ([i for i, s in zip(*h) if s < 0], [i for i, s in zip(*h) if s >= 0])
        for h in history
    ]


def split_contains2(
    history: list[tuple[list[int], list[float]]], activations: list[torch.Tensor]
) -> torch.Tensor | slice:
    """Return a mask for all items within the split.
    This version takes the history as index&sign.

    Args:
        history: Split history (layers[(index, sign)]).
        activations: Activations of the pre relu layers.

    Returns:
        Mask of items inside the split.
    """
    return split_contains(history_to_index(history), activations)


def polytope_contains(
    X: torch.Tensor,
    A: torch.Tensor | None = None,
    b: torch.Tensor | None = None,
    lower: torch.Tensor | None = None,
    upper: torch.Tensor | None = None,
    epsilon: float = 0.0,
) -> torch.Tensor | slice:
    """Return a mask for all items within the polytope.

    Args:
        X: Samples.
        A: Polytope constraints coefficients.
        b: Polytope constraints bias.
        lower: Lower bounds.
        upper: Upper bounds.
        epsilon: Tolerance.

    Returns:
        Mask of items inside the polytope.
    """
    if A is not None:
        assert b is not None
        check = torch.all(torch.einsum("n...,b...->nb", X, A) + b[None] >= -epsilon, 1)
    else:
        check = True
    if lower is not None:
        check &= (X >= lower).flatten(1).all(1)
    if upper is not None:
        check &= (X <= upper).flatten(1).all(1)
    if check is True:
        return slice(None)
    return check
