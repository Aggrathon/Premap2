import numbers
import sys
from io import BytesIO
from pathlib import Path

import torch


def is_test_or_debug() -> bool:
    """Try to detect debugging or pytest:ing"""
    return (
        hasattr(sys, "gettrace") and (sys.gettrace() is not None)
    ) or "pytest" in sys.modules


IS_TEST_OR_DEBUG: bool = is_test_or_debug()


def device_memory(device: str | torch.device) -> None | int:
    """Try to get the amount memory of the chosen GPU."""
    if not torch.cuda.is_available():
        return None
    try:
        return torch.cuda.get_device_properties(device).total_memory
    except:  # noqa: E722
        return None


def expand_patch(
    X: torch.Tensor,
    full: torch.Tensor,
    mask: torch.Tensor | tuple[int, int, int, int] | None,
) -> torch.Tensor:
    if mask is None:
        return X
    if X.shape[0] == 0:
        return full[None][:0]
    if full.size(0) != 1:
        full = full[None]
    if X.shape[1:] == full.shape[1:]:
        return X
    full_X = full.expand(X.shape[0], *full.shape[1:]).clone()
    if isinstance(mask, torch.Tensor):
        full_X[..., mask] = X
    else:
        x, y, w, h = mask
        full_X[..., x : x + w, y : y + h] = X
    return full_X


@torch.jit.script  # type: ignore
def ess_exp(log_weights: torch.Tensor) -> float:
    """Calculate the Effective Sample Size for log weights."""
    top = torch.logsumexp(log_weights, 0) * 2
    bot = torch.logsumexp(log_weights * 2, 0)
    return (top - bot).exp().item()


@torch.jit.script  # type: ignore
def sum_exp(x: torch.Tensor) -> torch.Tensor:
    """Calculate the `sum(exp(x), dim=0)`"""
    max = x.max()
    if torch.isfinite(max):
        return max.exp() * (x - max).exp().sum(0)
    else:
        return x.exp().sum(0)


def sizeof_tensors(item: object) -> int:
    """Calculate the size of the tensors in the item (traverses lists, tuples, dicts and objects)."""
    if isinstance(item, torch.Tensor):
        return item.storage().nbytes() + 1
        # return item.element_size() * item.nelement()
    elif isinstance(item, (list, tuple)) and len(item) > 0:
        size = sizeof_tensors(item[0])
        if size:
            return size + sum(sizeof_tensors(i) for i in item[1:])
        return 0
    elif isinstance(item, dict) and len(item) > 0:
        size = 0
        for v in item.values():
            size += sizeof_tensors(v)
            if size == 0:
                return 0
        return size
    elif hasattr(item, "__dict__") and len(item.__dict__) > 0:
        return sum(sizeof_tensors(v) for v in item.__dict__.values())
    return 0


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

    @property
    def layers(self) -> int:
        return len(self.handles)

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


class AppendableTensor:
    __slots__ = ["tensor", "size", "reserve"]

    def __init__(self, tensor: torch.Tensor | None = None, reserve: int = 0):
        self.reserve = reserve
        if tensor is None:
            self.size = 0
        else:
            self.tensor = tensor.detach()
            self.size = tensor.shape[0]

    def get(self) -> torch.Tensor:
        return self.tensor[: self.size]

    def append(self, tensor: torch.Tensor):
        if self.size == 0:
            self.tensor = tensor
            self.size = tensor.shape[0]
        else:
            size = tensor.shape[0]
            if self.size + size > self.tensor.shape[0]:
                new_size = max(self.reserve, (self.size + size + 100) * 3 // 2)
                new = tensor.new_empty((new_size, *tensor.shape[1:]))
                new[: self.size] = self.tensor[: self.size]
                self.tensor = new
            self.tensor[self.size : self.size + size] = tensor
            self.size += size

    def __len__(self) -> int:
        return self.size


class CycleTensor:
    __slots__ = ["tensor", "size", "index"]

    def __init__(self, tensor: torch.Tensor, size: int):
        self.tensor = tensor.new_empty((size, *tensor.shape[1:]))
        if tensor.shape[0] > size:
            self.size = size
            self.index = 0
            self.tensor[:] = tensor[-size:]
        else:
            self.index = self.size = tensor.shape[0]
            self.tensor[: self.size] = tensor

    def get(self) -> torch.Tensor:
        return self.tensor[: self.size]

    def get_random(self, num: int) -> torch.Tensor:
        return self.tensor[torch.randint(self.size, (num,))]

    def append(self, tensor: torch.Tensor):
        if self.index + tensor.shape[0] <= self.tensor.shape[0]:
            self.tensor[self.index : self.index + tensor.shape[0]] = tensor
            self.index += tensor.shape[0]
            self.size = max(self.size, self.index)
        else:
            self.size = self.tensor.shape[0]
            bef = self.size - self.index
            self.tensor[self.index :] = tensor[:bef]
            self.index = tensor.shape[0] - bef
            self.tensor[: self.index] = tensor[bef:]

    def __len__(self) -> int:
        return self.size


def history_contains(
    history: list[tuple[torch.Tensor, torch.Tensor]],
    activations: list[torch.Tensor],
    epsilon: float | None = None,
) -> torch.Tensor | slice:
    """Return a mask for all items within the split.
    This version takes the history as index_active&index_inactive.

    Args:
        history: Split history (layers[(below, above)]).
        activations: Activations of the pre relu layers.
        epsilon: Numerical tolerance.

    Returns:
        Mask of items inside the split.
    """
    mask = True
    for (active, inactive), a in zip(history, activations):
        eps = torch.finfo(a.dtype).eps if epsilon is None else epsilon
        if len(active):
            mask = (a.flatten(1)[:, active].min(1)[0] >= -eps) & mask
        if len(inactive):
            mask = (a.flatten(1)[:, inactive].max(1)[0] <= eps) & mask
    if isinstance(mask, bool):
        return slice(None)
    return mask


def history_to_index(
    history: list[tuple[list[int], list[float]]]
    | list[tuple[torch.Tensor, torch.Tensor]],
    sort: bool = False,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Converts the history from index&sign to index_above&index_below."""
    if history and isinstance(history[0][0], torch.Tensor):
        return [(i[s >= 0.0], i[s < 0.0]) for i, s in history]  # type: ignore
    array = sorted if sort else list
    return [
        (
            torch.LongTensor(array(i for i, s in zip(*h) if s >= 0)),
            torch.LongTensor(array(i for i, s in zip(*h) if s < 0)),
        )
        for h in history
    ]


def history_to_sign(
    history: list[tuple[torch.Tensor, torch.Tensor]],
    min_layers: int = 0,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Converts the history from index_above&index_below to index&sign"""
    dt = torch.get_default_dtype()
    his = [
        (
            torch.cat((a, b)),
            torch.cat((torch.ones_like(a, dtype=dt), -torch.ones_like(b, dtype=dt))),
        )
        for a, b in history
    ]
    while len(his) < min_layers:
        his.append(([], []))  # type: ignore
    return his


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
        for x, lb, ub in zip(X, lower, upper):  # type: ignore
            assert_bounds(x, lb, ub, epsilon)


def assert_history_contains(
    activations: list[torch.Tensor],
    history: list[tuple[torch.Tensor, torch.Tensor]],
    epsilon: float | None = None,
):
    mask = history_contains(history, activations, epsilon=epsilon)
    assert isinstance(mask, slice) or mask.all().item()


def assert_polytope_contains(
    X: torch.Tensor,
    A: torch.Tensor | None = None,
    b: torch.Tensor | None = None,
    lower: torch.Tensor | None = None,
    upper: torch.Tensor | None = None,
    epsilon: float | None = None,
):
    mask = polytope_contains(X, A, b, lower, upper, epsilon)
    assert isinstance(mask, slice) or mask.all().item()


def result_contains(
    X: torch.Tensor,
    result: Path | BytesIO | dict[str, object],
    model: torch.nn.Module,
    **kwargs,
) -> torch.Tensor:
    """Check which items are inside a finished preimage approximation.

    Args:
        X: Items.
        result: PREMAP result (file or already loaded dictionary).
        model: Prediction function/model.

    Returns:
        Boolean vector.
    """
    _, activations = WithActivations(model.to(X.device))(X)
    if not isinstance(result, dict):
        result = torch.load(result, map_location=X.device, **kwargs)
    contains = torch.zeros(X.shape[0], dtype=torch.bool)
    for A, b, _, _, hist in result["domains"]:  # type: ignore
        contp = polytope_contains(X, A.to(X.device), b.to(X.device))
        if isinstance(contp, slice):
            conts = history_contains(hist, activations)
            if not isinstance(conts, slice):
                contains |= conts
        elif contp.any():
            conts = history_contains(hist, activations)
            if isinstance(conts, slice):
                contains |= contp
            else:
                contains |= contp & conts
    return contains
