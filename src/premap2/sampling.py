from dataclasses import dataclass
from typing import TypeVar

import torch

from auto_LiRPA.patches import Patches
from premap2.raycast import raycast_batch
from premap2.tighten_bounds import tighten_backwards
from premap2.utils import (
    IS_TEST_OR_DEBUG,
    WithActivations,
    history_to_index,
    polytope_contains,
    split_contains,
)

try:
    from premap.preimage_beta_crown_solver_relu_split import LiRPAConvNet
except ImportError:
    pass


@dataclass
class Samples:
    """Dataclass for containing the samples and data needed to calculate priorities."""

    X: torch.Tensor
    y: torch.Tensor
    lower: torch.Tensor
    upper: torch.Tensor
    activations: None | list[torch.Tensor] = None
    priority: None | list[torch.Tensor] = None
    A: torch.Tensor | None = None
    b: torch.Tensor | None = None
    lAs: list[torch.Tensor] | None = None
    uAs: list[torch.Tensor] | None = None
    mask: torch.Tensor | None = None
    counter: int = 0
    num: int = 1000

    def split(self, layer: int, index: int) -> tuple["Samples", "Samples"]:
        """Split samples into two on a neuron.

        Args:
            layer: Layer to split on.
            index: Neuron to split on.

        Returns:
            Two `Samples`.
        """
        assert self.activations is not None
        mask = self.activations[layer].flatten(1)[:, index] > 0.0
        samples_left = Samples(
            self.X[mask].contiguous(),
            self.y[mask].contiguous(),
            self.lower,
            self.upper,
            [v[mask].contiguous() for v in self.activations],
            self.priority,
            A=self.A,
            b=self.b,
            lAs=self.lAs,
            uAs=self.uAs,
            mask=self.mask,
            counter=self.counter,
            num=self.num,
        )
        mask = ~mask
        samples_right = Samples(
            self.X[mask].contiguous(),
            self.y[mask].contiguous(),
            self.lower,
            self.upper,
            [v[mask].contiguous() for v in self.activations],
            self.priority,
            A=self.A,
            b=self.b,
            lAs=self.lAs,
            uAs=self.uAs,
            mask=self.mask,
            counter=self.counter,
            num=self.num,
        )
        return samples_left, samples_right

    def reuse(self) -> "Samples":
        """Reuse the samples. Triggers a recalculation of the priorities if they are not fresh enough."""
        self.counter += 1
        if self.counter >= 2:
            self.counter = 0
            self.priority = None
        return self

    def to(self, device: torch.device | str, non_blocking: bool = True) -> "Samples":
        """Move tensors to a device."""
        if isinstance(device, str):
            device = torch.device(device)
        if self.X.device == device:
            return self
        return Samples(
            X=_move_to(self.X, device, non_blocking),
            y=_move_to(self.y, device, non_blocking),
            lower=_move_to(self.lower, device, non_blocking),
            upper=_move_to(self.upper, device, non_blocking),
            activations=_move_to(self.activations, device, non_blocking),
            priority=_move_to(self.priority, device, non_blocking),
            A=_move_to(self.A, device, non_blocking),
            b=_move_to(self.b, device, non_blocking),
            lAs=_move_to(self.lAs, device, non_blocking),
            uAs=_move_to(self.uAs, device, non_blocking),
            mask=_move_to(self.mask, device, non_blocking),
            counter=self.counter,
            num=self.num,
        )

    def __len__(self) -> int:
        return self.X.shape[0]

    def unstable(self) -> list[torch.Tensor]:
        """Get masks for the unstable activations."""
        assert self.activations is not None
        return [(act > 0).any(0) & (act < 0).any(0) for act in self.activations]


TensorLike = TypeVar(
    "TensorLike", torch.Tensor, Patches, torch.Tensor | None, list[torch.Tensor] | None
)


def _move_to(
    tensor: TensorLike, device: torch.device | str, non_blocking: bool = True
) -> TensorLike:
    if tensor is None:
        return tensor
    elif isinstance(tensor, (list, tuple)):
        return type(tensor)(
            t.to(device=device, non_blocking=non_blocking) for t in tensor
        )
    elif isinstance(tensor, Patches):
        return tensor.create_similar(
            tensor.patches.to(device=device, non_blocking=non_blocking)
        )
    return tensor.to(device=device, non_blocking=non_blocking)


def _expand_patch(
    samples: torch.Tensor, full: torch.Tensor, mask: torch.Tensor | None
) -> torch.Tensor:
    if mask is None or samples.shape[1:] == mask.shape:
        return samples
    if samples.shape[0] == 0:
        return full.new_empty((0, *mask.shape))
    X = full.expand(samples.shape[0], *mask.shape).clone()
    X[:, mask] = samples
    return X


@dataclass
class LinearBounds:
    lA: torch.Tensor | Patches
    lb: torch.Tensor
    uA: torch.Tensor | Patches
    ub: torch.Tensor
    lower: torch.Tensor
    upper: torch.Tensor


def calc_constraints(
    net: "LiRPAConvNet",
    samples: list[Samples],
    history: list[list[tuple[list[int], list[float]]]] | None,
    lower: list[torch.Tensor],
    upper: list[torch.Tensor],
    debug: bool = IS_TEST_OR_DEBUG,
):
    """Calculate constraints to store in the `samples` (used to calculate priorities and sampling polytopes).

    Args:
        net: LiRPA wrapped network.
        samples: batch of `Samples`.
        history: Split history.
        lower: Lower bounds.
        upper: Upper bounds.
        debug: Activate additional asserts. Defaults to False unless debugging.
    """
    # This function assumes it is called directly after ´net.get_lower_bound´
    for i, s in enumerate(samples):
        if net.net.relus[0].inputs[0].lA is not None:
            s.lAs = [
                input.lA.detach()[:, i]
                for relu in net.net.relus
                for input in relu.inputs  # ReLUs have only one input
            ]
        if net.net.relus[0].inputs[0].uA is not None:
            s.uAs = [
                input.uA.detach()[:, i]
                for relu in net.net.relus
                for input in relu.inputs  # ReLUs have only one input
            ]
    if history is None:
        return
    Abs = net.get_intermediate_constraints(range(len(samples[0].activations)))
    for i, (s, hist) in enumerate(zip(samples, history)):
        hist = history_to_index(hist)
        ab = [
            LinearBounds(
                _get_Abs(input, "lA", i),
                _get_Abs(input, "lbias", i),
                _get_Abs(input, "uA", i),
                _get_Abs(input, "ubias", i),
                lb[i],
                ub[i],
            )
            for layer, lb, ub in zip(Abs, lower, upper)
            for input in layer.values()
        ]
        s.A, s.b = get_constraints(hist, ab, s.lower, s.upper, debug=debug)
        if s.A is not None and s.mask is not None:
            s.b = s.b + (s.A * s.lower)[:, ~s.mask].sum(1)
            s.A = s.A[:, s.mask]


def _get_Abs(
    Abs: dict[str, torch.Tensor | Patches], key: str, index: int
) -> torch.Tensor | Patches | None:
    value = Abs.get(key, None)
    if value is None:
        return value
    if isinstance(value, Patches):
        return value.create_similar(
            value.patches[:, index, None], output_shape=(1, *value.output_shape[1:])
        )
    elif index >= 0:
        return value[index]
    else:
        return value


def get_constraints(
    history: list[tuple[list[int], list[int]]],
    layers: list[LinearBounds],
    lower: torch.Tensor,
    upper: torch.Tensor,
    debug: bool = IS_TEST_OR_DEBUG,
) -> tuple[torch.Tensor, torch.Tensor] | tuple[None, None]:
    """Extract input constraints from linear bounds.

    Args:
        history: Split history.
        layers: Linear bounds.
        lower: Input lower bounds.
        upper: Input upper bounds.
        debug: Activate additional asserts. Defaults to False unless debugging.

    Returns:
       Constraint coefficients and biases.
    """
    As, bs = [], []
    for (below, above), Ab in zip(history, layers):
        if above:
            A, b, c = Ab.uA, Ab.ub.flatten(), Ab.lower.flatten()
            if isinstance(A, Patches):
                A = A.to_matrix(lower.shape)[0]
            As.append(A[above])
            bs.append(b[above] - c[above].to(b.device))
        if below:
            A, b, c = Ab.lA, Ab.lb.flatten(), Ab.upper.flatten()
            if isinstance(A, Patches):
                A = A.to_matrix(lower.shape)[0]
            As.append(-A[below])
            bs.append(c[below].to(b.device) - b[below])
    As, bs = torch.cat(As), torch.cat(bs)
    if As.shape[0] > 0:
        mid = (lower + upper) * 0.5
        dif = (upper - lower) * 0.5
        eps = torch.finfo(lower.dtype).eps * 3
        mask = (As * mid - As.abs() * dif).flatten(1).sum(1) + bs < -eps
        if not mask.all().item():
            # Some constraint boundaries might lie completely outside the bounding box
            As, bs = As[mask], bs[mask]
    if As.shape[0] > 0:
        return As, bs
    return None, None


def calc_samples(
    x: Samples | tuple[torch.Tensor, torch.Tensor],
    model: torch.nn.Module,
    num: int = 10_000,
    history: list[tuple[list[int], list[float]]] | None = None,
    debug: bool = IS_TEST_OR_DEBUG,
) -> Samples:
    """Generate uniform samples from the domain and compute their activations.

    Args:
        x: Previous samples or bounds for the input.
        model: Model to compute activations.
        num: Number of samples.
        history: Split history.
        debug: Enable additional asserts.

    Returns:
        samples: Generated samples.
    """
    with torch.no_grad():
        return get_samples(x, model, num, history, debug=debug)


def get_samples(
    x: Samples | tuple[torch.Tensor, torch.Tensor],
    model: torch.nn.Module,
    samples: int = 10_000,
    history: list[tuple[list[int], list[float]]] | None = None,
    debug: bool = IS_TEST_OR_DEBUG,
) -> Samples:
    """Generate uniform samples from the domain and compute their activations.

    Args:
        x: Previous samples or bounds for the input.
        model: Model to compute activations.
        samples: Number of samples.
        history: Split history.
        debug: Enable additional asserts.

    Returns:
        Generated samples.
    """
    model = WithActivations(model)
    if isinstance(x, Samples):
        x.X = _expand_patch(x.X, x.lower, x.mask)
        if x.activations is None:
            x.y, x.activations = model(x.X)
        if len(x) > samples * 8 // 10:
            return x.reuse()
        lower, upper, X, y, act = x.lower, x.upper, x.X, x.y, x.activations
        A, b, lAs, uAs, mask = x.A, x.b, x.lAs, x.uAs, x.mask
        if debug:
            assert (X >= lower).all().cpu().item()
            assert (X <= upper).all().cpu().item()
    else:
        lower, upper = x
        if debug:
            assert (lower <= upper).all().cpu().item()
        if lower.shape[0] != 1:
            lower = lower[:1]
            upper = upper[:1]
        A = b = lAs = uAs = X = y = act = None
        mask = (lower < upper)[0]
        if mask.count_nonzero() >= mask.numel() - 2:
            mask = None
    if history is None or sum((len(h[0]) for h in history), 0) == 0:
        # Without a domain we can just sample the bounding box
        if mask is not None:
            X = get_box_samples(lower[:, mask], upper[:, mask], samples * 5)
            X = _expand_patch(X, lower, mask)
        else:
            X = get_box_samples(lower, upper, samples * 5)
        y, act = model(X)
        return Samples(
            X, y, lower, upper, act, lAs=lAs, uAs=uAs, mask=mask, num=samples
        )
    if X is None or X.shape[0] < samples // 2:
        history = history_to_index(history)
        if A is not None:
            assert b is not None
            lower, upper = lower.clone(), upper.clone()
            if mask is not None:
                mlower = lower[:, mask]
                mupper = upper[:, mask]
                tighten_backwards(A, b, mlower, mupper, debug)
                if debug and X is not None:
                    con = polytope_contains(X[:, mask], A, b, mlower, mupper).all()
                    assert con.cpu().item()
                lower[:, mask] = mlower
                upper[:, mask] = mupper
            else:
                tighten_backwards(A, b, lower, upper, debug)
                if debug and X is not None:
                    assert polytope_contains(X, A, b, lower, upper).all().cpu().item()
        # Since ReLU splits are non-convex, we use rejection sampling.
        X, y, act = rejection_sample(
            X,
            y,
            act,
            A,
            b,
            lower,
            upper,
            model,
            samples,
            history,
            mask=mask,
            debug=debug,
        )
        if debug:
            check = True
            for (below, above), a in zip(history, act):
                if above:
                    check = (a.flatten(1)[:, above] >= 0).all(1) & check
                if below:
                    check = (a.flatten(1)[:, below] <= 0).all(1) & check
            assert check.all().cpu().item()
    return Samples(
        X, y, lower, upper, act, A=A, b=b, lAs=lAs, uAs=uAs, mask=mask, num=samples
    )


def get_box_samples(
    lower: torch.Tensor, upper: torch.Tensor, samples: int = 10_000
) -> torch.Tensor:
    """Generate uniform samples from a bounding box.

    Args:
        lower: Lower bound.
        upper: Upper bound.
        samples: Number of samples.

    Returns:
        X: samples.
    """
    X = torch.rand(samples, lower.numel(), dtype=lower.dtype, device=upper.device)
    X = lower[None] + X.view(samples, *lower.shape) * (upper - lower)[None]
    return torch.squeeze(X, dim=1)


def get_hit_and_run_samples(
    X: torch.Tensor,
    A: torch.Tensor,
    b: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    samples: int,
    batch: int = 128,
    steps: int = 5,
) -> torch.Tensor | None:
    """Generate uniform samples from a polytope.

    Args:
        X: Previous samples
        A: Polytope constraints coefficients.
        b: Polytope constraints biases.
        lower: Lower bound.
        upper: Upper bound.
        samples: Number of samples.
        batch: Batch size.
        steps: Thinning steps.

    Returns:
        Uniform samples from a polytope.
    """
    batch = min(batch, samples)
    if X.shape[0] < batch:
        reps = (X.shape[0] - 1) // batch + 1
        X = torch.repeat_interleave(X, reps, 0)
        x = X[:batch]
    else:
        x = X[torch.randperm(X.shape[0])[-batch:]]
    count = 0
    resets = 0
    X = X.new_empty((samples, *lower.shape[1:]))
    while resets < 10:
        xn = _hit_and_run(x, A, b, lower, upper, steps)
        xn = xn[polytope_contains(xn, A, b, lower, upper)]
        X[count : count + xn.shape[0]] = xn
        count += xn.shape[0]
        if count >= samples - batch // 2:
            break
        if xn.shape[0] < batch:
            x = X[torch.randperm(count)[: min(batch, samples - count)]]
            resets += 1
        else:
            x = xn[: samples - count]
    return X[:count]


def _hit_and_run(
    X: torch.Tensor,
    A: torch.Tensor,
    b: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    steps: int = 5,
) -> torch.Tensor | None:
    shape = [X.shape[0]] + [1] * (len(X.shape) - 1)
    for _ in range(steps):
        dir = torch.normal(0.0, 1.0, X.shape, dtype=X.dtype, device=X.device)
        # Reflection when x is at the bounding box to avoid sampling issues
        reflect = ((dir >= 0) | (X > lower)) & ((dir <= 0) | (X < upper))
        dir = torch.where(reflect, dir, -dir)
        # Handle locked dimensions
        dir = torch.where(lower == upper, 0.0, dir)
        # Normalize
        dir = dir / dir.abs().flatten(1).max(1)[0].view(shape)
        back, forward = raycast_batch(
            X, dir, A[None], b[None], lower[None], upper[None], verify=False
        )
        sample = torch.rand(forward.shape, device=X.device, dtype=X.dtype)
        X = X + dir * (sample * (forward - back) + back).view(shape)
    return X


def rejection_sample(
    X: torch.Tensor | None,
    y: torch.Tensor | None,
    act: list[torch.Tensor] | None,
    A: torch.Tensor | None,
    b: torch.Tensor | None,
    lower: torch.Tensor,
    upper: torch.Tensor,
    model: WithActivations | torch.nn.Module,
    samples: int,
    history: list[tuple[list[int], list[int]]],
    mask: torch.Tensor | None = None,
    max_iter: int = 20,
    debug: bool = IS_TEST_OR_DEBUG,
) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
    """Rejection sampling for a ReLU split domain.

    Args:
        X: Previous samples.
        y: Previous predictions,
        act: Previous activations.
        A: Polytope constraints coefficients.
        b: Polytope constraints bias.
        lower: Lower bounds.
        upper: Upper bounds.
        model: The model to compute the activations.
        samples: The number of samples to return.
        history: ReLU split history.
        patch_mask: Mask if input specification is a patch.
        max_iter: The number of batches to try.
        debug: Activate additional asserts. Defaults to False unless debugging.

    Returns:
        X: The sampled points.
        y: The labels of the sampled points.
        activations: The activations of the sampled points.
    """
    if X is None:
        count = 0
        Xs = []
        ys = []
        acts = []
        old_X = lower.new_tensor([])
    else:
        assert y is not None and act is not None
        count = X.shape[0]
        if count >= samples * 10 // 8:
            return (X, y, act)
        Xs = [X]
        old_X = X
        ys = [y]
        acts = [act]
    model = model if isinstance(model, WithActivations) else WithActivations(model)
    hit_and_run = False
    attempts = 0
    if mask is not None:
        old_X = old_X[:, mask]
        patch_x = lower
        lower = lower[:, mask]
        upper = upper[:, mask]
    while attempts < max_iter:
        n = max(50, min(samples * 3 // 2, (samples - count) * 2))
        if hit_and_run:
            X = get_hit_and_run_samples(old_X, A, b, lower, upper, n)
            attempts += 4
        else:
            X = get_box_samples(lower, upper, n)
            X = X[polytope_contains(X, A, b)]
            hit_and_run = X.shape[0] < n // 100
            attempts += 1
        if X is None or X.shape[0] == 0:
            continue
        old_X = X if X.shape[0] > 100 else torch.cat((old_X, X))
        if mask is not None:
            X = _expand_patch(X, patch_x, mask)
        y, act = model(X)
        inside = split_contains(history, act)
        X = X[inside]
        if X.shape[0] == 0:
            continue
        count += X.shape[0]
        Xs.append(X.contiguous())
        if X.shape[0] == y.shape[0]:
            ys.append(y)
            acts.append(act)
        else:
            ys.append(y[inside])
            acts.append([a[inside].contiguous() for a in act])
        if count >= samples * (2 - hit_and_run):
            break
    if count == 0:
        raise Exception("Unable to sample even a single point")
    return torch.cat(Xs, 0), torch.cat(ys, 0), [torch.cat(act, 0) for act in zip(*acts)]
