import numpy as np
import torch

from auto_LiRPA.bound_general import BoundedModule


def is_close(a: float | list[float], b: float | list[float], eps: float = 1e-6) -> bool:
    if torch.is_tensor(a):
        if torch.is_tensor(b):
            return torch.allclose(a, b, atol=eps)
        a = a.detach().cpu().numpy()
    if torch.is_tensor(b):
        b = b.detach().cpu().numpy()
    return np.allclose(a, b, atol=eps)


def model_linear(*sizes: int) -> torch.nn.Sequential:
    return torch.nn.Sequential(
        *[
            m
            for i, o in zip(sizes[:-1], sizes[1:])
            for m in (torch.nn.Linear(i, o), torch.nn.ReLU())
        ][:-1]
    )


def model_conv(*sizes: int, kernel: int = 3, size: int = 5) -> torch.nn.Sequential:
    layers = []
    for i, o in zip(sizes[:-3], sizes[1:]):
        layers.append(torch.nn.Conv2d(i, o, kernel, 1, 1))
        layers.append(torch.nn.ReLU())
    layers.append(torch.nn.Conv2d(sizes[-3], sizes[-2], kernel, 1, 0))
    layers.append(torch.nn.ReLU())
    layers.append(torch.nn.Flatten())
    layers.append(torch.nn.Linear(sizes[-2] * (size - 2) ** 2, sizes[-1]))
    return torch.nn.Sequential(*layers)


def get_intermediate_bounds(
    lirpa: BoundedModule,
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    bounds = {}
    for node in (r.inputs[0] for r in lirpa.relus):
        if not hasattr(node, "lower") or not hasattr(node, "upper"):
            lirpa.compute_intermediate_bounds(node)
        lower = node.lower.detach().clone() if node.lower is not None else None
        upper = node.upper.detach().clone() if node.upper is not None else None
        bounds[node.name] = (lower, upper)
    return bounds
