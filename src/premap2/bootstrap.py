import numpy as np
import torch

from premap2.utils import IS_TEST_OR_DEBUG


def bootstrap_split(
    mask: torch.Tensor,
    log_prob: torch.Tensor | None,
    num: int = 1000,
    *,
    debug: bool = IS_TEST_OR_DEBUG,
) -> tuple[torch.Tensor, torch.Tensor]:
    n = len(mask)
    samples = torch.randint(0, n, (num, n), device=mask.device)
    if log_prob is None:
        total = samples.new_tensor(n).log()
        left = mask[samples].sum(1)
        if debug:
            assert left.shape == (num,)
        return left.log() - total, (n - left).log() - total
    else:
        probs = log_prob[samples]
        total = torch.logsumexp(probs, 1)
        inf = probs.new_tensor(-np.inf)
        left = torch.logsumexp(torch.where(mask[samples], probs, inf), 1) - total
        right = torch.logsumexp(torch.where(~mask[samples], probs, inf), 1) - total
        if debug:
            assert total.shape == (num,) == left.shape == right.shape
            assert ((left.exp() + right.exp() - 1.0).abs() < 1e-5).all().item()
        return left, right


def bootstrap_select(
    mask1: torch.Tensor,
    mask2: torch.Tensor,
    log_prob: torch.Tensor | None = None,
    num: int = 1000,
    *,
    debug: bool = IS_TEST_OR_DEBUG,
) -> tuple[torch.Tensor, torch.Tensor]:
    n = len(mask1)
    assert len(mask2) == n
    samples = torch.randint(0, n, (num, n), device=mask1.device)
    if log_prob is None:
        total = samples.new_tensor(n).log()
        nz1 = mask1[samples].sum(1).log()
        nz2 = mask2[samples].sum(1).log()
        if debug:
            assert nz1.shape == (num,) == nz2.shape
        return nz1 - total, nz2 - total
    else:
        probs = log_prob[samples]
        total = torch.logsumexp(probs, 1)
        inf = probs.new_tensor(-np.inf)
        prob1 = torch.logsumexp(torch.where(mask1[samples], probs, inf), 1)
        prob2 = torch.logsumexp(torch.where(mask2[samples], probs, inf), 1)
        if debug:
            assert total.shape == (num,) == prob1.shape == prob2.shape
        return prob1 - total, prob2 - total


def confidence_interval(
    values: torch.Tensor, confidence: float = 0.9
) -> tuple[float, float]:
    """Calculate bootstrap confidence intervals using quantiles.

    Args:
        values: Bootstrap values.
        confidence: Interval width. Defaults to 0.9.

    Returns:
        The bounds of the confidence interval.
    """
    interval = (1.0 - confidence) * 0.5
    qs = torch.quantile(values, values.new_tensor([interval, 1.0 - interval]))
    low, high = qs[0].item(), qs[1].item()
    if np.isnan(low) or np.isnan(high):
        if qs.max().item == -np.inf:
            return 0.0, 0.0
    return low, high
