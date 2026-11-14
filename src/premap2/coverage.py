import numpy as np
import torch

from premap2.sampling import Domain
from premap2.utils import IS_TEST_OR_DEBUG


def calc_coverage(
    A_b_dict: dict[str, dict[str, dict[str, torch.Tensor]]],
    domains: list[Domain],
    output: torch.Tensor,
    threshold: torch.Tensor,
    under: bool,
    *,
    debug: bool = IS_TEST_OR_DEBUG,
):
    """Calculate the approximation ratio of the preimage.

    Args:
        A_b_dict: Dictionary containing the LiRPA linear bounds.
        domains: Branched domains.
        output: Linear output layer.
        threshold: Output threshold.
        debug: Run extra asserts. Defaults to False.
    """
    Ab_dict = {
        k: v
        for output in A_b_dict.values()
        for input in output.values()
        for k, v in input.items()
    }
    for A, b, domain in zip(
        Ab_dict["lA" if under else "uA"],
        Ab_dict["lbias" if under else "ubias"],
        domains,
    ):
        old_pri = domain.priority
        old = (domain.preimg_A, domain.preimg_b, domain.log_preimg, domain.log_approx)
        domain.preimg_A = A
        domain.preimg_b = b
        update_volumes(domain, output, threshold, under, debug=debug)
        if old_pri < domain.priority:
            domain.preimg_A, domain.preimg_b, domain.log_preimg, domain.log_approx = old
            domain.priority = old_pri


def update_volumes(
    domain: Domain,
    output: torch.Tensor,
    threshold: torch.Tensor,
    under: bool,
    *,
    debug: bool = IS_TEST_OR_DEBUG,
):
    """Update `domain.preimg_vol`, `domain.approx_vol` and `domain.priority`.

    Args:
        domain: Domain.
        output: Linear output layer.
        threshold: Output threshold.
        under: Is this an under approximation.
        debug: Enable additional asserts.
    """
    if domain.log_volume == -np.inf:
        domain.log_preimg = domain.log_approx = -np.inf
        domain.priority = -np.inf
        return

    A, bias = domain.preimg_A, domain.preimg_b
    pred = torch.einsum("o...,n...->no", output, domain.y)
    approx = torch.einsum("o...,n...->no", A, domain.X) + bias.view(1, -1)  # type: ignore
    if domain.log_prob is not None:
        preimage = torch.logsumexp(domain.log_prob[(pred >= threshold).all(1)], 0)
        verified = torch.logsumexp(domain.log_prob[(approx >= threshold).all(1)], 0)
        preimage, verified = preimage.item(), verified.item()
    else:
        with np.errstate(divide="ignore"):
            preimage = np.log((pred >= threshold).all(1).count_nonzero().item())
            verified = np.log((approx >= threshold).all(1).count_nonzero().item())
    total = domain.log_size()
    domain.log_preimg = domain.log_volume + preimage - total
    domain.log_approx = domain.log_volume + verified - total

    if debug:
        assert pred.shape == approx.shape
        if under:
            assert (approx <= pred + 1e-5).all().item()
            assert total + 1e-6 >= preimage >= verified
        else:
            assert (approx >= pred - 1e-5).all().item()
            assert preimage <= verified <= total + 1e-6

    if (
        (domain.log_preimg <= -18.0 if under else preimage >= total)
        or abs(verified - preimage) <= 1e-6
        or domain.log_volume <= -18.0
    ):
        domain.priority = -np.inf
    else:
        domain.priority = abs(domain.approx_vol - domain.preimg_vol)
