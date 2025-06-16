from typing import Protocol

import torch

from .sampling import Samples
from .utils import IS_TEST_OR_DEBUG


class ADomain(Protocol):
    preimg_A: torch.Tensor
    preimg_b: torch.Tensor
    preimg_vol: float
    volume: float
    c: torch.Tensor


def calc_initial_coverage(
    A_b_dict: dict[str, dict[str, dict[str, torch.Tensor | None]]],
    c: torch.Tensor,
    samples: Samples,
    under: bool = True,
    debug: bool = IS_TEST_OR_DEBUG,
) -> tuple[float, float, dict[str, torch.Tensor]]:
    """Calculate the coverage of the preimage approximation after the initial iteration.

    Args:
        A_b_dict: Dictionary containing the LiRPA linear bounds.
        c: Output specification as a linear tensor.
        samples: Samples.
        under: Is this an under approximation.
        debug: Run extra asserts. Defaults to False.

    Returns:
        preimage_volume: Volume estimate.
        coverage: Approximation coverage.
        A_b_dict: Flattened dictionary containing the LiRPA linear bounds.
    """
    A_b_dict = {
        k: v.squeeze(1) if k in ("uA", "lA") else v
        for output in A_b_dict.values()
        for input in output.values()
        for k, v in input.items()
        if v is not None
    }
    A = A_b_dict["lA" if under else "uA"]
    bias = A_b_dict["lbias" if under else "ubias"]
    preimg, verified = _calc_coverage(samples, A, bias, c, None, None, under, debug)
    if preimg > 0:
        return preimg / len(samples), verified / preimg, A_b_dict
    else:
        return 0.0, 0.0, A_b_dict


def calc_branched_coverage(
    A_b_dict: dict[str, dict[str, dict[str, torch.Tensor]]],
    samples: list[Samples],
    domains: list[ADomain],
    under: bool = True,
    debug: bool = IS_TEST_OR_DEBUG,
) -> tuple[list[tuple[float, float, float]], dict[str, torch.Tensor]]:
    """Calculate the preimage approximation coverage.

    Args:
        A_b_dict: Dictionary containing the LiRPA linear bounds.
        samples: Samples.
        domains: Branched domains.
        under: Is this an under approximation.
        debug: Run extra asserts. Defaults to False.

    Returns:
        subdomain_info: List of (`preimage_volume`, `coverage`, `domain_volume`) for each domain.
        A_b_dict: Flattened dictionary containing the LiRPA linear bounds.
    """
    A_b_dict = {
        k: v
        for output in A_b_dict.values()
        for input in output.values()
        for k, v in input.items()
    }
    left_subdomain_info = []
    right_subdomain_info = []
    for left_A, left_b, right_A, right_b, left, right, domain in zip(
        A_b_dict["lA" if under else "uA"][: len(domains)],
        A_b_dict["lbias" if under else "ubias"][: len(domains)],
        A_b_dict["lA" if under else "uA"][len(domains) :],
        A_b_dict["lbias" if under else "ubias"][len(domains) :],
        samples[: len(domains)],
        samples[len(domains) :],
        domains,
    ):
        num_total = len(left) + len(right)

        left_volume = len(left) / num_total * domain.volume
        left_preimg, left_verified = _calc_coverage(
            left,
            left_A,
            left_b,
            domain.c,
            domain.preimg_A,
            domain.preimg_b,
            under,
            debug,
        )
        if left_preimg > 0:
            left_vol = left_preimg / num_total * domain.volume
            left_cov = left_verified / left_preimg
        else:
            left_vol = left_cov = 0.0
        left_subdomain_info.append((left_vol, left_cov, left_volume))

        right_volume = len(right) / num_total * domain.volume
        right_preimg, right_verified = _calc_coverage(
            right,
            right_A,
            right_b,
            domain.c,
            domain.preimg_A,
            domain.preimg_b,
            under,
            debug,
        )
        if right_preimg > 0:
            right_vol = right_preimg / num_total * domain.volume
            right_cov = right_verified / right_preimg
        else:
            right_vol = right_cov = 0.0
        right_subdomain_info.append((right_vol, right_cov, right_volume))

        if debug:
            assert abs(domain.preimg_vol - left_vol - right_vol) < 1e-4
            # This assert should just be a warning, a decreasing approximation volume is an issue, but not invalid
            # new_vol = left_vol * left_cov + right_vol * right_cov
            # if under:
            #     assert domain.preimg_vol * domain.preimg_cov <= new_vol + 1e-4
            # else:
            #     assert domain.preimg_vol * domain.preimg_cov >= new_vol - 1e-4
    return left_subdomain_info + right_subdomain_info, A_b_dict


def _calc_coverage(
    samples: Samples,
    A: torch.Tensor,
    bias: torch.Tensor,
    c: torch.Tensor,
    prevA: None | torch.Tensor = None,
    prevb: None | torch.Tensor = None,
    under: bool = True,
    debug: bool = IS_TEST_OR_DEBUG,
) -> tuple[int, int]:
    c = c.squeeze(1)
    preimg_mask = (torch.einsum("o...,n...->no", c, samples.y) > 0).all(1)
    preimg_vol = preimg_mask.count_nonzero().cpu().item()
    if (under and preimg_vol == 0) or (not under and preimg_vol == len(samples)):
        return preimg_vol, preimg_vol
    else:
        approx = torch.einsum("o...,n...->no", A, samples.X) + bias.view(1, -1)
        verified = (approx >= 0).all(1).count_nonzero().cpu().item()
        if debug:
            out = torch.einsum("o...,n...->no", c, samples.y)
            if under:
                assert (approx <= out).all().item()
            else:
                assert (approx >= out).all().item()
            # This assert should just be a warning, a decreasing approximation volume is an issue, but not invalid
            # if prevA is not None and prevb is not None:
            #     approx2 = torch.einsum("o...,n...->no", prevA, X)
            #     approx2 += torch.atleast_2d(prevb)
            #     assert verified >= (approx2 >= 0).all(1).count_nonzero().cpu().item()
        return preimg_vol, verified
