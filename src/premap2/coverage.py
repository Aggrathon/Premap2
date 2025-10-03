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
    """Calculate the approximation ratio (coverage) of the preimage after the initial iteration.

    Args:
        A_b_dict: Dictionary containing the LiRPA linear bounds.
        c: Output specification as a linear tensor.
        samples: Samples.
        under: Is this an under approximation.
        debug: Run extra asserts. Defaults to False.

    Returns:
        preimage_volume: Preimage volume estimate.
        approx_volume: Approximation volume estimate.
        A_b_dict: Flattened dictionary containing the LiRPA linear bounds.
    """
    Ab_dict = {
        k: v.squeeze(1) if k in ("uA", "lA") else v
        for output in A_b_dict.values()
        for input in output.values()
        for k, v in input.items()
        if v is not None
    }
    A = Ab_dict["lA" if under else "uA"]
    bias = Ab_dict["lbias" if under else "ubias"]
    preimg, verified = _calc_coverage(samples, A, bias, c, under, debug)
    if debug:
        assert preimg >= verified if under else verified >= preimg
        assert preimg <= len(samples) and verified <= len(samples)
    return preimg / len(samples), verified / len(samples), Ab_dict


def calc_branched_coverage(
    A_b_dict: dict[str, dict[str, dict[str, torch.Tensor]]],
    samples: list[Samples],
    domains: list[ADomain],
    under: bool = True,
    debug: bool = IS_TEST_OR_DEBUG,
) -> tuple[list[tuple[float, float, float]], dict[str, torch.Tensor]]:
    """Calculate the approximation ratio of the preimage.

    Args:
        A_b_dict: Dictionary containing the LiRPA linear bounds.
        samples: Samples.
        domains: Branched domains.
        under: Is this an under approximation.
        debug: Run extra asserts. Defaults to False.

    Returns:
        subdomain_info: List of (`preimage_volume`, `approx_volume`, `domain_volume`) for each domain.
        A_b_dict: Flattened dictionary containing the LiRPA linear bounds.
    """
    Ab_dict = {
        k: v
        for output in A_b_dict.values()
        for input in output.values()
        for k, v in input.items()
    }
    left_info = []
    right_info = []
    for left_A, left_b, right_A, right_b, left, right, domain in zip(
        Ab_dict["lA" if under else "uA"][: len(domains)],
        Ab_dict["lbias" if under else "ubias"][: len(domains)],
        Ab_dict["lA" if under else "uA"][len(domains) :],
        Ab_dict["lbias" if under else "ubias"][len(domains) :],
        samples[: len(domains)],
        samples[len(domains) :],
        domains,
    ):
        scale = domain.volume / (len(left) + len(right))
        l_pre, l_app = _calc_coverage(left, left_A, left_b, domain.c, under, debug)
        left_info.append((l_pre * scale, l_app * scale, len(left) * scale))
        r_pre, r_app = _calc_coverage(right, right_A, right_b, domain.c, under, debug)
        right_info.append((r_pre * scale, r_app * scale, len(right) * scale))

        if debug:
            assert abs(domain.preimg_vol - (r_pre + l_pre) * scale) < 1e-4
            if under:
                assert len(left) >= l_pre >= l_app
                assert len(right) >= r_pre >= r_app
            else:
                assert len(left) >= l_app >= l_pre
                assert len(right) >= r_app >= r_pre
            # This assert should just be a warning, a decreasing approximation volume is an issue, but not invalid
            # new_vol = left_vol * left_cov + right_vol * right_cov
            # if under:
            #     assert domain.preimg_vol * domain.preimg_cov <= new_vol + 1e-4
            # else:
            #     assert domain.preimg_vol * domain.preimg_cov >= new_vol - 1e-4
    return left_info + right_info, Ab_dict


def _calc_coverage(
    samples: Samples,
    A: torch.Tensor,
    bias: torch.Tensor,
    c: torch.Tensor,
    under: bool = True,
    debug: bool = IS_TEST_OR_DEBUG,
) -> tuple[int, int]:
    c = c.squeeze(0)
    output = torch.einsum("o...,n...->no", c, samples.y)
    preimage = (output > 0).all(1).count_nonzero().cpu().item()
    if not debug and (preimage == 0 if under else preimage == len(samples)):
        return preimage, preimage
    approx = torch.einsum("o...,n...->no", A, samples.X) + bias.view(1, -1)
    verified = (approx >= 0).all(1).count_nonzero().cpu().item()
    if debug:
        if under:
            assert (approx <= output).all().item()
        else:
            assert (approx >= output).all().item()
    return preimage, verified
