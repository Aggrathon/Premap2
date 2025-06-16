from dataclasses import dataclass

import numpy as np
import torch

from premap2.coverage import calc_branched_coverage, calc_initial_coverage
from premap2.sampling import Samples


@dataclass
class Domain:
    preimg_A: torch.Tensor
    preimg_b: torch.Tensor
    preimg_vol: float
    c: torch.Tensor
    volume: float = 1.0


def test_calc_initial_coverage():
    """Test calc_initial_coverage with a simple case."""
    lA = torch.randn(2, 2)
    uA = torch.randn(2, 2)
    c = torch.eye(2)
    samples = Samples(
        torch.randn(10, 2), torch.randn(10, 2), torch.zeros(2), torch.ones(2)
    )
    lbias = (samples.y - torch.einsum("oi,ni->no", lA, samples.X)).min(0)[0] - 1e-6
    ubias = (samples.y - torch.einsum("oi,ni->no", uA, samples.X)).max(0)[0] + 1e-6
    A_b_dict = {"": {"": {"lA": lA, "lbias": lbias, "uA": uA, "ubias": ubias}}}
    assert (samples.y <= torch.einsum("oi,ni->no", uA, samples.X) + ubias[None]).all()
    assert (samples.y >= torch.einsum("oi,ni->no", lA, samples.X) + lbias[None]).all()

    preimg = (samples.y >= 0).all(1).float().mean().cpu().item()
    cov = (samples.X @ lA.T + lbias[None] >= 0).all(1).float().mean().cpu().item()
    assert preimg >= cov
    preimage_vol, coverage, _ = calc_initial_coverage(A_b_dict, c, samples, under=True)
    assert np.allclose(preimage_vol, preimg)
    if preimg > 0:
        print(preimg, preimage_vol, coverage, cov)
        assert np.allclose(coverage, cov / preimg)

    preimg = (samples.y >= 0).all(1).float().mean().cpu().item()
    cov = (samples.X @ uA.T + ubias[None] >= 0).all(1).float().mean().cpu().item()
    assert cov >= preimg
    preimage_vol, coverage, _ = calc_initial_coverage(A_b_dict, c, samples, False)
    assert np.allclose(preimage_vol, preimg)
    if preimg > 0:
        print(preimg, preimage_vol, coverage, cov)
        assert np.allclose(coverage, cov / preimg)


def test_calc_branched_coverage():
    """Test calc_branched_coverage with a basic scenario."""
    lA = torch.randn(2, 1, 2)
    lbias = torch.randn(2, 1)
    A_b_dict = {"": {"": {"lA": lA, "lbias": lbias}}}
    samples = [
        Samples(torch.randn(10, 2), torch.randn(10, 1), torch.zeros(2), torch.ones(2)),
        Samples(torch.randn(10, 2), torch.randn(10, 1), torch.zeros(2), torch.ones(2)),
    ]
    for s in samples:
        yb = (torch.einsum("boi,ni->bno", lA, s.X) + lbias[:, None]).max(0)[0]
        assert yb.shape == (10, 1)
        s.y = s.y + (yb - s.y).max(0)[0][None] + 1e-6
        assert torch.all(s.y >= yb).cpu().item()
        assert s.y.shape == (10, 1)
    preimg = torch.cat([s.y >= 0 for s in samples]).float().mean().cpu().item()
    domains = [Domain(lA[0], lbias[0], preimg, torch.ones(1, 1, 1))]

    subdomain_info, _ = calc_branched_coverage(A_b_dict, samples, domains, True)
    for i, (s, info) in enumerate(zip(samples, subdomain_info)):
        A_b_dict = {"": {"": {"lA": lA[i], "lbias": lbias[i]}}}
        preimage_vol, coverage, _ = calc_initial_coverage(
            A_b_dict, torch.ones(1, 1, 1), s
        )
        assert np.allclose(info[0], preimage_vol * 0.5)
        assert np.allclose(info[1], coverage)
        assert np.allclose(info[2], 0.5)
