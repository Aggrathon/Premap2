from dataclasses import dataclass

import torch

from premap2.coverage import calc_coverage
from premap2.sampling import Domain


@dataclass
class ADomain:
    preimg_A: torch.Tensor
    preimg_b: torch.Tensor
    preimg_vol: float
    c: torch.Tensor
    volume: float = 1.0


def test_calc_coverage():
    """Test calc_branched_coverage with a basic scenario."""
    batch = 4
    n = 10
    lA = torch.randn(batch, 1, 2)
    lbias = torch.randn(batch, 1)
    A_b_dict = {"": {"": {"lA": lA, "lbias": lbias}}}
    domains = [
        Domain(torch.zeros(2), torch.ones(2), torch.randn(n, 2), torch.randn(n, 1))
        for _ in range(batch)
    ]
    for i, s in enumerate(domains):
        yb = torch.einsum("oi,ni->no", lA[i], s.X) + lbias[i, None]
        assert yb.shape == (n, 1)
        s.y = s.y + (yb - s.y).max(0)[0][None] + 1e-6
        assert torch.all(s.y >= yb).cpu().item()
        assert s.y.shape == (n, 1)

    calc_coverage(A_b_dict, domains, torch.eye(1), torch.zeros(1), True, debug=True)

    for i, s in enumerate(domains):
        assert torch.equal(s.preimg_A, lA[i])
        assert torch.equal(s.preimg_b, lbias[i])

    for s in domains:
        s.log_prob = torch.rand(n)
    calc_coverage(A_b_dict, domains, torch.eye(1), torch.zeros(1), True, debug=True)
