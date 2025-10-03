from dataclasses import dataclass

import numpy as np
import torch

from premap2.domains import PriorityDomains, sizeof
from premap2.sampling import Samples, get_samples
from tests.premap2.utils import model_linear


@dataclass
class DummyDomain:
    samples: Samples | None
    priority: float


def test_priority_domains():
    # Test that the priority queue works as expected
    model = model_linear(100, 50, 10, 5)
    samples = get_samples((torch.zeros(1, 100), torch.ones(1, 100)), model, 100)
    size = sizeof(DummyDomain(samples, 0.0))
    domains = PriorityDomains(
        reduce_size=size, reduce_start=2, store_size=size * 2, store_start=3
    )
    for i in [1.0, -np.inf, 2.0, 0.0, -1.0, 5.0, -np.inf, 3.0, -2.0]:
        domains.add(DummyDomain(samples=samples, priority=i))  # type: ignore
    assert domains.pop().priority == 5.0
    assert len(domains) == len(list(domains.iter_final()))
