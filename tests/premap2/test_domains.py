from dataclasses import dataclass

import torch

from premap2.domains import PriorityDomains
from premap2.sampling import Samples, get_samples
from tests.premap2.utils import model_linear


@dataclass
class DummyDomain:
    samples: Samples | None
    priority: float


def test_priority_domains():
    # Test that the priority queue works as expected
    domains = PriorityDomains(
        reduce_size=0, reduce_start=2, store_size=0, store_start=4
    )
    model = model_linear(100, 50, 10, 5)
    for i in range(6):
        samples = get_samples((torch.zeros(1, 100), torch.ones(1, 100)), model, 100)
        domain = DummyDomain(samples=samples, priority=i)
        domains.add(domain)
    assert domains.pop().priority == 5
    domains.iter_final()
