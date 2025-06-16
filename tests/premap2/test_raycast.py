import torch

from premap2.raycast import raycast, raycast_batch

from .utils import is_close


def test_raycast():
    def raycasts(x, d, back, forward, poly, eps=1e-8):
        for x_, d_, b_, f_ in zip(x, d, back, forward):
            b, f = raycast(x_, d_, *poly, verify=True)
            assert is_close((b, f), (b_, f_), eps)
        a = tuple(None if x is None else x[None] for x in poly)
        b, f = raycast_batch(x, d, *a, verify=True)
        assert is_close(b, back, eps)
        assert is_close(f, forward, eps)

    poly = [None, None, torch.zeros(2), torch.ones(2)]
    raycasts(
        torch.tensor([[0.0, 0], [-1, 0], [-1, 0]]),
        torch.tensor([[1.0, 0], [1, 0], [2, 0]]),
        torch.tensor([0.0, 1.0, 0.5]),
        torch.tensor([1.0, 2.0, 1.0]),
        poly,
    )
    poly[0] = torch.tensor([[-1.0, -1.0]])
    poly[1] = torch.tensor([1.0])
    raycasts(
        torch.tensor([[0.5, 0], [0.5, 1], [0.5, 1]]),
        torch.tensor([[0.0, 1], [0, -1], [0, -0.5]]),
        torch.tensor([0.0, 0.5, 1.0]),
        torch.tensor([0.5, 1.0, 2.0]),
        poly,
        2e-5,
    )
    poly[0] = torch.cat((poly[0], torch.tensor([[1.0, 1.0]])))
    poly[1] = torch.cat((poly[1], torch.tensor([-0.8])))
    raycasts(
        torch.tensor([[0.5, 1], [0.5, 0.5]]),
        torch.tensor([[0.0, -1], [1, 1]]),
        torch.tensor([0.5, -0.1]),
        torch.tensor([0.7, 0.0]),
        poly,
        2e-5,
    )
    back, forward = raycast_batch(
        torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        -torch.tensor([[0.0, 1.0]]),
        torch.stack((torch.zeros_like(poly[0][:1]), poly[0][:1])),
        torch.stack((torch.zeros_like(poly[1][:1]), poly[1][:1])),
        torch.stack((poly[2], poly[2])),
        torch.stack((poly[3], poly[3])),
    )
    assert is_close(back, torch.tensor([-1.0, 0.0]), 2e-5)
    assert is_close(forward, torch.tensor([0.0, 1.0]), 2e-5)
