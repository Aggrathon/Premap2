import numpy as np
import pytest
import torch

from premap2.utils import assert_bounds, is_int


def test_int():
    assert is_int(1)
    assert is_int(np.ones((), dtype=int))
    assert is_int(np.ones((), dtype=np.uint8))
    assert is_int(np.ones((1,), dtype=np.int32))
    assert is_int(torch.ones((), dtype=torch.int32))
    assert is_int(torch.ones((1,), dtype=torch.uint8))
    assert not is_int(1.0)
    assert not is_int(np.ones(()))
    assert not is_int(np.ones((3,), dtype=int))
    assert not is_int(torch.ones(()))
    assert not is_int(torch.ones((3,), dtype=torch.uint8))


def test_bounds():
    assert_bounds(torch.rand((3,)), 0.0, 1.0)
    assert_bounds(torch.rand((3,)), torch.zeros(()), torch.ones(()))
    assert_bounds(torch.rand((3,)), torch.zeros((3,)), torch.ones((3,)))
    assert_bounds([torch.rand((3,))], [0.0], [1.0])
    assert_bounds([torch.rand((3,))], [0.0], torch.ones((1,)))
    with pytest.raises(AssertionError):
        assert_bounds(torch.rand((3,)), 1.0, 0.0)
