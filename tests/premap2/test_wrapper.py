from warnings import warn

import numpy as np
import torch

from premap2.utils import result_contains
from premap2.wrapper import PremapInPath, construct_config, get_arguments, premap

from .utils import is_close, model_conv, model_linear, temp_seed


def test_construct_config():
    def post(config):
        assert config["debug"]["asserts"]

    PremapInPath("tmp")
    with PremapInPath():
        construct_config(False, post, {"asserts": True})
        construct_config(False, post, asserts=True)
        construct_config(False, post, {"asserts": False}, asserts=True)

    premap(help=True)
    get_arguments(False)
    get_arguments(True, command_line=True)


def test_wrapper_fc():
    for i in range(10):
        with temp_seed(42 + i):
            model = model_linear(20, 10, 1).eval()
            x = torch.randn((1, 20))
        with torch.no_grad():
            list(model.parameters())[-1] *= 2.0
            list(model.parameters())[-1] -= model(x).ravel()
        res = premap(
            dataset=[x, 0, 1.0, -1.0],
            model=model,
            threshold=0.99,
            spec_type="bound",
            num_outputs=1,
            sample_num=100,
            branch_budget=10,
            sample_instability=False,
            asserts=True,
            ci=0.9,
        )
        result = torch.load(res[0])
        assert np.isfinite(result["approx_ci"]).all()
        res[0].seek(0)  # type: ignore
        assert torch.allclose(
            result_contains(x, result, model),
            result_contains(x, res[0], model),
        )
        if len(result["domains"]) > 1:
            res = premap(
                dataset=[x, 0, 1.0, -1.0],
                model=model,
                spec_type="bound",
                num_outputs=1,
                sample_num=100,
                branch_budget=3,
                sample_instability=True,
                asserts=True,
                silent=True,
            )
            return
    warn("test_wrapper_fc: Could not find a case that needed more than one domain")


def test_wrapper_fco():
    for i in range(10):
        with temp_seed(42 + i):
            model = model_linear(20, 10, 1).eval()
            x = torch.randn((1, 20))
        with torch.no_grad():
            list(model.parameters())[-1] *= 2.0
            list(model.parameters())[-1] -= model(x).ravel()
        res = premap(
            dataset=[x, 0, 1.0, -1.0],
            model=model,
            threshold=0.99,
            spec_type="bound",
            num_outputs=1,
            sample_num=100,
            branch_budget=3,
            sample_instability=True,
            asserts=True,
            under_approx=False,
            over_approx=True,
        )
        result = torch.load(res[0])
        res[0].seek(0)  # type: ignore
        assert torch.allclose(
            result_contains(x, result, model),
            result_contains(x, res[0], model),
        )
        if len(result["domains"]) > 1:
            return
    warn("test_wrapper_fco: Could not find a case that needed more than one domain")


def test_wrapper_mo(tmp_path):
    for i in range(10):
        with temp_seed(43 + i):
            model = model_linear(20, 10, 3).eval()
            x = torch.randn((1, 20))
        with torch.no_grad():
            list(model.parameters())[-1] *= 2.0
            list(model.parameters())[-1] -= model(x).ravel()
        res = premap(
            dataset=[x, 0, 1.0, -1.0],
            model=model,
            threshold=0.99,
            result_dir=tmp_path,
            spec_type="bound",
            robustness_type="verified-acc",
            num_outputs=3,
            sample_num=100,
            branch_budget=10,
            sample_instability=True,
            asserts=True,
        )
        result = torch.load(res[0])
        if len(result["domains"]) > 1:
            return
    warn("test_wrapper_mo: Could not find a case that needed more than one domain")


def test_wrapper_conv():
    for i in range(10):
        with temp_seed(42 + i):
            model = model_conv(3, 4, 5, 4, 1).eval()
            x = torch.randn((1, 3, 5, 5))
        with torch.no_grad():
            list(model.parameters())[-1] *= 2.0
            list(model.parameters())[-1] -= model(x).ravel()
        res = premap(
            under_approx=True,
            over_approx=False,
            dataset=[x, 0, 1.0, 0.0],
            model=model,
            patch_x=1,
            patch_y=1,
            patch_h=3,
            patch_w=3,
            atk_tp="patch",
            num_outputs=1,
            sample_num=200,
            branch_budget=4,
            asserts=True,
        )
        result = torch.load(res[0])
        if len(result["domains"]) > 1:
            return
    warn("test_wrapper_conv: Could not find a case that needed more than one domain")


def test_wrapper_conv2():
    for i in range(10):
        with temp_seed(42 + i):
            model = model_conv(3, 4, 5, 4, 1).eval()
            x = torch.randn((1, 3, 5, 5))
        with torch.no_grad():
            list(model.parameters())[-1] *= 2.0
            list(model.parameters())[-1] -= model(x).ravel()
        res = premap(
            under_approx=False,
            over_approx=True,
            threshold=1.01,
            dataset=[x, 0, 1.0, 0.0],
            model=model,
            patch_x=1,
            patch_y=1,
            patch_h=3,
            patch_w=3,
            atk_tp="patch",
            num_outputs=1,
            sample_num=200,
            branch_budget=3,
            asserts=True,
        )
        result = torch.load(res[0])
        if len(result["domains"]) > 1:
            return
    warn("test_wrapper_conv: Could not find a case that needed more than one domain")


def test_wrapper_log_prob():
    with temp_seed(42):
        model = model_linear(10, 5, 1).eval()
        x = torch.randn((1, 10))
    with torch.no_grad():
        list(model.parameters())[-1] *= 2.0
        list(model.parameters())[-1] -= model(x).ravel()
    res = premap(
        dataset=[x, 0, 1.0, -1.0],
        model=model,
        threshold=0.99,
        spec_type="bound",
        num_outputs=1,
        sample_num=100,
        branch_budget=3,
        asserts=True,
        silent=True,
        seed=42,
    )
    result = torch.load(res[0])
    if len(result["domains"]) <= 1:
        warn("test_wrapper: Seed did not needed more than one domain")
    res = premap(
        dataset=[x, 0, 1.0, -1.0],
        model=model,
        spec_type="bound",
        num_outputs=1,
        sample_num=100,
        branch_budget=3,
        asserts=True,
        silent=True,
        log_prob=lambda X: X.new_ones(len(X)),
        seed=42,
    )
    result2 = torch.load(res[0])
    assert len(result["domains"]) == len(result2["domains"])
    assert is_close(result["preimage_vol"], result2["preimage_vol"])
    assert is_close(result["approx_vol"], result2["approx_vol"])
    res = premap(
        dataset=[x, 0, 1.0, -1.0],
        model=model,
        spec_type="bound",
        num_outputs=1,
        sample_num=100,
        branch_budget=3,
        asserts=True,
        silent=True,
        log_prob=lambda X: -X[:, 1] * 10.0,
        seed=42,
    )
    result3 = torch.load(res[0])
    assert not is_close(result["preimage_vol"], result3["preimage_vol"])
    # assert not is_close(result["approx_vol"], result2["approx_vol"])
