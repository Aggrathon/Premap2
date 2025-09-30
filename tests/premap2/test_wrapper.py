from warnings import warn

import torch

from premap2.utils import result_contains
from premap2.wrapper import PremapInPath, construct_config, get_arguments, premap

from .utils import model_conv, model_linear, temp_seed


def test_construct_config():
    def post(config):
        assert config["debug"]["asserts"]

    with PremapInPath():
        construct_config(False, post, {"asserts": True})
        construct_config(False, post, asserts=True)
        construct_config(False, post, {"asserts": False}, asserts=True)

    get_arguments(False)


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
        )
        result = torch.load(res[0])
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
                branch_budget=10,
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
            branch_budget=10,
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


def test_wrapper_conv(tmp_path):
    for i in range(10):
        with temp_seed(42 + i):
            model = model_conv(3, 4, 5, 4, 1).eval()
            x = torch.randn((1, 3, 5, 5))
        with torch.no_grad():
            list(model.parameters())[-1] *= 2.0
            list(model.parameters())[-1] -= model(x).ravel()
        res = premap(
            dataset=[x, 0, 1.0, 0.0],
            model=model,
            result_dir=tmp_path,
            patch_x=1,
            patch_y=1,
            patch_h=3,
            patch_w=3,
            atk_tp="patch",
            num_outputs=1,
            sample_num=200,
            branch_budget=10,
            asserts=True,
        )
        result = torch.load(res[0])
        if len(result["domains"]) > 1:
            return
    warn("test_wrapper_conv: Could not find a case that needed more than one domain")
