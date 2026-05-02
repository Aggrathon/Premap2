import gc
import time
from copy import copy
from dataclasses import dataclass
from io import BytesIO
from math import exp
from pathlib import Path
from tempfile import TemporaryFile
from typing import IO, Callable, Iterator, TypeVar
from warnings import warn

import numpy as np
import torch
from sortedcontainers import SortedKeyList, SortedList

from premap2.bootstrap import bootstrap_select, bootstrap_split, confidence_interval
from premap2.utils import (
    IS_TEST_OR_DEBUG,
    WithActivations,
    device_memory,
    expand_patch,
    sizeof_tensors,
    sum_exp,
)

try:
    from auto_LiRPA.patches import Patches
    from premap.preimage_beta_crown_solver_relu_split import LiRPAConvNet
except ImportError:
    pass


class Domain:
    """Class for containing the samples and data needed to calculate priorities."""

    def __init__(
        self,
        lower_in: torch.Tensor | None = None,
        upper_in: torch.Tensor | None = None,
        X: torch.Tensor | None = None,
        y: torch.Tensor | None = None,
        activations: list[torch.Tensor] | None = None,
        log_prob: torch.Tensor | None = None,
        thighten: bool = True,
        confidence: bool = False,
    ):
        self.X = X
        self.y = y
        self.lower_in = lower_in
        self.upper_in = upper_in
        self.activations = activations
        self.log_prob = log_prob

        self.log_volume: float = 0.0
        self.preimg_vol: float = 1.0
        self.approx_vol: float = 0.0
        self.priority: float = 1.0
        self.depth: int = 0

        self.lgbs_volume = 0.0 if confidence else None
        self.lgbs_preimg = None
        self.lgbs_approx = None

        self.preimg_A: torch.Tensor | None = None
        self.preimg_b: torch.Tensor | None = None
        self.lower_out: torch.Tensor | None = None
        self.upper_out: torch.Tensor | None = None
        self.lower_all: list[torch.Tensor] = []
        self.upper_all: list[torch.Tensor] = []
        self.alpha: dict[str, dict[str, torch.Tensor]] = {}
        self.beta: list[torch.Tensor] = []
        self.beta_im: dict[str, dict[str, torch.Tensor]] = {}
        self.history: list[tuple[torch.Tensor, torch.Tensor]] = []
        self.stabilized: dict[tuple[int, bool], torch.Tensor] | None = None
        if thighten:
            self.stabilized = {}
        self.constraints: list[tuple[list[int], list[int]]] = []
        self.poly_A: torch.Tensor | None = None
        self.poly_b: torch.Tensor | None = None
        self.lower_As: list[torch.Tensor] = []
        self.upper_As: list[torch.Tensor] = []
        self.selection: list[tuple[float, int, int]] = []

    @property
    def volume(self) -> float:
        return exp(self.log_volume)

    def subset(self, mask: torch.Tensor) -> "Domain":
        """Get a subset of the samples of this domain.

        Args:
            mask: Subset of samples.

        Returns:
            A copy of this domain containing the subset.
        """
        new = copy(self)
        new.selection = copy(self.selection)
        new.lower_all = copy(self.lower_all)
        new.upper_all = copy(self.upper_all)
        new.history = copy(self.history)
        if self.stabilized is not None:
            new.stabilized = copy(self.stabilized)
        self.constraints = copy(self.constraints)
        if self.X is not None:
            new.X = self.X[mask].contiguous()
        if self.y is not None:
            new.y = self.y[mask].contiguous()
        if self.activations is not None:
            new.activations = [act[mask].contiguous() for act in self.activations]
        if self.log_prob is not None:
            new.log_prob = self.log_prob[mask].contiguous()
        new.depth += 1
        return new

    def split(
        self, layer: int, index: int, *, debug: bool = IS_TEST_OR_DEBUG
    ) -> tuple["Domain", "Domain"]:
        """Split the domain into two on a ReLU neuron.

        Args:
            layer: Layer to split on.
            index: Neuron to split on.

        Returns:
            The two resulting domains.
        """
        assert self.activations is not None
        mask = self.activations[layer].flatten(1)[:, index] >= 0.0
        dom_act, dom_ina = self.subset(mask), self.subset(~mask)
        dom_act.stabilize(layer, index, True)
        dom_ina.stabilize(layer, index, False)
        lena = len(dom_act)
        leni = len(dom_ina)
        if lena == 0:
            dom_act.log_volume = dom_act.priority = -np.inf
            dom_act.preimg_vol = dom_act.approx_vol = 0.0
            if dom_act.lgbs_volume is not None:
                dom_act.lgbs_volume = -np.inf
        elif leni == 0:
            dom_ina.log_volume = dom_ina.priority = -np.inf
            dom_ina.preimg_vol = dom_ina.approx_vol = 0.0
            if dom_ina.lgbs_volume is not None:
                dom_ina.lgbs_volume = -np.inf
        else:
            lend = len(self)
            if leni > lend // 200:
                dom_act.constrain(layer, index, True)
            if lena > lend // 200:
                dom_ina.constrain(layer, index, False)
            log_size = self.log_size()
            dom_act.log_volume += dom_act.log_size() - log_size
            dom_ina.log_volume += dom_ina.log_size() - log_size
            if self.lgbs_volume is not None:
                vol_left, vol_right = bootstrap_split(mask, self.log_prob, debug=debug)
                dom_act.lgbs_volume = dom_act.lgbs_volume + vol_left
                dom_ina.lgbs_volume = dom_ina.lgbs_volume + vol_right
            dom_act.preimg_vol = dom_act.approx_vol = np.nan
            dom_ina.preimg_vol = dom_ina.approx_vol = np.nan
        if debug:
            assert abs(dom_act.volume + dom_ina.volume - self.volume) < 1e-5
        return dom_act, dom_ina

    def constrain(self, layer: int, index: int, active: bool):
        """Add split to the constraints being calculated in `premap2.constraints.calc_constraints`.
        Call this after `self.split` if the split is not trivial.

        Args:
            layer: Layer of split.
            index: Index of split.
            active: Are the neurons constrained to active (or inactive)?
        """
        while len(self.constraints) <= layer:
            self.constraints.append((SortedList(), SortedList()))
        act, ina = self.constraints[layer]
        self.constraints[layer] = (
            (act + [index], ina) if active else (act, ina + [index])
        )

    def stabilize(self, layer: int, index: int | torch.Tensor, active: bool):
        """Add split to history and optionally to the stabilised list.

        Args:
            layer: Layer index.
            index: Neuron index/indices.
            active: Are the neurons constrained to active (or inactive)?
        """
        new = index if isinstance(index, torch.Tensor) else torch.tensor([index])
        # Update history
        while len(self.history) <= layer:
            self.history.append((new[:0], new[:0]))
        act, ina = self.history[layer]
        if active:
            self.history[layer] = (torch.cat((act, new.to(act.device))), ina)
        else:
            self.history[layer] = (act, torch.cat((ina, new.to(ina.device))))
        # Update bounds:
        if active and self.lower_all:
            lower_all = self.lower_all[layer].clone()
            lower_all.view(-1)[new] = lower_all.view(-1)[new].clip(min=0)
            self.lower_all[layer] = lower_all
        elif not active and self.upper_all:
            upper_all = self.upper_all[layer].clone()
            upper_all.view(-1)[new] = upper_all.view(-1)[new].clip(max=0)
            self.upper_all[layer] = upper_all
        # Update recently stabilized
        if self.stabilized is not None:
            if (layer, active) in self.stabilized:
                old = self.stabilized[(layer, active)]
                new = torch.cat((old, new.to(old.device)))
            self.stabilized[(layer, active)] = new

    def to(self, device: torch.device | str, non_blocking: bool = True) -> "Domain":
        """Move tensors to a device."""
        if isinstance(device, str):
            device = torch.device(device)
        check = True
        if self.lower_out is not None and self.lower_out.device != device:
            check = False
        if self.lower_in is not None and self.lower_in.device != device:
            check = False
        if check:
            return self
        for k, v in self.__dict__.items():
            self.__dict__[k] = recursive_to(v, device, non_blocking)
        return self

    def __len__(self) -> int:
        return 0 if self.X is None else self.X.size(0)

    def log_size(self) -> float:
        if len(self) == 0:
            return -np.inf
        elif self.log_prob is None:
            return torch.log(torch.tensor(len(self))).item()
        else:
            return torch.logsumexp(self.log_prob, 0).item()

    def unstable(self, sample: bool = True) -> list[torch.Tensor]:
        """Get masks for the unstable activations."""
        if sample:
            assert self.activations is not None
            return [(act > 0).any(0) & (act < 0).any(0) for act in self.activations]
        else:
            return [
                (u > 0.0) & (l < 0.0)
                for l, u in zip(self.lower_all[:-1], self.upper_all)
            ]

    def get_sample(self, num: int) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Get a subset of the samples with (non-log) weights (if applicable)."""
        assert self.X is not None
        num = min(num, len(self) // 2)
        if self.log_prob is None:
            return self.X[:num].contiguous()
        else:
            lsum = torch.logsumexp(self.log_prob[:num], 0)
            weights = torch.exp(self.log_prob[:num] - lsum)
            return self.X[:num].contiguous(), weights.contiguous()

    def minimize_memory(self, mask: torch.Tensor | None = None):
        if mask is not None and self.X is not None and self.X.shape[1:] == mask.shape:
            if torch.cuda.is_available():
                torch.cuda.synchronize()  # Required due to the non-blocking `.to()`
            mask = mask.to(self.X.device)
            self.X = self.X[:, mask].contiguous()
            if (A := self.preimg_A) is not None:
                self.preimg_b = self.preimg_b + (A * self.lower_in)[..., ~mask].sum(1)
                self.preimg_A = A[..., mask]
        self.activations = None
        self.log_prob = None

    def minimize_final(self):
        assert self.priority == -np.inf
        self.lower_in = self.upper_in = None
        self.X = self.y = None
        self.activations = self.log_prob = None
        self.lower_out = self.upper_out = None
        self.lower_all = self.upper_all = self.beta = []
        self.lower_As = self.upper_As = []
        self.beta_im = self.alpha = {}
        self.stabilized = None
        self.poly_A = self.poly_b = None
        self.selection = []
        self.constraints = []

    def expand_memory(self, mask: torch.Tensor | None) -> "Domain":
        if mask is None:
            return self
        if self.X is not None:
            self.X = expand_patch(self.X, self.lower_in, mask)  # type: ignore
        if (A := self.preimg_A) is not None:
            self.preimg_A = expand_patch(A, A.new_zeros(mask.shape), mask)
        return self

    def bootstrap_final(
        self,
        output: torch.Tensor,
        threshold: torch.Tensor,
        *,
        debug: bool = IS_TEST_OR_DEBUG,
    ):
        if (
            self.lgbs_volume is None
            or self.log_volume == -np.inf
            or self.lgbs_preimg is not None
        ):
            return
        assert (A := self.preimg_A) is not None and (b := self.preimg_b) is not None
        assert self.X is not None and self.y is not None
        pred = torch.einsum("o...,n...->no", output, self.y)
        approx = torch.einsum("o...,n...->no", A, self.X) + b.view(1, -1)
        pred_bs, appr_bs = bootstrap_select(
            (pred >= threshold).all(1),
            (approx >= threshold).all(1),
            self.log_prob,
            debug=debug,
        )
        self.lgbs_preimg = self.lgbs_volume + pred_bs
        self.lgbs_approx = self.lgbs_volume + appr_bs


T = TypeVar("T")


def recursive_to(var: T, device: torch.device | str, non_blocking: bool = True) -> T:
    """Move tensors to device (recurively loops through lists, tuples, and dictionaries)"""
    if var is None:
        return var
    elif isinstance(var, (list, tuple)):
        return type(var)(recursive_to(t, device, non_blocking) for t in var)
    elif isinstance(var, dict):
        return {k: recursive_to(v, device, non_blocking) for k, v in var.items()}  # type: ignore
    elif isinstance(var, Patches):
        return var.create_similar(  # type: ignore
            var.patches.detach().to(device, non_blocking=non_blocking)  # type: ignore
        )
    elif isinstance(var, torch.Tensor):
        return var.detach().to(device, non_blocking=non_blocking)  # type: ignore
    return var


class DomainList:
    @dataclass
    class Item:
        domain: Domain
        id: int
        size: int
        reduced: bool = False
        deviced: bool = False
        cached: None | IO[bytes] | tuple[int, int] = None

    def __init__(
        self,
        model: torch.nn.Module,
        threshold: torch.Tensor,
        output: torch.Tensor,
        under: bool = True,
        device: str | torch.device | None = None,
        lower_in: torch.Tensor | None = None,
        upper_in: torch.Tensor | None = None,
        log_prob: Callable[[torch.Tensor], torch.Tensor] | None = None,
        num_samples: int = 2_000,
        keep: int = 8,
        device_size: int = 3 << 30,
        reduce_size: int = 5 << 30,
        store_size: int = 10 << 30,
        interval: int = 4,
    ):
        """Priority queue / SortedList for domains.
        If too much memory is used, then the memory usage of lower priority domains is reduced.
        If even more memory is used, then the lowest priority domains are stored on disk.

        Args:
            keep: Only reduce/store domains with index(domain) >= keep.
            device_size: Only move domains if memory would exceed this limit.
            reduce_size: Only reduce domains if memory would exceed this limit.
            store_size: Only store domains if memory would exceed this limit.
            interval: Only go through all domains ever so often.
        """
        self.threshold = threshold.to(device)
        self.device = threshold.device
        self.output = output.to(self.device)
        self.under = under
        self.model = WithActivations(model.to(device=self.device))
        self.log_prob = log_prob
        self.num_samples = num_samples
        self._list = SortedKeyList(key=lambda i: (-i.domain.priority, i.id))
        self._cache = None  # Shared temporary file for finished domains
        self.keep = keep
        self.device_size = device_size
        if total := device_memory(self.device):
            self.device_size = min(total // 2, total - 4 << 30, device_size)
        self.reduce_size = reduce_size
        self.store_size = store_size
        self.interval = interval
        self._uppersize = 0
        self._count = 0
        self.lower_in = lower_in
        self.upper_in = upper_in
        if lower_in is not None and upper_in is not None:
            self.mask = (lower_in < upper_in)[0]
            if self.mask.count_nonzero() > self.mask.numel() // 2:
                self.mask = None
        else:
            self.mask = None

    def pop(self, index: int = 0) -> Domain:
        """Load and pop the highest priority domain."""
        return self.load(self._list.pop(index), False)

    def finished(self) -> bool:
        """Are all domains fully explored"""
        return len(self._list) == 0 or self._list[0].domain.priority == -np.inf  # type: ignore

    def __len__(self) -> int:
        return len(self._list)

    def __iter__(self) -> Iterator[Domain]:
        """Iterate over domains without loading"""
        return (i.domain for i in self._list)

    def create_domain(
        self,
        thighten: bool = False,
        confidence: bool = False,
        *,
        debug: bool = IS_TEST_OR_DEBUG,
    ) -> Domain:
        from premap2.sampling import calc_samples

        d = Domain(
            self.lower_in, self.upper_in, thighten=thighten, confidence=confidence
        ).to(self.device)
        calc_samples(
            d, self.model, self.num_samples, self.mask, self.log_prob, debug=debug
        )
        return d

    @torch.no_grad()
    def add(self, domain: Domain, *, debug: bool = IS_TEST_OR_DEBUG):
        """Add a domain to the list (reducing and storing as needed)."""
        if self.under and (domain.log_volume == -np.inf or domain.preimg_vol == 0):
            print("Dropping empty subdomain.")
            return
        if domain.upper_out is not None:
            th = self.threshold.to(domain.upper_out.device)
            if torch.any(domain.upper_out < th).item():
                print(f"Dropping subdomain without preimage ({domain.volume:.5f}).")
                return
            if torch.all(domain.lower_out > th).item():
                print("Subdomain fully verified")
                domain.priority = -np.inf
                assert domain.preimg_A is not None
                shape = domain.preimg_A.shape
                domain.preimg_A = domain.preimg_A.new_zeros(1).expand(shape)
                domain.preimg_b = domain.lower_out
                domain.approx_vol = domain.preimg_vol = domain.volume
        if debug:
            assert domain.preimg_vol <= domain.volume + 1e-6, (
                f"Invalid preimage volume: {domain.preimg_vol} <= {domain.volume}"
            )
        if domain.priority == -np.inf:
            if domain.lgbs_volume is not None and domain.log_volume != -np.inf:
                domain.to(self.device).bootstrap_final(self.output, self.threshold)
            domain.minimize_final()
        self._count += 1
        item = DomainList.Item(domain, self._count, 0)
        self._list.add(item)
        if domain.priority == -np.inf and self._cache is not None:
            self.store(item)
            return
        pos = self._list.index(item)
        if pos > self.keep and self._list[pos - 1].cached is not None:  # type: ignore
            self.store(item)
            return
        item.size = sizeof_tensors(domain)
        self._uppersize += item.size
        if self._uppersize <= self.device_size:
            return
        if pos > self.keep:
            prev: "DomainList.Item" = self._list[pos - 1]  # type: ignore
            if prev.cached is not None:
                self.store(item)
                return
            elif prev.reduced:
                self.reduce(item)
                if self._uppersize <= self.store_size:
                    return
            elif prev.deviced:
                item.deviced = True
                item.domain = item.domain.to("cpu")
                if self._uppersize <= self.reduce_size:
                    return
        if self._count % self.interval == 0:
            totsize = sum(item.size for item in self._list[: self.keep])
            item: "DomainList.Item"
            for item in self._list[self.keep :]:
                totsize += item.size
                if totsize > self.store_size:
                    if item.cached is None:
                        self.store(item)
                    else:
                        break
                elif totsize > self.reduce_size:
                    if not item.reduced:
                        self.reduce(item)
                elif totsize > self.device_size and not item.deviced:
                    item.deviced = True
                    item.domain = item.domain.to("cpu")

    def iter(self) -> Iterator[Domain]:
        """Iterate over domains with forced reducing and temporary loading."""
        item: "DomainList.Item"
        for item in self._list:
            yield self.temp_load(item).to(self.device).expand_memory(self.mask)
            if not item.deviced:
                item.domain.activations = None
                item.domain = item.domain.to("cpu")

    def reduce(self, item: "DomainList.Item", resize: bool = True):
        """Remove properties that can be recalculated from a domain."""
        item.reduced = True
        item.domain.minimize_memory(self.mask)
        if not item.deviced:
            item.deviced = True
            item.domain = item.domain.to("cpu")
        if resize:
            item.size = sizeof_tensors(item.domain)

    def store(self, item: "DomainList.Item"):
        """Store a domain in the disk cache."""
        if self._cache is None:
            self._cache = TemporaryFile()
            for i in reversed(self._list):
                if i.domain.priority == -np.inf:
                    self.store(i)
                else:
                    break
        if item.cached is not None:
            return
        if not item.reduced:
            self.reduce(item, False)
        if item.domain.priority == -np.inf:
            start = self._cache.seek(0, 2)
            torch.save(item.domain, self._cache, pickle_protocol=-1)
            self._cache.flush()
            item.cached = (start, self._cache.tell() - start)
        else:
            item.cached = TemporaryFile()
            torch.save(item.domain, item.cached, pickle_protocol=-1)
            item.cached.flush()
        for k, v in item.domain.__dict__.items():
            if not isinstance(v, (int, float, bool, str)):
                item.domain.__dict__[k] = None
        item.size = 0

    def load(self, item: "DomainList.Item", resize: bool = True) -> Domain:
        """Load a domain from the disk cache (if necessary)."""
        if item.cached is None:
            return item.domain
        assert self._cache is not None
        if isinstance(item.cached, tuple):
            start, size = item.cached
            self._cache.seek(start)
            obj = torch.load(BytesIO(self._cache.read(size)))
        else:
            item.cached.seek(0)
            obj = torch.load(BytesIO(item.cached.read()))
            item.cached.close()
        item.domain.__dict__.update(obj.__dict__)
        item.cached = None
        if resize:
            item.size = sizeof_tensors(item.domain)
        return item.domain

    def temp_load(self, item: "DomainList.Item") -> Domain:
        """Temporarily load a domain (without removing the cache)."""
        if item.cached is None:
            return item.domain
        elif isinstance(item.cached, tuple):
            assert self._cache is not None
            start, size = item.cached
            self._cache.seek(start)
            return torch.load(BytesIO(self._cache.read(size)))
        else:
            item.cached.seek(0)
            return torch.load(BytesIO(item.cached.read()))

    def __del__(self):
        if self._cache is not None:
            self._cache.close()
        item: "DomainList.Item"
        for item in reversed(self._list):
            if item.cached is None:
                break
            elif not isinstance(item.cached, tuple):
                item.cached.close()

    def get_batch(self, size: int, *, debug: bool = IS_TEST_OR_DEBUG) -> list[Domain]:
        from premap2.sampling import calc_samples

        if len(self._list) == 0:
            return []
        batch = []
        while (len(batch) < size) and not self.finished():
            domain = self.pop().to(self.device)
            for k, v in domain.alpha.items():
                for kk, vv in v.items():
                    domain.alpha[k][kk] = vv.to(dtype=torch.get_default_dtype())
            if self.mask is not None:
                domain.expand_memory(self.mask)
            calc_samples(
                domain,
                self.model,
                self.num_samples,
                mask=self.mask,
                log_prob=self.log_prob,
                debug=debug,
            )
            batch.append(domain)
        return batch

    def add_batch(
        self,
        net: "LiRPAConvNet",
        batch: list[Domain],
        A_b_dict: dict[str, dict[str, dict[str, torch.Tensor]]],
        lower_outs: torch.Tensor,
        upper_outs: torch.Tensor,
        lower_alls: list[torch.Tensor],
        upper_alls: list[torch.Tensor],
        alphas: dict[str, dict[str, torch.Tensor]] = {},
        betas: list[list[torch.Tensor]] = [],
        betas_im: list[dict[str, dict[str, torch.Tensor]]] = [],
        *,
        debug: bool = IS_TEST_OR_DEBUG,
    ):
        from premap2.constraints import calc_constraints
        from premap2.coverage import calc_coverage

        with torch.no_grad():
            for i, domain in enumerate(batch):
                domain.lower_out = lower_outs[i]
                domain.upper_out = upper_outs[i]
                domain.lower_all = [v[i] for v in lower_alls]
                domain.upper_all = [v[i] for v in upper_alls]
                domain.alpha = {
                    k: {kk: vv[:, :, i, None].contiguous() for kk, vv in v.items()}
                    for k, v in alphas.items()
                }
                if betas:
                    domain.beta = betas[i]
                if betas_im:
                    domain.beta_im = betas_im[i]
            calc_coverage(
                A_b_dict, batch, self.output, self.threshold, self.under, debug=debug
            )
        calc_constraints(
            net, batch, lower_alls, upper_alls, self.mask, self.num_samples
        )
        for domain in batch:
            self.add(domain, debug=debug)

    @torch.no_grad()
    def save(
        self,
        config: dict,
        dir_path: str | Path | None = None,
        total_time: float = np.nan,
        success: bool = False,
        times: list[float] | None = None,
        ratios: list[float] | None = None,
        num_domains: list[int] | None = None,
        confidence: float | None = None,
        *,
        debug: bool = IS_TEST_OR_DEBUG,
    ) -> Path | BytesIO:
        """Save PREMAP results and configuration.

        Args:
            domains: Domains of the approximation.
            config: Configuration as a dict.
            dir_path: Path to directory where the results are saved.
            total_time: Total time required by PREMAP.
            success: Whether PREMAP was successful.
            times: List of times at the end of each iteration.
            ratios: List of approximation ratios at the end of each iteration.
            num_domains: List of number of domains at the end of each iteration.
            confidence: Calculate confidence intervals.

        Returns:
            Path to where the results where saved.
        """
        domain_list = []
        preimage_vol = 0.0
        approx_vol = 0.0
        if confidence is not None:
            lgbs_preimg = []
            lgbs_approx = []
        gc.collect()
        for d in self.iter():
            preimage_vol += d.preimg_vol
            approx_vol += d.approx_vol
            domain_list.append(
                (
                    d.preimg_A.detach(),  # type: ignore
                    d.preimg_b.detach().ravel(),  # type: ignore
                    d.preimg_vol,
                    d.approx_vol,
                    d.history,
                )
            )
            if confidence is not None and d.lgbs_volume is not None:
                if d.lgbs_preimg is None:
                    d.bootstrap_final(self.output, self.threshold)
                if d.lgbs_preimg is not None:
                    lgbs_preimg.append(d.lgbs_preimg)
                    lgbs_approx.append(d.lgbs_approx)
            del d
        if debug:
            assert preimage_vol <= 1.0 + 1e-5
            assert approx_vol <= 1.0 + 1e-5
            assert len(domain_list) == len(self)
        preimage_vol = min(preimage_vol, 1.0)
        approx_vol = min(approx_vol, 1.0)
        if not isinstance(config.get("model", {}).get("name", None), (str, type(None))):
            cls = type(config["model"]["name"])
            config["model"]["name"] = cls.__qualname__
        if not isinstance(
            config.get("preimage", {}).get("log_prob", None), (str, type(None))
        ):
            config["preimage"]["log_prob"] = getattr(
                config["preimage"]["log_prob"],
                "__qualname__",
                str(config["preimage"]["log_prob"]),
            )
        out = {
            "config": config,
            "time": total_time,
            "success": success,
            "domains": domain_list,
            "times": times,
            "num_domains": num_domains,
            "ratios": ratios,
            "iterations": len(ratios) if ratios else 0,
            "preimage_vol": preimage_vol,
            "approx_vol": approx_vol,
            "ratio": approx_vol / preimage_vol if preimage_vol > 0.0 else np.nan,
            "__version__": 251219,
        }
        if not self.under:
            out["inv_ratio"] = preimage_vol / approx_vol if approx_vol > 0.0 else np.nan
        if confidence is not None:
            if lgbs_preimg:
                bs_preimg = sum_exp(torch.stack(lgbs_preimg, 0))
                bs_approx = sum_exp(torch.stack(lgbs_approx, 0))
                bs_ratio = bs_approx / bs_preimg
                bs_ratio[torch.isnan(bs_ratio)] = 0.0
                ci_preimg = confidence_interval(bs_preimg, confidence)
                ci_approx = confidence_interval(bs_approx, confidence)
                ci_ratio = confidence_interval(bs_ratio, confidence)
                ci_preimg = (min(1.0, ci_preimg[0]), min(1.0, ci_preimg[1]))
                ci_approx = (min(1.0, ci_approx[0]), min(1.0, ci_approx[1]))
                if not self.under:
                    bs_iratio = bs_preimg / bs_approx
                    bs_iratio[torch.isnan(bs_iratio)] = 0.0
                    ci_iratio = confidence_interval(bs_iratio, confidence)
            else:
                ci_preimg = (preimage_vol, preimage_vol)
                ci_approx = (approx_vol, approx_vol)
                ci_ratio = (np.nan, np.nan)
                warn("Results contains singular confidence intervals!")
                if not self.under:
                    ci_iratio = (np.nan, np.nan)
            print(
                f"Confidence intervals ({confidence:g}):\n",
                f" Preimage volume: ({ci_preimg[0]:.5f}, {ci_preimg[1]:.5f})\n",
                f" Approximation volume: ({ci_approx[0]:.5f}, {ci_approx[1]:.5f})\n",
                f" Approximation ratio: ({ci_ratio[0]:.5f}, {ci_ratio[1]:.5f})",
            )
            out["preimage_ci"] = ci_preimg
            out["approx_ci"] = ci_approx
            out["ratio_ci"] = ci_ratio
            if not self.under:
                print(f"  Inverse ratio: ({ci_iratio[0]:.5f}, {ci_iratio[1]:.5f})")
                out["inv_ratio_ci"] = ci_iratio
            if debug:
                assert ci_preimg[0] - 1e-8 <= preimage_vol <= ci_preimg[1] + 1e-8
                assert ci_approx[0] - 1e-8 <= approx_vol <= ci_approx[1] + 1e-8
        if dir_path is None:
            path = BytesIO()
            torch.save(out, path, pickle_protocol=5)
            path.seek(0)
        else:
            dir_path = Path(dir_path)
            dir_path.mkdir(parents=True, exist_ok=True)
            path = dir_path / f"premap_{time.strftime('%Y-%m-%d_%H-%M-%S')}.pt"
            torch.save(out, path, pickle_protocol=5)
            print("PREMAP results saved to:", str(path))
        return path
