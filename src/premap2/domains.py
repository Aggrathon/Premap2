import time
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryFile
from typing import Iterator, Protocol

import numpy as np
import torch
from sortedcontainers import SortedKeyList

from .sampling import Samples


class ADomain(Protocol):
    samples: Samples | None
    priority: float
    preimg_A: torch.Tensor
    preimg_b: torch.Tensor
    preimg_cov: float
    preimg_vol: float
    history: list


class PriorityDomains:
    def __init__(
        self,
        reduce_start: int = 8,
        reduce_size: int = 1 << 31,
        store_start: int = 32,
        store_size: int = 1 << 33,
    ):
        """Priority queue / SortedList for domains.
        If too much memory is used, then the memory usage of lower priority domains is reduced.
        If even more memory is used, then the lowest priority domains are stored on disk.

        Args:
            reduce_start: Only reduce domains with index(domain) >= reduce_start.
            reduce_size: Only reduce domains if memory would exceed this limit.
            store_start: Only store domains with index(domain) >= store_start.
            store_size: Only store domains if memory would exceed this limit.
        """
        self._list = SortedKeyList(key=lambda x: -x.priority)
        self._cache = None
        self._reducing = False
        self.reduce_start = reduce_start
        self.reduce_size = reduce_size
        self.store_start = store_start
        self.store_size = store_size

    def pop(self, index=0) -> ADomain:
        """Remove and get the highest priority domain."""
        return self.load(self._list.pop(index))

    def __getitem__(self, index) -> ADomain:
        return self._list[index]  # type: ignore

    def __len__(self):
        return len(self._list)

    def add(self, domain: ADomain):
        """Add a domain to the list (reducing and storing as needed)."""
        if self._cache is None:
            size = sizeof(domain)
            if size * len(self) > self.reduce_size:
                if not self._reducing:
                    self._reducing = True
                    self.reduce_start = max(self.reduce_start, len(self) - 2)
                if size * len(self) > self.store_size:
                    self._create_cache()
                    self.store_start = max(self.store_start, len(self) - 2)
        self._list.add(domain)
        if self._reducing:
            if domain.priority == -np.inf:
                self.store(self.reduce(domain))
                return
            pos = self._list.index(domain)
            if pos >= self.store_start:
                self.store(self.reduce(domain))
            elif pos >= self.reduce_start:
                self.reduce(domain)
            if len(self) > self.reduce_start:
                self.reduce(self[self.reduce_start])
            if len(self) > self.store_start:
                self.store(self[self.store_start])

    def iter_final(self) -> Iterator[ADomain]:
        """Iterate over domains (enforced loading and reducing)."""
        return (self.reduce(self.load(item)) for item in self._list)

    def _create_cache(self):
        self._cache = TemporaryFile()
        for i, item in enumerate(self._list):
            if i >= self.reduce_start:
                self.reduce(item)
            if i >= self.store_start or item.priority == -np.inf:
                self.store(item)

    def reduce(self, domain: ADomain) -> ADomain:
        """Remove properties that can be recalculated from a domain."""
        s = domain.samples
        if s is not None and s.priority is not None:
            s.activations = None
            s.priority = None
            if s.mask is not None and s.X.shape[1:] == s.mask.shape:
                # Using Sample.to(..., non_blocking=True) requires synchronization
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                s.X = s.X[:, s.mask].contiguous()
        return domain

    def store(self, domain: ADomain) -> ADomain:
        """Store a domain in the disk cache."""
        if self._cache is None or hasattr(domain, "__PriorityDomains_cache"):
            return domain
        start = self._cache.seek(0, 2)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        torch.save(domain, self._cache, pickle_protocol=5)
        self._cache.flush()
        for k, v in domain.__dict__.items():
            if not isinstance(v, (int, float, bool, str)):
                domain.__dict__[k] = None
        setattr(domain, "__PriorityDomains_cache", (start, self._cache.tell() - start))
        return domain

    def load(self, item: ADomain) -> ADomain:
        """Load a domain from the disk cache (if necessary)."""
        if self._cache is None or not hasattr(item, "__PriorityDomains_cache"):
            return item
        start, size = getattr(item, "__PriorityDomains_cache")
        self._cache.seek(start)
        obj = torch.load(BytesIO(self._cache.read(size)))
        item.__dict__.update(obj.__dict__)
        delattr(item, "__PriorityDomains_cache")
        return item

    def __del__(self):
        if self._cache is not None:
            self._cache.close()


def sizeof(item: object) -> int:
    """Calculate the size of the tensors in the item (traverses lists, tuples, dicts and objects)."""
    size = 0
    for v in item.__dict__.values():
        if _has_tensor(v):
            size += _nested_sizeof(v)
    return size


def _nested_sizeof(item: object) -> int:
    if isinstance(item, torch.Tensor):
        return item.element_size() * item.nelement()
    elif isinstance(item, (list, tuple)) and len(item) > 0:
        return sum(_nested_sizeof(i) for i in item)
    elif isinstance(item, dict) and len(item) > 0:
        return sum(_nested_sizeof(v) for v in item.values())
    elif hasattr(item, "__dict__") and len(item.__dict__) > 0:
        return sum(_nested_sizeof(v) for v in item.__dict__.values())
    return 0


def _has_tensor(item: object) -> bool:
    if isinstance(item, torch.Tensor):
        return True
    elif isinstance(item, (list, tuple)) and len(item) > 0:
        return _has_tensor(item[0])
    elif isinstance(item, dict) and len(item) > 0:
        return _has_tensor(item[next(iter(item))])
    elif hasattr(item, "__dict__") and len(item.__dict__) > 0:
        return any(_has_tensor(v) for v in item.__dict__.values())
    return False


def save_premap(
    domains: PriorityDomains | tuple[float, float, dict[str, torch.Tensor]],
    config: dict,
    dir_path: str | Path,
    total_time: float = np.nan,
    success: bool = False,
    times: list[float] | None = None,
    coverages: list[float] | None = None,
    num_domains: list[int] | None = None,
) -> Path:
    """Save PREMAP results and configuration.

    Args:
        domains: Domains of the approximation.
        config: Configuration as a dict.
        dir_path: Path to directory where the results are saved.
        total_time: Total time required by PREMAP.
        success: Whether PREMAP was successful.
        times: List of times at the end of each iteration.
        coverages: List of coverages at the end of each iteration.
        num_domains: List of number of domains at the end of each iteration.

    Returns:
        Path to where the results where saved.
    """
    if isinstance(domains, (list, tuple)):
        vol, cov, Ab = domains
        if "lA" in Ab:
            domains = [(Ab["lA"], Ab["lbias"], vol, cov, [])]
        else:
            domains = [(Ab["uA"], Ab["ubias"], vol, cov, [])]
    else:
        if not isinstance(domains, PriorityDomains):
            domains = domains.domains
        domains = [
            (
                d.preimg_A.detach().cpu(),
                d.preimg_b.detach().cpu(),
                d.preimg_vol,
                d.preimg_cov,
                d.history,
            )
            for d in domains.iter_final()
        ]
    if not isinstance(config["model"]["name"], (str, type(None))):
        cls = type(config["model"]["name"])
        config["model"]["name"] = ".".join((cls.__module__, cls.__qualname__))
    preimage_dict = {
        "config": config,
        "time": total_time,
        "success": success,
        "domains": domains,
        "times": times,
        "num_domains": num_domains,
        "coverages": coverages,
    }
    dir_path = Path(dir_path)
    dir_path.mkdir(parents=True, exist_ok=True)
    path = dir_path / f"premap_{time.strftime('%Y-%m-%d_%H-%M-%S')}.pt"
    torch.save(preimage_dict, path, pickle_protocol=5)
    print("PREMAP results saved to:", str(path))
    return path
