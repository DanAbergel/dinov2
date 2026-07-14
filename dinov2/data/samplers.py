# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# =============================================================================
# FMRI PROJECT CHANGES (upstream DINOv2 file, modified for our fMRI pipeline)
#   + ProportionalInfiniteSampler  (L242-338, framed below): an INFINITE, iteration-
#     based sampler whose every batch_size-block has a fixed per-dataset quota
#     (HCP4/ABIDE4/OASIS4/ADNI3/AOMIC1 = 16). Selected via SamplerType.PROPORTIONAL
#     in loaders.py; needs a dataset exposing dataset_indices (MixedFMRIDataset).
#   Everything else in this file is unchanged upstream DINOv2.
# =============================================================================

import itertools
from typing import Any, Optional
import warnings

import numpy as np
import torch
from torch.utils.data.sampler import Sampler

import dinov2.distributed as distributed


class EpochSampler(Sampler):
    def __init__(
        self,
        *,
        size: int,
        sample_count: int,
        shuffle: bool = False,
        seed: int = 0,
        start: Optional[int] = None,
        step: Optional[int] = None,
    ):
        self._size = size
        self._sample_count = sample_count
        self._shuffle = shuffle
        self._seed = seed
        self._start = distributed.get_global_rank() if start is None else start
        self._step = distributed.get_global_size() if step is None else step
        self._epoch = 0

    def __iter__(self):
        count = (self._size + self._sample_count - 1) // self._sample_count
        tiled_indices = np.tile(np.arange(self._sample_count), count)
        if self._shuffle:
            seed = self._seed * self._epoch if self._seed != 0 else self._epoch
            rng = np.random.default_rng(seed)
            iterable = rng.choice(tiled_indices, self._size, replace=False)
        else:
            iterable = tiled_indices[: self._size]

        yield from itertools.islice(iterable, self._start, None, self._step)

    def __len__(self):
        return (self._size - self._start + self._step - 1) // self._step

    def set_epoch(self, epoch):
        self._epoch = epoch


def _get_numpy_dtype(size: int) -> Any:
    return np.int32 if size <= 2**31 else np.int64


def _get_torch_dtype(size: int) -> Any:
    return torch.int32 if size <= 2**31 else torch.int64


def _generate_randperm_indices(*, size: int, generator: torch.Generator):
    """Generate the indices of a random permutation."""
    dtype = _get_torch_dtype(size)
    # This is actually matching PyTorch's CPU implementation, see: https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/TensorFactories.cpp#L900-L921
    perm = torch.arange(size, dtype=dtype)
    for i in range(size):
        j = torch.randint(i, size, size=(1,), generator=generator).item()

        # Always swap even if no-op
        value = perm[j].item()
        perm[j] = perm[i].item()
        perm[i] = value
        yield value


class InfiniteSampler(Sampler):
    def __init__(
        self,
        *,
        sample_count: int,
        shuffle: bool = False,
        seed: int = 0,
        start: Optional[int] = None,
        step: Optional[int] = None,
        advance: int = 0,
    ):
        self._sample_count = sample_count
        self._seed = seed
        self._shuffle = shuffle
        self._start = distributed.get_global_rank() if start is None else start
        self._step = distributed.get_global_size() if step is None else step
        self._advance = advance

    def __iter__(self):
        if self._shuffle:
            iterator = self._shuffled_iterator()
        else:
            iterator = self._iterator()

        yield from itertools.islice(iterator, self._advance, None)

    def _iterator(self):
        assert not self._shuffle

        while True:
            iterable = range(self._sample_count)
            yield from itertools.islice(iterable, self._start, None, self._step)

    def _shuffled_iterator(self):
        assert self._shuffle

        # Instantiate a generator here (rather than in the ctor) to keep the class
        # picklable (requirement of mp.spawn)
        generator = torch.Generator().manual_seed(self._seed)

        while True:
            iterable = _generate_randperm_indices(size=self._sample_count, generator=generator)
            yield from itertools.islice(iterable, self._start, None, self._step)


# The following function is somewhat equivalent to _new_shuffle_tensor_slice below,
# but avoids a full in-place random permutation generation.
def _shuffle_tensor_slice(
    *, tensor: torch.Tensor, start: int = 0, step: int = 1, generator: torch.Generator
) -> np.ndarray:
    stop = len(tensor)
    count = stop // step
    drop_count = stop - step * count
    if drop_count:
        warnings.warn(f"# of dropped samples: {drop_count}")

    dtype = _get_numpy_dtype(stop)
    result = np.empty(count, dtype=dtype)

    for i in range(count):
        j = torch.randint(0, i + 1, size=(1,), generator=generator).item() if i > 0 else 0

        result[i] = result[j]
        result[j] = tensor[start + i * step].item()

    return result


def _new_shuffle_tensor_slice(
    *, tensor: torch.Tensor, start: int = 0, step: int = 1, generator: torch.Generator
) -> np.ndarray:
    stop = len(tensor)
    count = stop // step
    dtype = torch.int64  # Needed for using randperm result as indices
    count = stop // step
    drop_count = stop - step * count
    if drop_count:
        warnings.warn(f"# of dropped samples: {drop_count}")
    indices = torch.randperm(count, dtype=dtype, generator=generator)
    return tensor[start::step][indices].numpy()


def _make_seed(seed: int, start: int, iter_count: int) -> int:
    # NOTE: Tried a few variants (including iter_count << 32), this one worked best.
    return seed + start + (iter_count << 24)


class ShardedInfiniteSampler(Sampler):
    def __init__(
        self,
        *,
        sample_count: int,
        shuffle: bool = False,
        seed: int = 0,
        start: Optional[int] = None,
        step: Optional[int] = None,
        advance: int = 0,
        use_new_shuffle_tensor_slice: bool = False,
    ):
        self._sample_count = sample_count
        self._seed = seed
        self._shuffle = shuffle
        self._start = distributed.get_global_rank() if start is None else start
        self._step = distributed.get_global_size() if step is None else step
        self._advance = advance
        self._iter_count = 0
        self._shuffle_tensor_slice_fn = (
            _new_shuffle_tensor_slice if use_new_shuffle_tensor_slice else _shuffle_tensor_slice
        )

    def __iter__(self):
        iter_count = self._advance // self._sample_count
        if iter_count > 0:
            self._advance -= iter_count * self._sample_count
            self._iter_count += iter_count

        if self._shuffle:
            iterator = self._shuffled_iterator()
        else:
            iterator = self._iterator()

        yield from itertools.islice(iterator, self._advance, None)

    def _iterator(self):
        assert not self._shuffle

        while True:
            iterable = range(self._sample_count)
            yield from itertools.islice(iterable, self._start, None, self._step)

    def _shuffled_iterator(self):
        assert self._shuffle

        # Instantiate a generator here (rather than in the ctor) to be keep the class
        # picklable (requirement of mp.spawn)
        generator = torch.Generator()

        # Always shuffle everything first
        generator.manual_seed(self._seed)
        dtype = _get_torch_dtype(self._sample_count)
        perm = torch.randperm(self._sample_count, dtype=dtype, generator=generator)

        while True:
            # Re-seed on each iteration to allow skipping whole permutations
            seed = _make_seed(self._seed, self._start, self._iter_count)
            generator.manual_seed(seed)

            iterable = self._shuffle_tensor_slice_fn(
                tensor=perm, start=self._start, step=self._step, generator=generator
            )
            yield from iterable
            self._iter_count += 1


# ┌───────────────────────────────────────────────────────────────────────────┐
# │ FMRI ADDITION — not in upstream DINOv2.                                     │
# │ ProportionalInfiniteSampler: per-batch dataset quota as an infinite stream. │
# └───────────────────────────────────────────────────────────────────────────┘
class ProportionalInfiniteSampler(Sampler):
    """Infinite index stream whose every consecutive ``batch_size`` block has a
    fixed per-dataset composition (the ``quota``). Drop-in for DINOv2's infinite,
    iteration-based loop.

    Unlike InfiniteSampler/ShardedInfiniteSampler, this does NOT apply
    ``[start::step]`` striding (which would shuffle datasets across batch
    boundaries and break the quota). Instead each DDP rank produces its OWN
    proportional stream, seeded by its rank. Therefore the DataLoader's
    ``batch_size`` MUST equal ``sum(quota)`` so that each loader batch is exactly
    one proportional block.

    Args:
        dataset_indices: ``{dataset_name: [global indices]}`` (exposed by
            MixedFMRIDataset).
        quota: ``{dataset_name: count_per_batch}``. Keys absent from
            dataset_indices (or with count 0) are ignored.
        seed, advance: as in InfiniteSampler.
        rank: defaults to the DDP rank; seeds this rank's private stream. (No
            world_size: unlike InfiniteSampler we do not [start::step]-stride, so
            the number of ranks is never needed — each rank generates its own
            full proportional stream.)
    """

    DEFAULT_QUOTA = {"HCP": 4, "ABIDE": 4, "OASIS": 4, "ADNI": 3, "AOMIC": 1}

    def __init__(
        self,
        *,
        dataset_indices,
        quota=None,
        seed: int = 0,
        advance: int = 0,
        rank: Optional[int] = None,
    ):
        self._indices = {k: list(v) for k, v in dataset_indices.items() if v}
        quota = quota or self.DEFAULT_QUOTA
        self._quota = {k: int(q) for k, q in quota.items()
                       if k in self._indices and q > 0}
        if not self._quota:
            raise ValueError(
                f"quota {list(quota)} matches none of datasets {list(self._indices)}"
            )
        self._batch_size = sum(self._quota.values())
        self._seed = seed
        self._advance = advance
        self._rank = distributed.get_global_rank() if rank is None else rank
        # stable per-dataset seed offset, independent of dict insertion order
        self._offset = {name: i for i, name in enumerate(sorted(self._quota))}

    @property
    def batch_size(self) -> int:
        return self._batch_size

    def _pool(self, name, cycle):
        """rank- and cycle-dependent shuffle of this dataset's global indices."""
        g = torch.Generator().manual_seed(
            self._seed
            + 1_000_003 * (self._rank + 1)
            + 7_919 * self._offset[name]
            + cycle
        )
        idxs = self._indices[name]
        order = torch.randperm(len(idxs), generator=g).tolist()
        return [idxs[i] for i in order]

    def _iterator(self):
        pools, ptr, cyc = {}, {}, {}
        for name in self._quota:
            pools[name], ptr[name], cyc[name] = self._pool(name, 0), 0, 0
        batch_idx = 0
        while True:
            batch = []
            for name, q in self._quota.items():
                pool, p = pools[name], ptr[name]
                for _ in range(q):
                    if p >= len(pool):                 # exhausted -> reshuffle + cycle
                        cyc[name] += 1
                        pool = self._pool(name, cyc[name])
                        pools[name] = pool
                        p = 0
                    batch.append(pool[p])
                    p += 1
                ptr[name] = p
            g = torch.Generator().manual_seed(
                self._seed + 1_000_003 * (self._rank + 1) + 31 * batch_idx
            )
            order = torch.randperm(len(batch), generator=g).tolist()
            batch_idx += 1
            for i in order:
                yield batch[i]

    def __iter__(self):
        yield from itertools.islice(self._iterator(), self._advance, None)
# └── end FMRI ADDITION: ProportionalInfiniteSampler ──────────────────────────┘
