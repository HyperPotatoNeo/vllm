# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.v1.worker.gpu_model_runner import (
    ManagedContextCPUTransferWorker,
    _coalesce_managed_context_block_ranges,
)


@pytest.mark.parametrize(
    ("src_blocks", "dst_blocks", "expected"),
    [
        ([], [], []),
        ([1], [10], [(1, 10, 1)]),
        ([1, 2, 3], [10, 11, 12], [(1, 10, 3)]),
        ([1, 2, 4, 5], [10, 11, 20, 21], [(1, 10, 2), (4, 20, 2)]),
        ([1, 2, 3], [10, 12, 13], [(1, 10, 1), (2, 12, 2)]),
        ([1, 3, 5], [10, 11, 12], [(1, 10, 1), (3, 11, 1), (5, 12, 1)]),
        ([1, 1, 2], [10, 11, 12], [(1, 10, 1), (1, 11, 2)]),
        ([5, 4, 3], [20, 19, 18], [(5, 20, 1), (4, 19, 1), (3, 18, 1)]),
    ],
)
def test_coalesce_managed_context_block_ranges(
    src_blocks: list[int],
    dst_blocks: list[int],
    expected: list[tuple[int, int, int]],
) -> None:
    assert _coalesce_managed_context_block_ranges(src_blocks, dst_blocks) == expected


def test_coalesce_managed_context_block_ranges_rejects_length_mismatch() -> None:
    with pytest.raises(ValueError, match="length mismatch"):
        _coalesce_managed_context_block_ranges([1, 2], [10])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_managed_context_cpu_transfer_worker_round_trip() -> None:
    worker = ManagedContextCPUTransferWorker(num_cpu_blocks=8)
    gpu_cache = torch.arange(
        8 * 4,
        dtype=torch.float32,
        device="cuda",
    ).reshape(8, 4)
    cpu_cache = torch.empty((8, 4), dtype=torch.float32, device="cpu")
    if torch.cuda.is_available():
        cpu_cache = cpu_cache.pin_memory()

    worker.gpu_kv_caches = {"kv": gpu_cache}
    worker.cpu_kv_caches = {"kv": cpu_cache}
    worker.load_stream = torch.cuda.Stream()
    worker.store_stream = torch.cuda.Stream()

    src_blocks = [0, 1, 2, 5]
    cpu_blocks = [3, 4, 5, 7]

    worker._launch_copy(src_blocks, cpu_blocks, is_store=True, event_idx=1)
    worker.shutdown()
    torch.testing.assert_close(
        cpu_cache[cpu_blocks],
        gpu_cache[src_blocks].cpu(),
    )

    gpu_cache.zero_()
    worker._launch_copy(cpu_blocks, src_blocks, is_store=False, event_idx=2)
    worker.shutdown()
    torch.testing.assert_close(
        gpu_cache[src_blocks].cpu(),
        cpu_cache[cpu_blocks],
    )
