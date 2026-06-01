# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections import deque

import pytest

from vllm.v1.core.sched.scheduler import (
    ManagedContextSpan,
    _managed_context_cpu_reload_block_demand,
    _managed_context_gpu_reload_wait_reason,
    _pop_contiguous_managed_context_cpu_blocks,
)

pytestmark = pytest.mark.cpu_test


def test_pop_contiguous_managed_context_cpu_blocks_prefers_lowest_run() -> None:
    free = deque([5, 2, 3, 4, 9, 10])

    assert _pop_contiguous_managed_context_cpu_blocks(free, 3) == [2, 3, 4]
    assert list(free) == [5, 9, 10]


def test_pop_contiguous_managed_context_cpu_blocks_removes_non_adjacent_run() -> None:
    free = deque([8, 1, 6, 2, 3])

    assert _pop_contiguous_managed_context_cpu_blocks(free, 3) == [1, 2, 3]
    assert list(free) == [8, 6]


def test_pop_contiguous_managed_context_cpu_blocks_returns_none_if_fragmented() -> None:
    free = deque([0, 2, 4, 6])

    assert _pop_contiguous_managed_context_cpu_blocks(free, 2) is None
    assert list(free) == [0, 2, 4, 6]


def test_pop_contiguous_managed_context_cpu_blocks_rejects_invalid_size() -> None:
    free = deque([0, 1, 2])

    assert _pop_contiguous_managed_context_cpu_blocks(free, 0) is None
    assert _pop_contiguous_managed_context_cpu_blocks(free, 4) is None
    assert list(free) == [0, 1, 2]


def _span(span_id: str, status: str, blocks: int) -> ManagedContextSpan:
    return ManagedContextSpan(
        span_id=span_id,
        trace_id="trace",
        request_id="request",
        absolute_turn_start=0,
        absolute_turn_end=0,
        token_ids=[],
        entries=[],
        kv_block_count=blocks,
        logical_start_by_group=[],
        position_offset_frame=0,
        evict_start=0,
        evict_end=0,
        created_at=0.0,
        status=status,
    )


def test_managed_context_cpu_reload_block_demand_counts_only_cold_spans() -> None:
    spans = [
        _span("T0001", "cpu_offloaded", 3),
        _span("T0002", "gpu_pinned", 7),
        _span("T0003", "cpu_hot", 5),
        _span("T0004", "cpu_offloaded", 2),
    ]

    assert _managed_context_cpu_reload_block_demand(spans) == 5


def test_managed_context_gpu_reload_wait_reason_accounts_extra_blocks() -> None:
    assert (
        _managed_context_gpu_reload_wait_reason(
            reload_blocks=4,
            extra_blocks=3,
            free_blocks=7,
        )
        is None
    )

    reason = _managed_context_gpu_reload_wait_reason(
        reload_blocks=4,
        extra_blocks=3,
        free_blocks=6,
    )

    assert reason is not None
    assert "required=7" in reason
    assert "free=6" in reason
