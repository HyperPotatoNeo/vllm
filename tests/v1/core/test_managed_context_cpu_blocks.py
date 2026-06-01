# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections import deque

import pytest

from vllm.v1.core.sched.scheduler import (
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
