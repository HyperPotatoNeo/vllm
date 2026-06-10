# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from vllm.sampling_params import SamplingParams
from vllm.v1.request import Request, RequestStatus


def test_request_status_fmt_str():
    """Test that the string representation of RequestStatus is correct."""
    assert f"{RequestStatus.WAITING}" == "WAITING"
    assert (
        f"{RequestStatus.WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR}"
        == "WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR"
    )
    assert f"{RequestStatus.WAITING_FOR_REMOTE_KVS}" == "WAITING_FOR_REMOTE_KVS"
    assert f"{RequestStatus.WAITING_FOR_STREAMING_REQ}" == "WAITING_FOR_STREAMING_REQ"
    assert f"{RequestStatus.RUNNING}" == "RUNNING"
    assert f"{RequestStatus.PREEMPTED}" == "PREEMPTED"
    assert f"{RequestStatus.FINISHED_STOPPED}" == "FINISHED_STOPPED"
    assert f"{RequestStatus.FINISHED_LENGTH_CAPPED}" == "FINISHED_LENGTH_CAPPED"
    assert f"{RequestStatus.FINISHED_ABORTED}" == "FINISHED_ABORTED"
    assert f"{RequestStatus.FINISHED_IGNORED}" == "FINISHED_IGNORED"


def test_compact_replay_snapshot_marks_evicted_rows_dead_at_writer_boundary():
    request = Request(
        request_id="req",
        prompt_token_ids=[1, 2, 3, 4],
        sampling_params=SamplingParams(max_tokens=8),
        pooling_params=None,
    )
    request.append_output_token_ids([5, 6])

    assert request.mark_compact_replay_eviction(evict_start=1, tokens_evicted=2)

    request.append_output_token_ids([7, 8])
    snapshot = request.compact_replay_snapshot()

    assert snapshot is not None
    assert snapshot.token_ids == (1, 2, 3, 4, 5, 6, 7, 8)
    assert snapshot.death_indices == (8, 6, 6, 8, 8, 8, 8, 8)
    assert snapshot.live_writer_indices == (0, 3, 4, 5, 6, 7)
    assert snapshot.evictions == 1


def test_compact_replay_snapshot_includes_padding_rows():
    request = Request(
        request_id="req",
        prompt_token_ids=[1, 2],
        sampling_params=SamplingParams(max_tokens=1),
        pooling_params=None,
    )

    assert request.mark_compact_replay_eviction(evict_start=1, tokens_evicted=1)
    request.append_padding_token_ids(padding_token_id=0, count=2)
    snapshot = request.compact_replay_snapshot()

    assert snapshot is not None
    assert snapshot.token_ids == (1, 2, 0, 0)
    assert snapshot.death_indices == (4, 2, 4, 4)
    assert snapshot.live_writer_indices == (0, 2, 3)


def test_compact_replay_eviction_rejects_invalid_live_range():
    request = Request(
        request_id="req",
        prompt_token_ids=[1, 2, 3],
        sampling_params=SamplingParams(max_tokens=1),
        pooling_params=None,
    )
    before = request.compact_replay_snapshot()

    assert not request.mark_compact_replay_eviction(
        evict_start=2,
        tokens_evicted=2,
    )

    assert request.compact_replay_snapshot() == before
    assert request._kve_compact_replay_last_error is not None
