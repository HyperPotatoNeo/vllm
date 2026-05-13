# SPDX-License-Identifier: Apache-2.0
"""Deterministic online shuffle-control helpers."""

from __future__ import annotations

import hashlib
import random

import msgspec


class ShuffleEvent(
    msgspec.Struct,
    array_like=True,  # type: ignore[call-arg]
    omit_defaults=True,  # type: ignore[call-arg]
    gc=False,
):  # type: ignore[call-arg]
    num_output_tokens_at_shuffle: int
    chunk_index: int
    chunk_start: int
    chunk_end: int


class NoiseEvent(
    msgspec.Struct,
    array_like=True,  # type: ignore[call-arg]
    omit_defaults=True,  # type: ignore[call-arg]
    gc=False,
):  # type: ignore[call-arg]
    num_output_tokens_at_noise: int
    chunk_index: int
    chunk_start: int
    chunk_end: int
    target: str
    std: float


def _event_seed(
    *,
    base_seed: int,
    request_id: str,
    chunk_index: int,
    namespace: str,
) -> int:
    payload = f"{namespace}:{base_seed}:{request_id}:{chunk_index}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], byteorder="little", signed=False)


def should_shuffle_chunk(
    *,
    base_seed: int,
    request_id: str,
    chunk_index: int,
    probability: float,
) -> bool:
    if probability <= 0:
        return False
    if probability >= 1:
        return True
    rng = random.Random(
        _event_seed(
            base_seed=base_seed,
            request_id=request_id,
            chunk_index=chunk_index,
            namespace="shuffle-bernoulli",
        )
    )
    return rng.random() < probability


def chunk_permutation(
    *,
    base_seed: int,
    request_id: str,
    chunk_index: int,
    chunk_len: int,
) -> list[int]:
    indices = list(range(chunk_len))
    rng = random.Random(
        _event_seed(
            base_seed=base_seed,
            request_id=request_id,
            chunk_index=chunk_index,
            namespace="shuffle-permutation",
        )
    )
    rng.shuffle(indices)
    return indices


def should_noise_chunk(
    *,
    base_seed: int,
    request_id: str,
    chunk_index: int,
    probability: float,
) -> bool:
    if probability <= 0:
        return False
    if probability >= 1:
        return True
    rng = random.Random(
        _event_seed(
            base_seed=base_seed,
            request_id=request_id,
            chunk_index=chunk_index,
            namespace="noise-bernoulli",
        )
    )
    return rng.random() < probability


def noise_seed(
    *,
    base_seed: int,
    request_id: str,
    chunk_index: int,
) -> int:
    return _event_seed(
        base_seed=base_seed,
        request_id=request_id,
        chunk_index=chunk_index,
        namespace="noise-gaussian",
    )
