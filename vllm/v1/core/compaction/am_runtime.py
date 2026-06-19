# SPDX-License-Identifier: Apache-2.0
"""Attention-matching runtime helpers for the vLLM integration path.

This module ports the core AM math into the forked vLLM tree so the worker can
compact live KV caches without depending on the separate research package.

The online baseline implemented here compacts the oldest prefix region into a
synthetic prefix of size ``stride`` while keeping the newest
``window - 2 * stride`` tokens exact. That preserves the fork's window/stride
semantics while making the compaction step tractable online.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from collections.abc import Sequence
from typing import Optional

import torch


DEFAULT_PROGRESSIVE_SCHEDULE = [
    # The online vLLM baseline needs a more aggressive schedule than the
    # research-time default so a live batch does not spend hundreds of
    # sequential OMP/NNLS updates inside one compaction event.
    (64, 4, 4),
    (160, 8, 8),
    (None, 16, 8),
]

# Qwen-family chat templates separate messages with a single newline after
# ``<|im_end|>``. Turn-window AM treats that separator as part of the completed
# turn boundary so compaction never cuts between ``<|im_end|>`` and the next
# ``<|im_start|>``.
DEFAULT_TURN_SEPARATOR_TOKEN_ID_SEQUENCE = (198,)
DEFAULT_TURN_SEPARATOR_TOKEN_IDS = frozenset(DEFAULT_TURN_SEPARATOR_TOKEN_ID_SEQUENCE)


@dataclass(frozen=True)
class AttentionMatchingCompactionPlan:
    source_len: int
    protected_prefix_len: int
    synthetic_prefix_len: int
    exact_kept_tokens: int
    target_len: int
    offset_delta: int
    compact_region_len: int
    exact_region_start: int
    query_region_start: int
    query_region_len: int


@dataclass
class AttentionMatchingRequestState:
    synthetic_prefix_len: int
    protected_prefix_len: int = 0
    version: int = 0
    layer_betas: dict[str, torch.Tensor] = field(default_factory=dict)


@dataclass
class AttentionMatchingSnapshot:
    version: int
    target_len: int
    synthetic_prefix_len: int
    protected_prefix_len: int
    position_offset: int
    layer_keys: dict[str, torch.Tensor] = field(default_factory=dict)
    layer_values: dict[str, torch.Tensor] = field(default_factory=dict)
    layer_betas: dict[str, torch.Tensor] = field(default_factory=dict)


def advance_attention_matching_turn_boundary(
    token_ids: Sequence[int],
    boundary: int,
    source_len: int,
    turn_padding_token_id: int | None,
) -> int:
    """Advance past inter-turn padding/separators after ``<|im_end|>``.

    Historical runs padded immediately after ``<|im_end|>`` even though Qwen's
    chat template then emits a newline. New runs pad after the newline. This
    scanner intentionally accepts both layouts:

    ``<|im_end|> filler* newline* filler*`` and
    ``<|im_end|> newline* filler*``.
    """

    def consume_padding(pos: int) -> int:
        if turn_padding_token_id is None:
            return pos
        while pos < source_len and int(token_ids[pos]) == turn_padding_token_id:
            pos += 1
        return pos

    boundary = consume_padding(boundary)
    while (
        boundary < source_len
        and int(token_ids[boundary]) in DEFAULT_TURN_SEPARATOR_TOKEN_IDS
    ):
        boundary += 1
        boundary = consume_padding(boundary)
    return boundary


def build_attention_matching_plan(
    *,
    num_computed_tokens: int,
    window_size: int,
    stride: int,
    num_prompt_tokens: int,
    protected_prefix_len: int = 0,
) -> AttentionMatchingCompactionPlan | None:
    """Build the online AM plan for the current request state.

    The oldest ``compact_region_len`` computed tokens are replaced by a
    synthetic AM prefix of length ``stride``. The newest
    ``window_size - 2 * stride`` computed tokens are kept exact.
    """
    if window_size <= 0 or stride <= 0 or num_computed_tokens <= window_size:
        return None
    if window_size < 2 * stride:
        raise ValueError(
            "attention_matching requires compaction_window_size >= 2 * "
            f"compaction_stride, got window={window_size}, stride={stride}"
        )
    # Match the existing kv-eviction policy: do not compact prompt-only runs.
    if num_computed_tokens <= num_prompt_tokens:
        return None

    protected_prefix_len = min(max(protected_prefix_len, 0), num_computed_tokens)
    exact_kept_tokens = window_size - 2 * stride
    max_exact_kept_tokens = max(num_computed_tokens - protected_prefix_len, 0)
    exact_kept_tokens = min(exact_kept_tokens, max_exact_kept_tokens)
    compact_region_len = (
        num_computed_tokens - protected_prefix_len - exact_kept_tokens
    )
    if compact_region_len <= stride:
        return None

    target_len = protected_prefix_len + stride + exact_kept_tokens
    offset_delta = num_computed_tokens - target_len
    # Avoid pathological one-token AM waves. The AM objective is unchanged:
    # we still summarize compact_region_len tokens into stride synthetic
    # tokens, but we wait until doing so removes at least one full stride.
    if offset_delta < stride:
        return None
    exact_region_start = num_computed_tokens - exact_kept_tokens
    if exact_kept_tokens > 0:
        query_region_start = exact_region_start
        query_region_len = exact_kept_tokens
    else:
        query_region_start = 0
        query_region_len = compact_region_len

    return AttentionMatchingCompactionPlan(
        source_len=num_computed_tokens,
        protected_prefix_len=protected_prefix_len,
        synthetic_prefix_len=stride,
        exact_kept_tokens=exact_kept_tokens,
        target_len=target_len,
        offset_delta=offset_delta,
        compact_region_len=compact_region_len,
        exact_region_start=exact_region_start,
        query_region_start=query_region_start,
        query_region_len=query_region_len,
    )


def build_attention_matching_turn_plan(
    *,
    num_computed_tokens: int,
    synthetic_prefix_len: int,
    token_ids: Sequence[int],
    max_turns: int,
    keep_recent_turns: int,
    turn_end_token_id: int | None,
    turn_padding_token_id: int | None = None,
    protect_first_user: bool = True,
    min_protected_prefix_len: int = 0,
) -> AttentionMatchingCompactionPlan | None:
    """Build an AM plan using completed chat turns as the compaction unit.

    This mirrors Markovian Thinker turn-window semantics for TextWorld-style
    comparisons while still using AM to synthesize KV for the evicted region.
    The assumed rendered chat shape is:

    ``system, user, assistant, user, assistant, ...``

    with every message ending in ``turn_end_token_id``. If
    ``turn_padding_token_id`` is set, a contiguous filler run immediately after
    each message end is included in that message boundary. A completed turn is
    a user/assistant pair after the system prefix. The plan compacts old
    completed turns, keeps ``keep_recent_turns`` completed turns exact, and
    also keeps any in-flight tail exact.
    """
    if (
        synthetic_prefix_len <= 0
        or max_turns <= 0
        or keep_recent_turns <= 0
        or turn_end_token_id is None
        or num_computed_tokens <= 0
    ):
        return None

    source_len = min(num_computed_tokens, len(token_ids))
    if source_len <= 0:
        return None

    turn_ends: list[int] = []
    pos = 0
    while pos < source_len:
        token_id = token_ids[pos]
        if token_id != turn_end_token_id:
            pos += 1
            continue

        boundary = advance_attention_matching_turn_boundary(
            token_ids,
            pos + 1,
            source_len,
            turn_padding_token_id,
        )
        turn_ends.append(boundary)
        pos = boundary
    # Need at least system + one user/assistant pair.
    if len(turn_ends) < 3:
        return None

    completed_turns = (len(turn_ends) - 1) // 2
    if completed_turns <= max_turns:
        return None

    keep_recent_turns = min(keep_recent_turns, max_turns, completed_turns)
    first_kept_turn = completed_turns - keep_recent_turns
    exact_region_start = turn_ends[2 * first_kept_turn]
    protected_prefix_len = turn_ends[1] if protect_first_user else turn_ends[0]
    protected_prefix_len = max(protected_prefix_len, min_protected_prefix_len)
    protected_prefix_len = min(max(protected_prefix_len, 0), exact_region_start)

    exact_kept_tokens = source_len - exact_region_start
    compact_region_len = exact_region_start - protected_prefix_len
    if compact_region_len <= synthetic_prefix_len:
        return None

    target_len = protected_prefix_len + synthetic_prefix_len + exact_kept_tokens
    offset_delta = source_len - target_len
    if offset_delta <= 0:
        return None

    query_region_start = exact_region_start
    query_region_len = exact_kept_tokens
    if query_region_len <= 0:
        query_region_start = protected_prefix_len
        query_region_len = compact_region_len

    return AttentionMatchingCompactionPlan(
        source_len=source_len,
        protected_prefix_len=protected_prefix_len,
        synthetic_prefix_len=synthetic_prefix_len,
        exact_kept_tokens=exact_kept_tokens,
        target_len=target_len,
        offset_delta=offset_delta,
        compact_region_len=compact_region_len,
        exact_region_start=exact_region_start,
        query_region_start=query_region_start,
        query_region_len=query_region_len,
    )


def select_query_indices(length: int, max_queries: int, device: torch.device) -> torch.Tensor:
    """Choose deterministic cache-key query indices for AM."""
    if length <= 0:
        return torch.empty(0, dtype=torch.long, device=device)
    if max_queries <= 0 or length <= max_queries:
        return torch.arange(length, device=device, dtype=torch.long)
    query_idx = torch.linspace(
        0,
        length - 1,
        steps=max_queries,
        device=device,
        dtype=torch.float32,
    ).round().to(torch.long)
    return torch.unique_consecutive(query_idx)


class CompactionAlgorithm:
    """Base utilities shared by AM compaction algorithms."""

    @staticmethod
    def _require_finite(name: str, tensor: torch.Tensor) -> None:
        if not torch.isfinite(tensor).all():
            raise RuntimeError(f"attention_matching produced non-finite {name}")

    @staticmethod
    def _solve_lstsq(
        A: torch.Tensor,
        B: torch.Tensor,
        *,
        name: str,
    ) -> torch.Tensor:
        solution = torch.linalg.lstsq(A, B, driver="gels").solution
        device_name = "cpu" if A.device.type == "cpu" else "gpu"
        CompactionAlgorithm._require_finite(f"{name} {device_name} lstsq solution", solution)
        return solution

    @staticmethod
    def _solve_lstsq_cpu_rank_deficient(
        A: torch.Tensor,
        B: torch.Tensor,
        *,
        name: str,
    ) -> torch.Tensor:
        A_cpu = A.cpu()
        B_cpu = B.cpu()
        for driver in ("gelsd", "gelss"):
            try:
                solution = torch.linalg.lstsq(A_cpu, B_cpu, driver=driver).solution
                CompactionAlgorithm._require_finite(
                    f"{name} cpu {driver} lstsq solution",
                    solution,
                )
                return solution.to(A.device)
            except Exception:
                continue
        raise RuntimeError(
            f"attention_matching {name} rank-deficient cpu lstsq failed"
        )

    def _compute_C2(
        self,
        C1: torch.Tensor,
        beta: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        queries: torch.Tensor,
        attention_bias: torch.Tensor | None = None,
        ridge_lambda: float = 0,
        solver: str = "lstsq",
        ridge_scale: str = "spectral",
    ) -> torch.Tensor:
        dtype_param = K.dtype
        d = K.shape[1]
        inv_sqrt_d = (1.0 / d) ** 0.5
        queries32 = queries.to(torch.float32)
        K32 = K.to(torch.float32)
        C132 = C1.to(torch.float32)

        sK32 = (queries32 @ K32.T) * inv_sqrt_d
        if attention_bias is not None:
            bias32 = torch.broadcast_to(attention_bias.to(torch.float32), sK32.shape)
            sK32 = sK32 + bias32
        self._require_finite("C2 key logits", sK32)
        m_K = sK32.max(dim=1, keepdim=True)[0]
        exp_sK = torch.exp(sK32 - m_K)
        attn_K = exp_sK / exp_sK.sum(dim=1, keepdim=True)
        Y = attn_K @ V.to(torch.float32)

        sC32 = (queries32 @ C132.T) * inv_sqrt_d + beta.to(torch.float32)
        self._require_finite("C2 compacted logits", sC32)
        m_C = sC32.max(dim=1, keepdim=True)[0]
        exp_sC = torch.exp(sC32 - m_C)
        X = exp_sC / exp_sC.sum(dim=1, keepdim=True)
        self._require_finite("C2 design matrix", X)
        self._require_finite("C2 targets", Y)

        n, t = X.shape
        if ridge_lambda == 0:
            lam = 0
        elif ridge_scale == "spectral":
            try:
                lam = ridge_lambda * (torch.linalg.matrix_norm(X, ord=2) ** 2)
            except Exception:
                lam = ridge_lambda * ((torch.linalg.matrix_norm(X, ord="fro") ** 2) / t)
        elif ridge_scale == "frobenius":
            lam = ridge_lambda * ((torch.linalg.matrix_norm(X, ord="fro") ** 2) / t)
        elif ridge_scale == "fixed":
            lam = ridge_lambda
        else:
            raise ValueError(f"Unknown ridge_scale: {ridge_scale}")

        solve_dtype = torch.float64
        X_solve = X.to(dtype=solve_dtype)
        Y_solve = Y.to(dtype=solve_dtype)

        def _pinv_fallback() -> torch.Tensor:
            try:
                solution = torch.linalg.pinv(X_solve) @ Y_solve
                self._require_finite("C2 pinv solution", solution)
                return solution
            except Exception:
                solution = (
                    torch.linalg.pinv(X_solve.cpu()) @ Y_solve.cpu()
                ).to(X_solve.device)
                self._require_finite("C2 cpu pinv solution", solution)
                return solution

        if solver == "lstsq":
            C2_solve = None
            if lam == 0:
                try:
                    C2_solve = self._solve_lstsq(X_solve, Y_solve, name="C2")
                except Exception:
                    C2_solve = None

            if C2_solve is None:
                spectral_norm = torch.linalg.matrix_norm(X_solve).item()
                base_lam = max(lam, (spectral_norm**2) * 1e-8, 1e-8)
                for lam_scale in (1.0, 1e2, 1e4, 1e6, 1e8):
                    reg = base_lam * lam_scale
                    try:
                        if n < t:
                            XXt = X_solve @ X_solve.T
                            XXt = 0.5 * (XXt + XXt.T)
                            XXt.diagonal().add_(reg)
                            chol = torch.linalg.cholesky(XXt)
                            Z = torch.cholesky_solve(Y_solve, chol)
                            C2_solve = X_solve.T @ Z
                        else:
                            XtX = X_solve.T @ X_solve
                            XtX = 0.5 * (XtX + XtX.T)
                            XtX.diagonal().add_(reg)
                            chol = torch.linalg.cholesky(XtX)
                            XtY = X_solve.T @ Y_solve
                            C2_solve = torch.cholesky_solve(XtY, chol)
                        if torch.isfinite(C2_solve).all():
                            break
                        C2_solve = None
                    except Exception:
                        C2_solve = None
                if C2_solve is None:
                    try:
                        C2_solve = self._solve_lstsq_cpu_rank_deficient(
                            X_solve,
                            Y_solve,
                            name="C2",
                        )
                    except Exception:
                        C2_solve = _pinv_fallback()
        elif solver == "pinv":
            if lam == 0:
                C2_solve = _pinv_fallback()
            elif n >= t:
                XtX = X_solve.T @ X_solve
                XtX = 0.5 * (XtX + XtX.T)
                XtX.diagonal().add_(lam)
                try:
                    C2_solve = torch.linalg.pinv(XtX) @ (X_solve.T @ Y_solve)
                    self._require_finite("C2 regularized pinv solution", C2_solve)
                except Exception:
                    XtX_cpu = XtX.cpu()
                    XtY_cpu = (X_solve.T @ Y_solve).cpu()
                    C2_solve = (torch.linalg.pinv(XtX_cpu) @ XtY_cpu).to(X_solve.device)
                    self._require_finite("C2 cpu regularized pinv solution", C2_solve)
            else:
                XXt = X_solve @ X_solve.T
                XXt = 0.5 * (XXt + XXt.T)
                XXt.diagonal().add_(lam)
                try:
                    C2_solve = X_solve.T @ (torch.linalg.pinv(XXt) @ Y_solve)
                    self._require_finite(
                        "C2 low-rank regularized pinv solution",
                        C2_solve,
                    )
                except Exception:
                    XXt_cpu = XXt.cpu()
                    Y_cpu = Y_solve.cpu()
                    C2_solve = (
                        X_solve.T
                        @ (torch.linalg.pinv(XXt_cpu).to(X_solve.device)
                           @ Y_cpu.to(X_solve.device))
                    )
                    self._require_finite(
                        "C2 cpu low-rank regularized pinv solution",
                        C2_solve,
                    )
        elif solver == "cholesky":
            C2_solve = None
            base_lam = max(lam, 1e-8)
            for lam_scale in (1.0, 1e2, 1e4, 1e6, 1e8):
                reg = base_lam * lam_scale
                try:
                    if n < t:
                        XXt = X_solve @ X_solve.T
                        XXt = 0.5 * (XXt + XXt.T)
                        XXt.diagonal().add_(reg)
                        chol = torch.linalg.cholesky(XXt)
                        C2_solve = X_solve.T @ torch.cholesky_solve(Y_solve, chol)
                    else:
                        XtX = X_solve.T @ X_solve
                        XtX = 0.5 * (XtX + XtX.T)
                        XtX.diagonal().add_(reg)
                        chol = torch.linalg.cholesky(XtX)
                        C2_solve = torch.cholesky_solve(X_solve.T @ Y_solve, chol)
                    if torch.isfinite(C2_solve).all():
                        break
                    C2_solve = None
                except Exception:
                    C2_solve = None
            if C2_solve is None:
                C2_solve = _pinv_fallback()
        else:
            raise ValueError(f"Unknown solver: {solver}")

        self._require_finite("C2 solution", C2_solve)
        return C2_solve.to(dtype_param)

    def _compute_C2_batched(
        self,
        C1: torch.Tensor,
        beta: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        queries: torch.Tensor,
        attention_bias: torch.Tensor | None = None,
        ridge_lambda: float = 0,
        solver: str = "lstsq",
        ridge_scale: str = "spectral",
    ) -> torch.Tensor:
        dtype_param = K.dtype
        _, _, d = K.shape
        inv_sqrt_d = (1.0 / d) ** 0.5
        queries32 = queries.to(torch.float32)
        K32 = K.to(torch.float32)
        C132 = C1.to(torch.float32)

        sK32 = torch.matmul(queries32, K32.transpose(1, 2)) * inv_sqrt_d
        if attention_bias is not None:
            if attention_bias.ndim == 2:
                bias32 = attention_bias.to(torch.float32).unsqueeze(0)
            else:
                bias32 = attention_bias.to(torch.float32)
            sK32 = sK32 + torch.broadcast_to(bias32, sK32.shape)
        self._require_finite("C2 batched key logits", sK32)
        m_K = sK32.max(dim=2, keepdim=True)[0]
        exp_sK = torch.exp(sK32 - m_K)
        attn_K = exp_sK / exp_sK.sum(dim=2, keepdim=True)
        Y = torch.matmul(attn_K, V.to(torch.float32))

        sC32 = torch.matmul(queries32, C132.transpose(1, 2)) * inv_sqrt_d
        sC32 = sC32 + beta.to(torch.float32).unsqueeze(1)
        self._require_finite("C2 batched compacted logits", sC32)
        m_C = sC32.max(dim=2, keepdim=True)[0]
        exp_sC = torch.exp(sC32 - m_C)
        X = exp_sC / exp_sC.sum(dim=2, keepdim=True)
        self._require_finite("C2 batched design matrix", X)
        self._require_finite("C2 batched targets", Y)

        _, n, t = X.shape
        if ridge_lambda == 0:
            lam = torch.zeros(X.shape[0], device=X.device, dtype=torch.float64)
        elif ridge_scale == "spectral":
            try:
                lam = ridge_lambda * (torch.linalg.matrix_norm(X, ord=2, dim=(1, 2)).to(torch.float64) ** 2)
            except Exception:
                lam = ridge_lambda * (
                    (torch.linalg.matrix_norm(X, ord="fro", dim=(1, 2)).to(torch.float64) ** 2) / t
                )
        elif ridge_scale == "frobenius":
            lam = ridge_lambda * (
                (torch.linalg.matrix_norm(X, ord="fro", dim=(1, 2)).to(torch.float64) ** 2) / t
            )
        elif ridge_scale == "fixed":
            lam = torch.full(
                (X.shape[0],),
                ridge_lambda,
                device=X.device,
                dtype=torch.float64,
            )
        else:
            raise ValueError(f"Unknown ridge_scale: {ridge_scale}")

        solve_dtype = torch.float64
        X_solve = X.to(dtype=solve_dtype)
        Y_solve = Y.to(dtype=solve_dtype)

        def _per_head_fallback() -> torch.Tensor:
            return torch.stack(
                [
                    self._compute_C2(
                        C1[head_idx],
                        beta[head_idx],
                        K[head_idx],
                        V[head_idx],
                        queries[head_idx],
                        attention_bias=attention_bias,
                        ridge_lambda=ridge_lambda,
                        solver=solver,
                        ridge_scale=ridge_scale,
                    )
                    for head_idx in range(K.shape[0])
                ],
                dim=0,
            )

        if solver == "lstsq":
            C2_solve = None
            if torch.all(lam == 0):
                try:
                    C2_solve = self._solve_lstsq(X_solve, Y_solve, name="C2 batched")
                except Exception:
                    C2_solve = None

            if C2_solve is None:
                spectral_norm = torch.linalg.matrix_norm(X_solve, dim=(1, 2)).to(torch.float64)
                base_lam = torch.maximum(
                    lam,
                    torch.maximum((spectral_norm**2) * 1e-8, torch.full_like(lam, 1e-8)),
                )
                for lam_scale in (1.0, 1e2, 1e4, 1e6, 1e8):
                    reg = base_lam * lam_scale
                    try:
                        if n < t:
                            XXt = X_solve @ X_solve.transpose(1, 2)
                            XXt = 0.5 * (XXt + XXt.transpose(1, 2))
                            diag_idx = torch.arange(n, device=X_solve.device)
                            XXt[:, diag_idx, diag_idx] += reg.unsqueeze(1)
                            chol = torch.linalg.cholesky(XXt)
                            Z = torch.cholesky_solve(Y_solve, chol)
                            C2_solve = X_solve.transpose(1, 2) @ Z
                        else:
                            XtX = X_solve.transpose(1, 2) @ X_solve
                            XtX = 0.5 * (XtX + XtX.transpose(1, 2))
                            diag_idx = torch.arange(t, device=X_solve.device)
                            XtX[:, diag_idx, diag_idx] += reg.unsqueeze(1)
                            chol = torch.linalg.cholesky(XtX)
                            XtY = X_solve.transpose(1, 2) @ Y_solve
                            C2_solve = torch.cholesky_solve(XtY, chol)
                        if torch.isfinite(C2_solve).all():
                            break
                        C2_solve = None
                    except Exception:
                        C2_solve = None
                if C2_solve is None:
                    return _per_head_fallback()
        elif solver == "pinv":
            if torch.all(lam == 0):
                try:
                    C2_solve = torch.linalg.pinv(X_solve) @ Y_solve
                    self._require_finite("C2 batched pinv solution", C2_solve)
                except Exception:
                    return _per_head_fallback()
            elif n >= t:
                XtX = X_solve.transpose(1, 2) @ X_solve
                XtX = 0.5 * (XtX + XtX.transpose(1, 2))
                diag_idx = torch.arange(t, device=X_solve.device)
                XtX[:, diag_idx, diag_idx] += lam.unsqueeze(1)
                try:
                    C2_solve = torch.linalg.pinv(XtX) @ (X_solve.transpose(1, 2) @ Y_solve)
                    self._require_finite("C2 batched regularized pinv solution", C2_solve)
                except Exception:
                    return _per_head_fallback()
            else:
                XXt = X_solve @ X_solve.transpose(1, 2)
                XXt = 0.5 * (XXt + XXt.transpose(1, 2))
                diag_idx = torch.arange(n, device=X_solve.device)
                XXt[:, diag_idx, diag_idx] += lam.unsqueeze(1)
                try:
                    C2_solve = X_solve.transpose(1, 2) @ (torch.linalg.pinv(XXt) @ Y_solve)
                    self._require_finite("C2 batched low-rank regularized pinv solution", C2_solve)
                except Exception:
                    return _per_head_fallback()
        elif solver == "cholesky":
            C2_solve = None
            base_lam = torch.maximum(lam, torch.full_like(lam, 1e-8))
            for lam_scale in (1.0, 1e2, 1e4, 1e6, 1e8):
                reg = base_lam * lam_scale
                try:
                    if n < t:
                        XXt = X_solve @ X_solve.transpose(1, 2)
                        XXt = 0.5 * (XXt + XXt.transpose(1, 2))
                        diag_idx = torch.arange(n, device=X_solve.device)
                        XXt[:, diag_idx, diag_idx] += reg.unsqueeze(1)
                        chol = torch.linalg.cholesky(XXt)
                        C2_solve = X_solve.transpose(1, 2) @ torch.cholesky_solve(Y_solve, chol)
                    else:
                        XtX = X_solve.transpose(1, 2) @ X_solve
                        XtX = 0.5 * (XtX + XtX.transpose(1, 2))
                        diag_idx = torch.arange(t, device=X_solve.device)
                        XtX[:, diag_idx, diag_idx] += reg.unsqueeze(1)
                        chol = torch.linalg.cholesky(XtX)
                        C2_solve = torch.cholesky_solve(X_solve.transpose(1, 2) @ Y_solve, chol)
                    if torch.isfinite(C2_solve).all():
                        break
                    C2_solve = None
                except Exception:
                    C2_solve = None
            if C2_solve is None:
                return _per_head_fallback()
        else:
            raise ValueError(f"Unknown solver: {solver}")

        self._require_finite("C2 batched solution", C2_solve)
        return C2_solve.to(dtype_param)

    def _direct_C2(
        self,
        C1: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        indices: torch.Tensor | list[int] | None = None,
    ) -> torch.Tensor:
        if indices is not None:
            if isinstance(indices, torch.Tensor):
                indices_tensor = indices.to(device=V.device, dtype=torch.long)
            else:
                indices_tensor = torch.tensor(indices, device=V.device, dtype=torch.long)
            return V[indices_tensor]
        C1_norms_sq = (C1**2).sum(dim=1)
        K_norms_sq = (K**2).sum(dim=1)
        pairwise_dots = C1 @ K.T
        squared_distances = (
            C1_norms_sq.unsqueeze(1) + K_norms_sq.unsqueeze(0) - 2 * pairwise_dots
        )
        nearest_indices = squared_distances.argmin(dim=1)
        return V[nearest_indices]

    def _compute_C2_with_method(
        self,
        C1: torch.Tensor,
        beta: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        queries: torch.Tensor,
        method: str = "lsq",
        indices: torch.Tensor | list[int] | None = None,
        attention_bias: torch.Tensor | None = None,
        ridge_lambda: float = 0,
        solver: str = "lstsq",
        ridge_scale: str = "spectral",
    ) -> torch.Tensor:
        if method == "direct":
            return self._direct_C2(C1, K, V, indices)
        if method == "lsq":
            return self._compute_C2(
                C1,
                beta,
                K,
                V,
                queries,
                attention_bias=attention_bias,
                ridge_lambda=ridge_lambda,
                solver=solver,
                ridge_scale=ridge_scale,
            )
        raise ValueError(f"Unknown C2 computation method: {method}")

    def _compute_C2_with_method_batched(
        self,
        C1: torch.Tensor,
        beta: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        queries: torch.Tensor,
        method: str = "lsq",
        indices: list[torch.Tensor] | None = None,
        attention_bias: torch.Tensor | None = None,
        ridge_lambda: float = 0,
        solver: str = "lstsq",
        ridge_scale: str = "spectral",
    ) -> torch.Tensor:
        if method == "direct":
            assert indices is not None
            return torch.stack(
                [
                    self._direct_C2(C1[head_idx], K[head_idx], V[head_idx], indices[head_idx])
                    for head_idx in range(K.shape[0])
                ],
                dim=0,
            )
        if method == "lsq":
            return self._compute_C2_batched(
                C1,
                beta,
                K,
                V,
                queries,
                attention_bias=attention_bias,
                ridge_lambda=ridge_lambda,
                solver=solver,
                ridge_scale=ridge_scale,
            )
        raise ValueError(f"Unknown C2 computation method: {method}")

    @staticmethod
    def _nnls_pg(
        M: torch.Tensor,
        y: torch.Tensor,
        iters: int = 0,
        lower_bound: float = 1e-12,
        upper_bound: float | None = None,
    ) -> torch.Tensor:
        n, t = M.shape
        min_val = 1e-12 if lower_bound is None else lower_bound
        solve_dtype = torch.float64 if M.dtype != torch.float64 else M.dtype
        M_solve = M.to(dtype=solve_dtype)
        y_solve = y.to(dtype=solve_dtype)
        B: torch.Tensor | None = None
        CompactionAlgorithm._require_finite("NNLS matrix", M_solve)
        CompactionAlgorithm._require_finite("NNLS target", y_solve)

        try:
            B = CompactionAlgorithm._solve_lstsq(
                M_solve,
                y_solve.unsqueeze(1),
                name="NNLS",
            ).squeeze(1)
        except Exception:
            B = None

        if B is None:
            spectral_norm = torch.linalg.matrix_norm(M_solve).item()
            base_lam = max((spectral_norm**2) * 1e-8, 1e-8)

            for lam_scale in (1.0, 1e2, 1e4, 1e6, 1e8):
                lam = base_lam * lam_scale
                try:
                    if n < t:
                        gram = M_solve @ M_solve.T
                        gram = 0.5 * (gram + gram.T)
                        gram.diagonal().add_(lam)
                        chol = torch.linalg.cholesky(gram)
                        B = M_solve.T @ torch.cholesky_solve(
                            y_solve.unsqueeze(1), chol
                        ).squeeze(1)
                    else:
                        gram = M_solve.T @ M_solve
                        gram = 0.5 * (gram + gram.T)
                        gram.diagonal().add_(lam)
                        chol = torch.linalg.cholesky(gram)
                        B = torch.cholesky_solve(
                            (M_solve.T @ y_solve).unsqueeze(1), chol
                        ).squeeze(1)
                    if torch.isfinite(B).all():
                        break
                    B = None
                except Exception:
                    B = None

        if B is None:
            try:
                B = CompactionAlgorithm._solve_lstsq_cpu_rank_deficient(
                    M_solve,
                    y_solve.unsqueeze(1),
                    name="NNLS",
                ).squeeze(1)
            except Exception:
                try:
                    B = (torch.linalg.pinv(M_solve) @ y_solve.unsqueeze(1)).squeeze(1)
                    CompactionAlgorithm._require_finite("NNLS pinv solution", B)
                except Exception:
                    B = (
                        torch.linalg.pinv(M_solve.cpu()) @ y_solve.cpu().unsqueeze(1)
                    ).squeeze(1).to(M_solve.device)
                    CompactionAlgorithm._require_finite("NNLS cpu pinv solution", B)

        B = B.clamp_min_(min_val)
        if upper_bound is not None:
            B = B.clamp_max_(upper_bound)
        if iters == 0:
            return B

        u = torch.randn(t, device=M_solve.device, dtype=M_solve.dtype)
        u = u / (u.norm() + 1e-12)
        for _ in range(3):
            v = M_solve @ u
            if v.norm() == 0:
                break
            v = v / v.norm()
            u = M_solve.T @ v
            if u.norm() == 0:
                break
            u = u / u.norm()
        sigma = (u @ (M_solve.T @ (M_solve @ u))).sqrt().clamp_min(1e-6)
        eta = 1.0 / (sigma**2).clamp_min(1e-6)
        for _ in range(iters):
            grad = M_solve.T @ (M_solve @ B - y_solve)
            B = (B - eta * grad).clamp_min_(min_val)
            if upper_bound is not None:
                B = B.clamp_max_(upper_bound)
        return B


class OMPCompaction(CompactionAlgorithm):
    """Orthogonal matching pursuit compaction used by the AM baseline."""

    def __init__(
        self,
        nnls_iters: int = 0,
        nnls_lower_bound: float | None = None,
        nnls_upper_bound: float | None = None,
        c2_method: str = "lsq",
        k_choice: int = 1,
        c2_ridge_lambda: float = 0,
        c2_solver: str = "lstsq",
        c2_ridge_scale: str = "spectral",
        nnls_interval: int = 1,
        use_abs_corr: bool = False,
        normalize_exp_scores: bool = False,
        progressive_schedule: Optional[list[tuple[float | None, int, int]]] = None,
        zerobeta: bool = False,
    ):
        self.nnls_iters = nnls_iters
        self.nnls_lower_bound = nnls_lower_bound
        self.nnls_upper_bound = nnls_upper_bound
        self.c2_method = c2_method
        self.k_choice = k_choice
        self.c2_ridge_lambda = c2_ridge_lambda
        self.c2_solver = c2_solver
        self.c2_ridge_scale = c2_ridge_scale
        self.nnls_interval = nnls_interval
        self.use_abs_corr = use_abs_corr
        self.normalize_exp_scores = normalize_exp_scores
        self.progressive_schedule = progressive_schedule
        self.zerobeta = zerobeta

    def _get_schedule_params(self, num_selected: int) -> tuple[int, int]:
        if self.progressive_schedule is None:
            return self.k_choice, self.nnls_interval
        for max_keys, k_choice, nnls_interval in self.progressive_schedule:
            if max_keys is None or num_selected < max_keys:
                return k_choice, nnls_interval
        return self.progressive_schedule[-1][1], self.progressive_schedule[-1][2]

    @staticmethod
    def _stable_log_weights(B: torch.Tensor) -> torch.Tensor:
        B64 = B.to(torch.float64)
        CompactionAlgorithm._require_finite("NNLS weights", B64)
        beta64 = torch.log(B64)
        CompactionAlgorithm._require_finite("NNLS log-weights", beta64)
        return beta64.to(torch.float32)

    def compute_compacted_cache(
        self,
        K: torch.Tensor,
        V: torch.Tensor,
        queries: torch.Tensor,
        t: int,
        attention_bias: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        C1, beta, indices = self._select_keys_omp(K, queries, t, attention_bias)
        if self.zerobeta:
            beta = torch.zeros_like(beta)
        C2 = self._compute_C2_with_method(
            C1,
            beta,
            K,
            V,
            queries,
            method=self.c2_method,
            indices=indices,
            attention_bias=attention_bias,
            ridge_lambda=self.c2_ridge_lambda,
            solver=self.c2_solver,
            ridge_scale=self.c2_ridge_scale,
        )
        return C1, beta, C2, indices

    def compute_compacted_cache_batched(
        self,
        K: torch.Tensor,
        V: torch.Tensor,
        queries: torch.Tensor,
        t: int,
        attention_bias: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        """Compute AM compaction for many KV heads at once.

        Inputs are shaped ``[num_heads, num_tokens, head_dim]`` for ``K``/``V``
        and ``[num_heads, num_queries, head_dim]`` for ``queries``.

        The batched path only fuses the score/exp-score setup across heads.
        Head-local OMP/NNLS/C2 decisions remain unchanged.
        """
        num_heads, _, d = queries.shape
        inv_sqrt_d = (1.0 / d) ** 0.5
        queries32 = queries.to(torch.float32)
        K32 = K.to(torch.float32)
        scores32 = torch.matmul(queries32, K32.transpose(1, 2)) * inv_sqrt_d
        if attention_bias is not None:
            if attention_bias.ndim == 2:
                bias32 = attention_bias.to(torch.float32).unsqueeze(0)
            else:
                bias32 = attention_bias.to(torch.float32)
            scores32 = scores32 + torch.broadcast_to(bias32, scores32.shape)
        self._require_finite("OMP batched logits", scores32)
        max_scores = scores32.max(dim=2, keepdim=True)[0]
        exp_scores = torch.exp(scores32 - max_scores)
        targets = exp_scores.sum(dim=2)
        self._require_finite("OMP batched target weights", targets)

        synthetic_keys, betas, selected_indices = self._select_keys_omp_from_exp_scores_batched(
            K,
            exp_scores,
            targets,
            t,
        )
        c2_betas = betas
        if self.zerobeta:
            c2_betas = torch.zeros_like(betas)
            betas = c2_betas

        synthetic_values = self._compute_C2_with_method_batched(
            synthetic_keys,
            c2_betas,
            K,
            V,
            queries,
            method=self.c2_method,
            indices=[selected_indices[head_idx] for head_idx in range(num_heads)],
            attention_bias=attention_bias,
            ridge_lambda=self.c2_ridge_lambda,
            solver=self.c2_solver,
            ridge_scale=self.c2_ridge_scale,
        )

        return (
            synthetic_keys,
            betas,
            synthetic_values,
            [selected_indices[head_idx] for head_idx in range(num_heads)],
        )

    def _solve_nnls(
        self,
        M: torch.Tensor,
        target: torch.Tensor,
        prev_B: torch.Tensor | None,
        iteration: int,
        nnls_interval: int | None = None,
    ) -> tuple[torch.Tensor, bool]:
        interval = nnls_interval if nnls_interval is not None else self.nnls_interval
        should_solve = prev_B is None or (interval is not None and iteration % interval == 0)
        if should_solve:
            return (
                self._nnls_pg(
                    M,
                    target,
                    self.nnls_iters,
                    self.nnls_lower_bound,
                    self.nnls_upper_bound,
                ),
                True,
            )
        i = M.shape[1]
        prev_i = prev_B.shape[0]
        B = torch.zeros(i, dtype=prev_B.dtype, device=M.device)
        B[:prev_i] = prev_B
        B[prev_i:] = 1e-12 if self.nnls_lower_bound is None else self.nnls_lower_bound
        return B, False

    def _solve_nnls_batched(
        self,
        M: torch.Tensor,
        target: torch.Tensor,
        prev_B: torch.Tensor | None,
        iteration: int,
        nnls_interval: int | None = None,
    ) -> tuple[torch.Tensor, bool]:
        interval = nnls_interval if nnls_interval is not None else self.nnls_interval
        should_solve = prev_B is None or (interval is not None and iteration % interval == 0)
        if not should_solve:
            assert prev_B is not None
            num_heads, _, i = M.shape
            prev_i = prev_B.shape[1]
            B = torch.zeros(num_heads, i, dtype=prev_B.dtype, device=M.device)
            B[:, :prev_i] = prev_B
            B[:, prev_i:] = 1e-12 if self.nnls_lower_bound is None else self.nnls_lower_bound
            return B, False

        if self.nnls_iters != 0:
            return (
                torch.stack(
                    [
                        self._nnls_pg(
                            M[head_idx],
                            target[head_idx],
                            self.nnls_iters,
                            self.nnls_lower_bound,
                            self.nnls_upper_bound,
                        )
                        for head_idx in range(M.shape[0])
                    ],
                    dim=0,
                ),
                True,
            )

        min_val = 1e-12 if self.nnls_lower_bound is None else self.nnls_lower_bound
        solve_dtype = torch.float64 if M.dtype != torch.float64 else M.dtype
        M_solve = M.to(dtype=solve_dtype)
        target_solve = target.to(dtype=solve_dtype)
        self._require_finite("NNLS batched matrix", M_solve)
        self._require_finite("NNLS batched target", target_solve)

        B = None
        try:
            B = self._solve_lstsq(
                M_solve,
                target_solve.unsqueeze(-1),
                name="NNLS batched",
            ).squeeze(-1)
        except Exception:
            B = None

        if B is None or not torch.isfinite(B).all():
            B = torch.stack(
                [
                    self._nnls_pg(
                        M[head_idx],
                        target[head_idx],
                        self.nnls_iters,
                        self.nnls_lower_bound,
                        self.nnls_upper_bound,
                    )
                    for head_idx in range(M.shape[0])
                ],
                dim=0,
            )

        B = B.clamp_min_(min_val)
        if self.nnls_upper_bound is not None:
            B = B.clamp_max_(self.nnls_upper_bound)
        return B, True

    def _select_keys_omp(
        self,
        K: torch.Tensor,
        queries: torch.Tensor,
        t: int,
        attention_bias: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        n, d = queries.shape
        dtype_param = K.dtype
        inv_sqrt_d = (1.0 / d) ** 0.5
        queries32 = queries.to(torch.float32)
        K32 = K.to(torch.float32)
        scores32 = (queries32 @ K32.T) * inv_sqrt_d
        if attention_bias is not None:
            bias32 = torch.broadcast_to(attention_bias.to(torch.float32), scores32.shape)
            scores32 = scores32 + bias32
        self._require_finite("OMP logits", scores32)
        max_scores = scores32.max(dim=1, keepdim=True)[0]
        exp_scores = torch.exp(scores32 - max_scores)
        target = exp_scores.sum(dim=1)
        self._require_finite("OMP target weights", target)
        return self._select_keys_omp_from_exp_scores(K, exp_scores, target, t)

    def _select_keys_omp_from_exp_scores(
        self,
        K: torch.Tensor,
        exp_scores: torch.Tensor,
        target: torch.Tensor,
        t: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        T = K.shape[0]
        device = K.device
        dtype_param = K.dtype

        selected_indices_tensor = torch.zeros(t, dtype=torch.long, device=device)
        beta32 = torch.zeros(t, dtype=torch.float32, device=device)
        current = torch.zeros_like(target)
        mask_selected = torch.zeros(T, dtype=torch.bool, device=device)

        i = 0
        prev_B = None
        iteration = 0
        while i < t:
            residual = target - current
            if self.normalize_exp_scores:
                exp_scores_norm = torch.norm(exp_scores, dim=0, keepdim=True)
                corr = (
                    (exp_scores / (exp_scores_norm + 1e-12))
                    * residual.unsqueeze(1)
                ).sum(dim=0)
            else:
                corr = (exp_scores * residual.unsqueeze(1)).sum(dim=0)

            current_k_choice, current_nnls_interval = self._get_schedule_params(i)
            remaining = T - i
            k_select = min(current_k_choice, remaining, t - i)
            if k_select == 0:
                break

            corr_select = torch.abs(corr) if self.use_abs_corr else corr
            corr_select[mask_selected] = -float("inf")
            top_k_indices = torch.topk(corr_select, k_select, largest=True).indices
            next_i = i + k_select
            selected_indices_tensor[i:next_i] = top_k_indices
            mask_selected[top_k_indices] = True
            i = next_i

            M = exp_scores[:, selected_indices_tensor[:i]]
            B, _ = self._solve_nnls(M, target, prev_B, iteration, current_nnls_interval)
            prev_B = B
            beta32[:i] = self._stable_log_weights(B)
            current = (M.to(B.dtype) @ B).to(current.dtype)
            iteration += 1

        if i == 0:
            raise RuntimeError("OMPCompaction failed to select any keys")
        if self.nnls_interval > 1:
            M = exp_scores[:, selected_indices_tensor[:i]]
            B = self._nnls_pg(
                M,
                target,
                self.nnls_iters,
                self.nnls_lower_bound,
                self.nnls_upper_bound,
            )
            beta32[:i] = self._stable_log_weights(B)

        selected_indices = selected_indices_tensor[:i]
        C1 = K[selected_indices]
        beta = beta32[:i].to(dtype_param)
        return C1, beta, selected_indices

    def _select_keys_omp_from_exp_scores_batched(
        self,
        K: torch.Tensor,
        exp_scores: torch.Tensor,
        target: torch.Tensor,
        t: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        num_heads, _, T = exp_scores.shape
        device = K.device
        dtype_param = K.dtype

        selected_indices_tensor = torch.zeros(
            (num_heads, t), dtype=torch.long, device=device
        )
        beta32 = torch.zeros((num_heads, t), dtype=torch.float32, device=device)
        current = torch.zeros_like(target)
        mask_selected = torch.zeros((num_heads, T), dtype=torch.bool, device=device)

        i = 0
        prev_B = None
        iteration = 0
        while i < t:
            residual = target - current
            if self.normalize_exp_scores:
                exp_scores_norm = torch.norm(exp_scores, dim=1, keepdim=True)
                corr = (
                    (exp_scores / (exp_scores_norm + 1e-12))
                    * residual.unsqueeze(2)
                ).sum(dim=1)
            else:
                corr = (exp_scores * residual.unsqueeze(2)).sum(dim=1)

            current_k_choice, current_nnls_interval = self._get_schedule_params(i)
            remaining = T - i
            k_select = min(current_k_choice, remaining, t - i)
            if k_select == 0:
                break

            corr_select = torch.abs(corr) if self.use_abs_corr else corr
            corr_select = corr_select.masked_fill(mask_selected, -float("inf"))
            top_k_indices = torch.topk(corr_select, k_select, dim=1, largest=True).indices
            next_i = i + k_select
            selected_indices_tensor[:, i:next_i] = top_k_indices
            mask_selected.scatter_(1, top_k_indices, True)
            i = next_i

            gather_index = selected_indices_tensor[:, :i].unsqueeze(1).expand(
                num_heads, exp_scores.shape[1], i
            )
            M = torch.gather(exp_scores, 2, gather_index)
            B, _ = self._solve_nnls_batched(
                M,
                target,
                prev_B,
                iteration,
                current_nnls_interval,
            )
            prev_B = B
            beta32[:, :i] = self._stable_log_weights(B)
            current = (M.to(B.dtype) * B.unsqueeze(1)).sum(dim=2).to(current.dtype)
            iteration += 1

        if i == 0:
            raise RuntimeError("OMPCompaction batched path failed to select any keys")
        if self.nnls_interval > 1:
            gather_index = selected_indices_tensor[:, :i].unsqueeze(1).expand(
                num_heads, exp_scores.shape[1], i
            )
            M = torch.gather(exp_scores, 2, gather_index)
            B, _ = self._solve_nnls_batched(M, target, None, 0, 1)
            beta32[:, :i] = self._stable_log_weights(B)

        selected_indices = selected_indices_tensor[:, :i]
        gather_index = selected_indices.unsqueeze(2).expand(num_heads, i, K.shape[2])
        C1 = torch.gather(K, 1, gather_index)
        beta = beta32[:, :i].to(dtype_param)
        return C1, beta, selected_indices
