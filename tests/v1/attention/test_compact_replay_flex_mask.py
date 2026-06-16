import pytest
import torch

pytest.importorskip("torch.nn.attention.flex_attention")

from vllm.v1.attention.backends.flex_attention import FlexAttentionMetadata


def _metadata_with_identity_writer_mapping(
    *,
    active: bool,
) -> FlexAttentionMetadata:
    seq_len = 7
    metadata = FlexAttentionMetadata(
        causal=True,
        num_actual_tokens=seq_len,
        max_query_len=seq_len,
        query_start_loc=torch.tensor([0, seq_len], dtype=torch.int32),
        max_seq_len=seq_len,
        seq_lens=torch.tensor([seq_len], dtype=torch.int32),
        block_table=torch.arange(seq_len, dtype=torch.int32).reshape(1, seq_len),
        slot_mapping=torch.arange(seq_len, dtype=torch.int64),
        use_cascade=False,
        common_prefix_len=0,
        cu_prefix_query_lens=None,
        prefix_kv_lens=None,
        suffix_kv_lens=None,
        total_cache_tokens=seq_len,
        block_size=1,
        max_possible_sequence_length=seq_len,
        num_reqs=1,
        physical_to_logical=torch.arange(
            seq_len,
            dtype=torch.long,
        ).reshape(1, seq_len),
        decode_offset=torch.tensor([0], dtype=torch.int32),
        num_blocks_per_seq=torch.tensor([seq_len], dtype=torch.int32),
    )
    metadata.doc_ids = torch.zeros(seq_len, dtype=torch.int32)
    metadata.compact_replay_death_indices = torch.tensor(
        [[7, 4, 7, 7, 7, 7, 7]],
        dtype=torch.long,
    )
    metadata.compact_replay_active_reqs = torch.tensor([active], dtype=torch.bool)
    return metadata


def _mask_value(metadata: FlexAttentionMetadata, q_idx: int, kv_idx: int) -> bool:
    mask_mod = metadata.get_paged_mask_mod()
    return bool(
        mask_mod(
            torch.tensor(0),
            torch.tensor(0),
            torch.tensor(q_idx),
            torch.tensor(kv_idx),
        ).item()
    )


def test_compact_replay_flex_mask_hides_rows_after_death_index() -> None:
    metadata = _metadata_with_identity_writer_mapping(active=True)

    assert _mask_value(metadata, q_idx=3, kv_idx=1)
    assert not _mask_value(metadata, q_idx=4, kv_idx=1)
    assert not _mask_value(metadata, q_idx=3, kv_idx=5)


def test_compact_replay_flex_mask_is_noop_for_inactive_request() -> None:
    metadata = _metadata_with_identity_writer_mapping(active=False)

    assert _mask_value(metadata, q_idx=5, kv_idx=1)
