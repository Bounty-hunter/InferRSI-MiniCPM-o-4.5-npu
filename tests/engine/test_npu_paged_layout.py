import pytest
import torch

from engine.runtime.npu_paged import NpuDecoderRunner, NpuPagedKV, decode_bucket_size
from engine.runtime.paged import DecoderSpec


SPEC = DecoderSpec(2, 64, 4, 2, 16, 10_000.0, 1e-6, False)


def pool(pages=12, page_size=4):
    return NpuPagedKV(SPEC, pages, page_size, dtype=torch.float32, device="cpu")


def test_block_table_slot_mapping_and_append():
    kv = pool()
    first = kv.layout([7, 9], [5, 2])
    assert first.starts == [0, 0]
    assert first.lengths == [5, 2]
    assert first.block_table.tolist() == [[0, 1], [2, -1]]
    assert first.slot_mapping.tolist() == [0, 1, 2, 3, 4, 8, 9]
    assert first.qo_indptr.tolist() == [0, 5, 7]
    assert first.last_token_index.tolist() == [4, 6]

    second = kv.layout([9, 7], [3, 1])
    assert second.starts == [2, 5]
    assert second.lengths == [5, 6]
    assert second.block_table.tolist() == [[2, 3], [0, 1]]
    assert second.slot_mapping.tolist() == [10, 11, 12, 5]
    assert kv.used_pages == 4


def test_capacity_truncate_and_release_are_deterministic():
    kv = pool(pages=4)
    kv.layout([1], [7])
    assert kv.pages[1] == [0, 1]
    assert kv.can_fit({2: 8})
    assert not kv.can_fit({2: 9})
    kv.truncate(1, 3)
    assert kv.pages_needed(1, 5) == 0
    kv.release(1)
    assert kv.free[-2:] == [1, 0]
    assert kv.used_pages == 0


@pytest.mark.parametrize("active,want", [(1, 1), (2, 2), (3, 4), (8, 8), (9, 16), (17, 32), (32, 32)])
def test_decode_buckets(active, want):
    assert decode_bucket_size(active) == want


@pytest.mark.parametrize("active", [0, 33])
def test_decode_bucket_rejects_out_of_range(active):
    with pytest.raises(ValueError):
        decode_bucket_size(active)


def test_decode_bucket_reuses_an_existing_larger_graph():
    runner = object.__new__(NpuDecoderRunner)
    runner.graph_max_batch = 32
    runner.decode_graphs = {}
    assert runner.decode_graph_bucket(3) == 4

    runner.decode_graphs = {16: object()}
    assert runner.decode_graph_bucket(3) == 16
    assert runner.decode_graph_bucket(12) == 16
    assert runner.decode_graph_bucket(17) == 32

    runner.decode_graphs = {8: object(), 16: object()}
    assert runner.decode_graph_bucket(3) == 8
    assert runner.decode_graph_bucket(9) == 16


def test_graph_padding_uses_private_dummy_pages_and_preserves_live_rows():
    kv = pool(pages=16)
    layout = kv.layout([10, 11, 12], [1, 1, 1])
    runner = object.__new__(NpuDecoderRunner)
    runner.kv = kv
    runner._graph_pad_seqs = []
    x = torch.arange(3 * SPEC.hidden_size, dtype=torch.float32).reshape(3, SPEC.hidden_size)

    padded_x, padded = runner._padded_decode(x, layout, 4)

    assert padded_x.shape == (4, SPEC.hidden_size)
    assert torch.equal(padded_x[:3], x)
    assert padded_x[3].count_nonzero() == 0
    assert padded.context_lens.tolist() == [1, 1, 1, 1]
    assert padded.block_table[:3].tolist() == layout.block_table.tolist()
    assert padded.block_table[3, 0].item() not in layout.block_table[:, 0].tolist()
    assert padded.slot_mapping[:3].tolist() == layout.slot_mapping.tolist()
