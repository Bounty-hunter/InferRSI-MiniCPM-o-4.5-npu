from types import SimpleNamespace

import torch

from engine.runtime.token2wav_batch import BatchedToken2Wav


def test_flow_cache_move_preserves_shared_tensors() -> None:
    bt = BatchedToken2Wav.__new__(BatchedToken2Wav)
    bt.offload_stats = {
        "h2d_tensor_copies": 0,
        "h2d_tensor_reuses": 0,
        "d2h_tensor_copies": 0,
        "d2h_tensor_reuses": 0,
    }
    shared = torch.arange(8)
    own_a = torch.ones(2)
    own_b = torch.zeros(2)
    streams = [
        SimpleNamespace(flow_cache={"shared": shared, "own": own_a, "meta": 1}),
        SimpleNamespace(flow_cache={"shared": shared, "own": own_b, "meta": 2}),
    ]

    bt._move_flow_caches(streams, "cpu", "h2d")

    assert streams[0].flow_cache["shared"] is streams[1].flow_cache["shared"]
    assert streams[0].flow_cache["own"] is not streams[1].flow_cache["own"]
    assert streams[0].flow_cache["meta"] == 1
    assert streams[1].flow_cache["meta"] == 2
    assert bt.offload_stats == {
        "h2d_tensor_copies": 3,
        "h2d_tensor_reuses": 1,
        "d2h_tensor_copies": 0,
        "d2h_tensor_reuses": 0,
    }
