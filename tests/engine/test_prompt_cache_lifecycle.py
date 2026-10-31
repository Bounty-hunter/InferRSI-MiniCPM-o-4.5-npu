"""Prompt-cache ownership tests that do not require model weights or an accelerator."""

from types import SimpleNamespace

import torch

from engine.models.token2wav import Prompt
from engine.runtime.token2wav_batch import BatchedToken2Wav


def test_shared_prompt_stream_keeps_cache_only_in_stream_state():
    bt = BatchedToken2Wav.__new__(BatchedToken2Wav)
    bt.fast = None
    bt.t2w = SimpleNamespace()
    flow = {"large": torch.ones(4)}
    hift = {"mel": torch.zeros(1, 80, 0)}
    prompt = Prompt(
        tokens=torch.ones(1, 2, dtype=torch.int64),
        tokens_len=torch.tensor([2]),
        spk=torch.ones(1, 4),
        mels=torch.ones(1, 3, 80),
        mels_len=torch.tensor([3]),
        flow_cache=flow,
        hift_cache=hift,
    )

    stream = bt.new_stream(prompt, noise_seed=7, share_prompt=True)

    assert stream.prompt is not prompt
    assert stream.prompt.flow_cache == {} and stream.prompt.hift_cache == {}
    assert stream.flow_cache["large"] is flow["large"]
    assert stream.hift_cache["mel"] is hift["mel"]
    assert stream.prompt.spk is prompt.spk and stream.prompt.mels is prompt.mels
