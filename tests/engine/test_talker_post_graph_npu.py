"""L0 counter-sampler replay regression; not a speech/performance acceptance."""
from dataclasses import replace

import pytest
import torch

pytest.importorskip("torch_npu")

from engine.runtime.talker_post_graph import NpuTalkerPostGraphs
from engine.talker_sampler import PAD, DecodeState, TalkerSampling, draw, step, uniform

SEEDS = [0, 42, (1 << 63) - 1, (1 << 60) + 1234567, 1 << 32, 1, 2035223770, (1 << 32) - 1]
CFG = TalkerSampling(semantics="omni", rng="counter")
VOCAB, WIDTH = 6563, 256
EOS = VOCAB - 1


def assert_states_equal(a, b):
    for field in ("seeds", "steps", "history", "finished", "active"):
        assert torch.equal(getattr(a, field).cpu(), getattr(b, field).cpu()), field


@pytest.mark.parametrize("rows", [1, 3, 4, 8])
def test_block_replay_matches_eager(rows):
    torch.npu.set_device(0)
    generator = torch.Generator().manual_seed(172 + rows)
    head = (torch.randn(VOCAB, WIDTH, generator=generator) * .02).bfloat16().npu()
    eager, captured = [DecodeState.new(SEEDS[:rows], CFG, "npu:0") for _ in range(2)]
    for state in (eager, captured):
        state.steps.fill_(60)
    post = NpuTalkerPostGraphs(head, EOS, CFG)
    for _ in range(2):  # Rebind the next block using the updated frame indices.
        assert post.begin(captured, frames=16), post.failures
        for frame in range(16):
            hidden = (torch.randn(rows, WIDTH, generator=generator) * .25).bfloat16().npu()
            expected = step(eager, torch.nn.functional.linear(hidden, head).float(), EOS, CFG)
            observed, feed = post.step(hidden)
            assert torch.equal(expected.cpu(), observed.cpu()), (rows, frame, expected.cpu(), observed.cpu())
            assert torch.equal(feed.cpu(), observed.clamp_min(0).cpu())
        with pytest.raises(RuntimeError, match="exhausted"):
            post.step(hidden)
        post.finish()
        assert_states_equal(eager, captured)
    with pytest.raises(RuntimeError, match="not bound"):
        post.step(hidden)


def test_eos_and_inactive_rows_freeze_in_padded_bucket():
    torch.npu.set_device(0)
    cfg = replace(CFG, min_new_tokens=0, max_new_tokens=69)
    generator = torch.Generator().manual_seed(173)
    head = (torch.randn(VOCAB, WIDTH, generator=generator) * .02).bfloat16()
    head[:, -1] = 0
    head[EOS, -1] = 20
    head = head.npu()
    eager, captured = [DecodeState.new(SEEDS[:3], cfg, "npu:0") for _ in range(2)]
    for state in (eager, captured):
        state.steps.fill_(60)
        state.active[1] = False
    post = NpuTalkerPostGraphs(head, EOS, cfg)
    assert post.begin(captured, frames=16), post.failures
    outputs = []
    for frame in range(16):
        hidden = (torch.randn(3, WIDTH, generator=generator) * .25).bfloat16()
        hidden[:, -1] = 0
        if frame == 2:
            hidden[0, -1] = 10  # Forced EOS in the middle of the block.
        hidden = hidden.npu()
        expected = step(eager, torch.nn.functional.linear(hidden, head).float(), EOS, cfg)
        observed, _ = post.step(hidden)
        assert torch.equal(expected.cpu(), observed.cpu()), frame
        outputs.append(observed.cpu())
    post.finish()
    assert_states_equal(eager, captured)
    outputs = torch.stack(outputs)
    assert outputs[2, 0] == EOS and (outputs[3:, 0] == PAD).all()
    assert (outputs[:, 1] == PAD).all()
    assert int(captured.steps[0]) == 63 and int(captured.steps[1]) == 60
    assert int(captured.steps[2]) == 69 and bool(captured.finished[2])
    with pytest.raises(ValueError, match="at least one"):
        post.begin(captured, frames=0)


def test_block_uniforms_and_optional_draw_are_unchanged():
    torch.npu.set_device(0)
    seeds = torch.tensor(SEEDS, dtype=torch.int64)
    steps = torch.arange(8192, dtype=torch.int64)[:, None]
    expected = uniform(seeds[None, :], steps)
    actual = uniform(seeds.npu()[None, :], steps.npu()).cpu()
    assert torch.equal(expected, actual)
    generator = torch.Generator().manual_seed(174)
    logits = torch.randn(8, VOCAB, generator=generator).npu()
    npu_seeds, npu_steps = seeds.npu(), steps[60].expand(8).npu()
    assert torch.equal(
        draw(logits, npu_seeds, npu_steps).cpu(),
        draw(logits, npu_seeds, npu_steps, uniform_values=uniform(npu_seeds, npu_steps)).cpu(),
    )
