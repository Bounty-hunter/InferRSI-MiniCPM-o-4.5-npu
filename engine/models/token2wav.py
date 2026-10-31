"""Token2Wav (flow-matching DiT + HiFT) streaming decode with per-request state.

Reference: ``stepaudio2.Token2wav`` from ``minicpmo-utils`` 1.0.6 (Apache-2.0),
which the MiniCPM-o 4.5 HF model uses (``modeling_minicpmo.py:261-266``), and the
HF stream driver (``init_token2wav_cache`` / ``streaming_generate``):

* the reference voice is loaded at 16 kHz, written as 16-bit PCM and passed to
  ``_prepare_prompt`` (S3 tokens, CAMPPlus speaker embedding, 24 kHz prompt mel);
* ``set_stream_cache`` runs the flow over the prompt tokens right-padded with
  three silence tokens (4218) to build the conformer/estimator caches;
* every 28-token window (see ``engine.token2wav_chunker``) goes through
  ``flow.inference_chunk`` with 10 Euler steps, the attention caches are
  trimmed to the prompt plus the last 100 frames, HiFT decodes the mel with an
  8-frame mel / 3840-sample source cache, consecutive chunks are cross-faded
  with a Hamming window, the first non-final chunk is shifted by 3840 samples of
  leading silence, and non-final chunks hold back their last 3840 samples.

The reference keeps one global stream state per model instance; here each
request owns a ``StreamState``, so many requests can share the weights.  The
flow's fixed noise buffer (``rand_noise``) is drawn at construction; seeding the
construction makes it reproducible across processes.
"""

from __future__ import annotations

import contextlib
import io
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

SAMPLE_RATE = 24000
MEL_CACHE = 8
SOURCE_CACHE = MEL_CACHE * 480
SILENCE_TOKEN = 4218


@contextlib.contextmanager
def strict_fp32():
    """cuDNN convolutions in true FP32 (no TF32) for the duration.

    HiFT's source integrates f0 over every sample and its output phase is ``sin`` of a conv output, so TF32 rounding
    (torch's cuDNN default) moves the waveform by ~30% relative L2 against FP32 and makes it depend on the batch size
    through cuDNN's algorithm choice.  AC-3.4 compares against FP32, so Token2Wav runs with TF32 off.
    """
    if not torch.cuda.is_available():
        yield
        return
    prev = torch.backends.cudnn.allow_tf32
    torch.backends.cudnn.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cudnn.allow_tf32 = prev


def own_flow_cache(cache: dict, n_timesteps: int) -> dict:
    """A stream's copy of the flow caches the reference returns.

    The reference returns its model-wide scratch buffers (``cnn_cache_buffer``, ``att_cache_buffer`` slices of the DiT
    and the flow-matching solver, sized for 16 Euler steps); the next request's chunk overwrites them.  Only the first
    ``n_timesteps`` step entries are ever read, so a stream keeps just those.
    """
    out = {k: v.clone() for k, v in cache.items()}
    for k in ("estimator_cnn_cache", "estimator_att_cache"):
        out[k] = cache[k][:n_timesteps].clone()
    return out


def clone_tree(x):
    if isinstance(x, torch.Tensor):
        return x.clone()
    if isinstance(x, dict):
        return {k: clone_tree(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(clone_tree(v) for v in x)
    return x


def fade_in_out(fade_in: torch.Tensor, fade_out: torch.Tensor, window: torch.Tensor) -> torch.Tensor:
    n = int(window.shape[0] / 2)
    out = fade_in.clone()
    out[..., :n] = fade_in[..., :n] * window[:n] + fade_out[..., -n:] * window[n:]
    return out


@dataclass
class Prompt:
    tokens: torch.Tensor
    tokens_len: torch.Tensor
    spk: torch.Tensor
    mels: torch.Tensor
    mels_len: torch.Tensor
    flow_cache: dict  # after set_stream_cache: the per-request starting state
    hift_cache: dict


@dataclass
class StreamState:
    prompt: Prompt
    flow_cache: dict
    hift_cache: dict
    chunks: int = 0
    noise_seed: int = 0  # HiFT source noise of chunk k is drawn from chunk_noise_seed(noise_seed, k)


def chunk_noise_seed(noise_seed: int, chunk: int) -> int:
    return (noise_seed * 1_000_003 + chunk) & ((1 << 63) - 1)


def hift_noise(hift, seeds: list[int], n_samples: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    """The HiFT source noise of a batch: row ``i`` draws what a batch-1 reference call draws from the global generator
    right after ``torch.manual_seed(seeds[i])`` (``SineGen2``: initial harmonic phases ``rand``, then the additive noise
    ``randn``; the discarded noise branch of ``SourceModuleHnNSF2`` is not drawn)."""
    gen = hift.m_source.l_sin_gen
    gens = [torch.Generator(device=device).manual_seed(int(x)) for x in seeds]
    rand_ini = torch.cat([torch.rand(1, gen.dim, device=device, generator=g) for g in gens])
    # the reference fills ``randn_like(sine)`` and ``sine`` is a transposed (channel-major) view, so draw [1, dim, n]
    noise = torch.cat([torch.randn(1, gen.dim, n_samples, device=device, generator=g) for g in gens]).transpose(1, 2)
    return rand_ini, noise


def hift_samples(hift, mel_frames: int) -> int:
    return int(mel_frames * hift.f0_upsamp.scale_factor)


def istft_overlap_add(spec: torch.Tensor, n_fft: int, hop: int, window: torch.Tensor) -> torch.Tensor:
    """``torch.istft(spec, n_fft, hop, n_fft, window, center=True)`` without its host-synchronising window check (so it
    can be graph-captured): per-frame inverse real FFT, windowing, overlap-add, division by the summed squared window,
    and the ``center`` trim of ``n_fft // 2`` samples at both ends."""
    frames = torch.fft.irfft(spec, n=n_fft, dim=1) * window[None, :, None]  # [B, n_fft, T]
    T = frames.shape[2]
    total = (T - 1) * hop + n_fft
    y = F.fold(frames, output_size=(1, total), kernel_size=(1, n_fft), stride=(1, hop))[:, 0, 0]
    env = F.fold((window ** 2)[None, :, None].expand(1, n_fft, T), output_size=(1, total), kernel_size=(1, n_fft), stride=(1, hop))[0, 0, 0]
    half = n_fft // 2
    return (y / env)[:, half:total - half]


PARTS = ("ups", "src", "res")  # an upsampling stage: its transposed convolution, the source branch, the resblocks


def tf32_stages(tf32, num_upsamples: int) -> frozenset:
    """The (stage, part) pairs that ``decode_tf32`` runs in TF32: every part of every upsampling stage for True, none for
    False, else the listed entries (a config's ``t2w_hift_decode_tf32`` may be a list): a stage index for all of its
    parts or ``"<stage>:<part>"`` with a part of ``PARTS``."""
    if tf32 is True:
        tf32 = range(num_upsamples)
    if not tf32:
        return frozenset()
    out = set()
    for e in tf32:
        i, _, part = str(e).partition(":")
        assert 0 <= int(i) < num_upsamples and part in ("",) + PARTS, (tf32, num_upsamples)
        out |= {(int(i), q) for q in ((part,) if part else PARTS)}
    return frozenset(out)


def _hift_decode(hift, x: torch.Tensor, s: torch.Tensor, tf32=False) -> torch.Tensor:
    """``HiFTGenerator.decode`` with ``istft_overlap_add`` in place of ``torch.istft``.  ``tf32``: the upsampling stages
    (``ups``, the source ``source_downs`` / ``source_resblocks`` and ``resblocks``) in TF32 (``decoder_tf32``), all of
    them or the listed ones (``tf32_stages``); the input and output convolutions stay in the caller's precision."""
    import torch.nn.functional as Fn

    stages = tf32_stages(tf32, hift.num_upsamples)
    s_stft_real, s_stft_imag = hift._stft(s.squeeze(1))
    s_stft = torch.cat([s_stft_real, s_stft_imag], dim=1)
    x = hift.conv_pre(x)
    prec = lambda i, part: decoder_tf32() if (i, part) in stages else contextlib.nullcontext()  # noqa: E731
    for i in range(hift.num_upsamples):
        x = Fn.leaky_relu(x, hift.lrelu_slope)
        with prec(i, "ups"):
            x = hift.ups[i](x)
        if i == hift.num_upsamples - 1:
            x = hift.reflection_pad(x)
        with prec(i, "src"):
            si = hift.source_resblocks[i](hift.source_downs[i](s_stft))
        x = x + si
        xs = None
        with prec(i, "res"):
            for j in range(hift.num_kernels):
                xs = hift.resblocks[i * hift.num_kernels + j](x) if xs is None else xs + hift.resblocks[i * hift.num_kernels + j](x)
        x = xs / hift.num_kernels
    x = Fn.leaky_relu(x)
    x = hift.conv_post(x)
    nf = hift.istft_params["n_fft"]
    magnitude = torch.clip(torch.exp(x[:, :nf // 2 + 1, :]), max=1e2)
    phase = torch.sin(x[:, nf // 2 + 1:, :])
    y = istft_overlap_add(torch.complex(magnitude * torch.cos(phase), magnitude * torch.sin(phase)), nf, hift.istft_params["hop_len"],
                          hift.stft_window)
    return torch.clamp(y, -hift.audio_limit, hift.audio_limit)


@contextlib.contextmanager
def decoder_tf32():
    """TF32 convolutions and matmuls for the duration (HiFT's upsampling stages only, ``decode_tf32``).

    The f0 predictor and the sine source stay strict FP32: their rounding integrates into the output phase (TF32 there
    moves the waveform by ~34% relative L2).  So do the decoder's input and output convolutions: with only the
    upsampling stages in TF32 the waveform moves by at most 2.4e-3 against strict FP32 under the same seeded noise (134
    speech windows of 32 voices; 6.4e-3 with the whole decoder in TF32; ``tools/spike/hift_precision_probe.py``),
    within AC-3.4's HiFT bound of 1e-2; the served combination is checked in ``tests/engine/test_hift_tf32.py``.  With
    the batched setup encoder one chunk reached 1.06e-2 (B24); all of it came from stage 0's transposed convolution,
    and with that one in FP32 (``[1, 2, "0:src", "0:res"]``) the maximum is 3.2e-3 (``tools/spike/hift_tf32_stages.py``)."""
    if torch.cuda.is_available():
        prev = torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32
        torch.backends.cudnn.allow_tf32 = torch.backends.cuda.matmul.allow_tf32 = True
        try:
            yield
        finally:
            torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32 = prev
        return
    npu = getattr(torch, "npu", None)
    if npu is not None and npu.is_available():
        prev = npu.conv.allow_hf32, npu.matmul.allow_hf32
        npu.conv.allow_hf32 = npu.matmul.allow_hf32 = True
        try:
            yield
        finally:
            npu.conv.allow_hf32, npu.matmul.allow_hf32 = prev
        return
    yield


def hift_forward_noise(hift, mel: torch.Tensor, cache_source: torch.Tensor, rand_ini: torch.Tensor, noise: torch.Tensor,
                       decode_tf32: bool = False):
    """``HiFTGenerator.forward`` with the source noise given (graph-capturable: no host work).  ``decode_tf32``: the
    decoder after the source in TF32 (``decoder_tf32``; an approved reduced precision, off by default)."""
    f0 = hift.f0_predictor(mel)
    s = hift.f0_upsamp(f0[:, None]).transpose(1, 2)  # [b, n, 1]
    src = hift.m_source
    gen = src.l_sin_gen
    fn = torch.multiply(s, torch.arange(1, gen.harmonic_num + 2, device=s.device, dtype=torch.float32)[None, None, :])
    rad = (fn / gen.sampling_rate) % 1
    rand_ini = rand_ini.clone()
    rand_ini[:, 0] = 0
    rad[:, 0, :] = rad[:, 0, :] + rand_ini
    rad = F.interpolate(rad.transpose(1, 2), scale_factor=1 / gen.upsample_scale, mode="linear").transpose(1, 2)
    phase = torch.cumsum(rad, dim=1) * 2 * np.pi
    phase = F.interpolate(phase.transpose(1, 2) * gen.upsample_scale, scale_factor=gen.upsample_scale, mode="linear").transpose(1, 2)
    sine = torch.sin(phase) * gen.sine_amp
    uv = gen._f02uv(fn[..., :1])
    sine = sine * uv + (uv * gen.noise_std + (1 - uv) * gen.sine_amp / 3) * noise
    s = src.l_tanh(src.l_linear(sine)).transpose(1, 2)
    if cache_source.shape[2] != 0:
        s[:, :, :cache_source.shape[2]] = cache_source
    return _hift_decode(hift, mel, s, decode_tf32), s


def hift_forward(hift, mel: torch.Tensor, cache_source: torch.Tensor, seeds: list[int], decode_tf32: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    """``HiFTGenerator.forward`` with the source noise of row ``i`` drawn from its own generator seeded ``seeds[i]``, so a
    batched call equals per-stream calls and either equals the reference under that seeding."""
    src = hift.m_source
    assert type(src).__name__ == "SourceModuleHnNSF2" and not src.l_sin_gen.flag_for_pulse, type(src)
    rand_ini, noise = hift_noise(hift, seeds, hift_samples(hift, mel.shape[2]), mel.device)
    return hift_forward_noise(hift, mel, cache_source, rand_ini, noise, decode_tf32)


class Token2Wav:
    def __init__(self, model_dir: str | Path, float16: bool = False, n_timesteps: int = 10, noise_seed: int = 0, spk_threads: int = 1,
                 device: str | torch.device = "cuda"):
        import os

        # The shared Ascend image's site-packages are root-owned.  Give numba
        # (imported by librosa) a per-user writable cache instead of letting
        # it reject the package source path as an uncacheable locator.
        numba_cache = Path(os.environ.setdefault("NUMBA_CACHE_DIR", f"/tmp/minicpmo-numba-{os.getuid()}"))
        numba_cache.mkdir(parents=True, exist_ok=True)
        try:
            from stepaudio2 import Token2wav
        except ImportError:
            # The Ascend image packages StepAudio2's vocoder as the historical
            # top-level ``token2wav`` module.
            from token2wav import Token2wav

        from engine.runtime.npu_token2wav_patch import patch_cosyvoice_attention, patch_hift, redirect_cuda_to

        torch.manual_seed(noise_seed)  # the flow's rand_noise buffer is drawn here
        self.device = torch.device(device)
        if self.device.type == "npu":
            patch_cosyvoice_attention()
        with redirect_cuda_to(self.device):
            import inspect

            kwargs = {"float16": float16}
            if "n_timesteps" in inspect.signature(Token2wav).parameters:
                kwargs["n_timesteps"] = n_timesteps
            self.ref = Token2wav(str(model_dir), **kwargs)
        if spk_threads != 1:
            import onnxruntime

            opt = onnxruntime.SessionOptions()
            opt.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
            opt.intra_op_num_threads = spk_threads
            opt.inter_op_num_threads = 1
            opt.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
            self.ref.spk_model = onnxruntime.InferenceSession(f"{model_dir}/campplus.onnx", sess_options=opt, providers=["CPUExecutionProvider"])
        # CAMPPlus is CPU-only and ONNX Runtime releases the GIL.  Starting it
        # before the S3 tokenizer lets the CPU embedding overlap both the
        # remaining feature work and the prompt encoder on the NPU.  One task
        # is sufficient because prompt preparation is serialized around the
        # model's shared encoder scratch buffers.
        self.spk_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="campplus")
        self.flow, self.hift = self.ref.flow, self.ref.hift
        if self.device.type == "npu":
            patch_hift(self.hift)
        self.hift.stft_window = self.hift.stft_window.to(self.device)  # keep the plain tensor beside the module
        self.float16, self.n_timesteps = float16, n_timesteps
        self.window = torch.from_numpy(np.hamming(2 * SOURCE_CACHE)).to(self.device)

    def _amp(self):
        return torch.amp.autocast(self.device.type, dtype=torch.float16, enabled=self.float16)

    def features(self, ref_audio: str | Path | bytes, *, defer_speaker: bool = False) -> tuple:
        """Reference voice -> (S3 tokens, token lengths, CAMPPlus embedding, 24 kHz prompt mel, mel lengths): the HF 16 kHz
        PCM16 path, no flow work (safe off the engine thread)."""
        import librosa
        import soundfile as sf
        import s3tokenizer
        import torchaudio.compliance.kaldi as kaldi
        from flashcosyvoice.utils.audio import mel_spectrogram

        profile = getattr(self, "profile_features", False)
        feature_profile = getattr(self, "feature_profile", None)
        if profile and feature_profile is None:
            self.feature_profile = feature_profile = {}
        if profile:
            import time

            started = time.perf_counter()

            def mark(name):
                nonlocal started
                getattr(torch, self.device.type).current_stream().synchronize()
                now = time.perf_counter()
                acc = feature_profile.setdefault(name, [0, 0.0])
                acc[0] += 1
                acc[1] += now - started
                started = now
        else:
            mark = lambda _name: None  # noqa: E731

        src = io.BytesIO(ref_audio) if isinstance(ref_audio, (bytes, bytearray)) else str(ref_audio)
        wav16, _ = librosa.load(src, sr=16000, mono=True)
        mark("decode")
        with tempfile.NamedTemporaryFile(suffix=".wav") as f, torch.inference_mode():
            sf.write(f.name, wav16, 16000)  # PCM_16, as in init_token2wav_cache
            # Reproduce token2wav._prepare_prompt without torchaudio.load: the
            # Ascend image intentionally omits torchcodec, while soundfile and
            # librosa already provide the exact PCM path needed here.
            pcm16, _ = librosa.load(f.name, sr=16000, mono=True)
            audio16 = torch.from_numpy(pcm16)
            mark("pcm16_roundtrip")
            spk_feat = kaldi.fbank(audio16.unsqueeze(0), num_mel_bins=80, dither=0, sample_frequency=16000)
            spk_feat -= spk_feat.mean(dim=0, keepdim=True)
            spk_input = spk_feat.unsqueeze(0).numpy()
            spk_name = self.ref.spk_model.get_inputs()[0].name
            spk = self.spk_pool.submit(self.ref.spk_model.run, None, {spk_name: spk_input})
            mark("speaker_submit")
            mels = s3tokenizer.log_mel_spectrogram(audio16)
            mels, mels_lens = s3tokenizer.padding([mels])
            mark("s3_logmel")
            tokens, token_lens = self.ref.audio_tokenizer.quantize(mels.to(self.device), mels_lens.to(self.device))
            mark("s3_quantize")
            pcm24, _ = librosa.load(f.name, sr=24000, mono=True)
            prompt_mel = mel_spectrogram(torch.from_numpy(pcm24).unsqueeze(0)).transpose(1, 2).squeeze(0)
            prompt_mels = prompt_mel.unsqueeze(0).to(self.device)
            prompt_mels_lens = torch.tensor([prompt_mels.shape[1]], dtype=torch.int32, device=self.device)
            need = tokens.shape[1] * self.flow.up_rate - prompt_mels.shape[1]
            prompt_mels = F.pad(prompt_mels, (0, 0, 0, need), mode="replicate")
            mark("prompt_mel")
            if profile or not defer_speaker:
                spk = torch.tensor(spk.result()[0], device=self.device)
                if profile:
                    mark("speaker_wait")
            return tokens, token_lens, spk, prompt_mels, prompt_mels_lens

    def resolve_features(self, feats: tuple) -> tuple:
        """Materialize a deferred CAMPPlus result on the current NPU stream."""
        tokens, tokens_len, spk, mels, mels_len = feats
        if hasattr(spk, "result"):
            spk = spk.result()[0]
        if not isinstance(spk, torch.Tensor):
            spk = torch.tensor(spk, device=self.device)
        elif spk.device != self.device:
            spk = spk.to(self.device)
        return tokens, tokens_len, spk, mels, mels_len

    def setup(self, feats: tuple) -> Prompt:
        """Prompt features -> the starting stream caches (reference ``setup_cache`` over the prompt plus 3 silence tokens)."""
        feats = self.resolve_features(feats)
        tokens, tokens_len, spk, mels, mels_len = feats
        pad = torch.full((1, 3), SILENCE_TOKEN, device=tokens.device, dtype=tokens.dtype)
        from engine.runtime.npu_token2wav_patch import math_sdpa

        with torch.inference_mode(), strict_fp32(), self._amp(), math_sdpa():
            flow_cache = own_flow_cache(self.flow.setup_cache(torch.cat([tokens, pad], dim=1), mels, spk, n_timesteps=self.n_timesteps),
                                        self.n_timesteps)
        return self.prompt(feats, flow_cache)

    def prompt(self, feats: tuple, flow_cache: dict) -> Prompt:
        feats = self.resolve_features(feats)
        tokens, tokens_len, spk, mels, mels_len = feats
        hift_cache = dict(mel=torch.zeros(1, mels.shape[2], 0, device=self.device), source=torch.zeros(1, 1, 0, device=self.device),
                          speech=torch.zeros(1, 0, device=self.device))
        return Prompt(tokens, tokens_len, spk, mels, mels_len, flow_cache, hift_cache)

    def prepare(self, ref_audio: str | Path | bytes) -> Prompt:
        """Reference voice -> prompt features and starting stream caches (the HF 16 kHz PCM16 path)."""
        return self.setup(self.features(ref_audio))

    def new_stream(self, prompt: Prompt, noise_seed: int = 0) -> StreamState:
        return StreamState(prompt, clone_tree(prompt.flow_cache), clone_tree(prompt.hift_cache), noise_seed=noise_seed)

    @torch.inference_mode()
    def decode(self, st: StreamState, window: list[int], last: bool) -> np.ndarray:
        """One window of codec tokens -> float32 waveform samples (the reference ``stream(..., return_waveform=True)``)."""
        with strict_fp32():
            return self._decode(st, window, last)

    def _decode(self, st: StreamState, window: list[int], last: bool) -> np.ndarray:
        p = st.prompt
        from engine.runtime.npu_token2wav_patch import math_sdpa

        tok = torch.tensor([window], dtype=torch.int32, device=self.device)
        with self._amp(), math_sdpa():
            mel, cache = self.flow.inference_chunk(token=tok, spk=p.spk, cache=st.flow_cache, last_chunk=last, n_timesteps=self.n_timesteps)
        st.flow_cache = own_flow_cache(cache, self.n_timesteps)
        keep = p.mels.shape[1]
        c = st.flow_cache
        if c["estimator_att_cache"].shape[4] > keep + 100:
            c["estimator_att_cache"] = torch.cat([c["estimator_att_cache"][:, :, :, :, :keep], c["estimator_att_cache"][:, :, :, :, -100:]], dim=4)
        if c["conformer_att_cache"].shape[3] > keep + 100:
            c["conformer_att_cache"] = torch.cat([c["conformer_att_cache"][:, :, :, :keep, :], c["conformer_att_cache"][:, :, :, -100:, :]], dim=3)
        h = st.hift_cache
        mel = torch.concat([h["mel"], mel], dim=2)
        speech, source = hift_forward(self.hift, mel, h["source"], [chunk_noise_seed(st.noise_seed, st.chunks)])
        if h["speech"].shape[-1] > 0:
            speech = fade_in_out(speech, h["speech"], self.window)
        first = h["speech"].shape[-1] == 0
        st.hift_cache = dict(mel=mel[..., -MEL_CACHE:].clone(), source=source[:, :, -SOURCE_CACHE:].clone(), speech=speech[:, -SOURCE_CACHE:].clone())
        if not last:
            if first:
                speech = torch.cat([torch.zeros(1, SOURCE_CACHE, device=speech.device), speech[:, :-SOURCE_CACHE]], dim=1)
            else:
                speech = speech[:, :-SOURCE_CACHE]
        st.chunks += 1
        return speech.float().cpu().numpy()[0]
