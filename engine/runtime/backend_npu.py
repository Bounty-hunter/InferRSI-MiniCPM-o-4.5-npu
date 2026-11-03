"""Functional Ascend backend for the dedicated MiniCPM-o engine.

This is the eager correctness milestone: a single owner thread, paged Thinker
and Talker KV, and request-owned Token2Wav state.  The continuous Track-S NPU
scheduler and NPUGraph buckets build on the same cache/runtime interfaces.
"""

from __future__ import annotations

import queue
import threading
import gc
import hashlib
import traceback
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import torch

from engine.runtime.backend import MODEL_DIR, request_seed
from engine.runtime.backend_s import THINKER_CHUNK, LiveS, TrackSBackend


# Keep a bounded Token2Wav coalescing window through the C16 tail.  Requiring a
# completely full 16-row target made the scheduler fall back to singleton
# microbatches as soon as the first request completed.
_T2W_BATCH_WAIT_MIN_ACTIVE = 8


class AscendTrackSBackend(TrackSBackend):
    def __init__(
        self,
        model_dir: Path = MODEL_DIR,
        device: str = "npu:0",
        max_live: int = 32,
        thinker_pages: int = 512,
        talker_pages: int = 768,
        page_size: int = 128,
        float16_flow: bool = False,
        fused_decoder_ops: bool = True,
        thinker_decoder_op_mode: str | None = None,
        talker_decoder_op_mode: str | None = None,
        use_graphs: bool = False,
        graph_max_batch: int = 32,
        talker_static_shape_attention: bool = False,
        talker_static_kv_capacity: int = 512,
        talker_k: int = 4,
        talker_k_max: int = 8,
        first_chunk: int = 8,
        steady_chunk: int = 25,
        prepare_workers: int = 1,
        prompt_setup_batching: bool = False,
        prompt_setup_wait_ms: float = 0.0,
        spk_threads: int = 2,
        talker_min_new_tokens: int = 50,
        talker_sampling_semantics: str = "hf",
        talker_sampling_rng: str = "auto",
        t2w_cache_offload: bool = False,
        t2w_euler_graphs: bool = False,
        t2w_npu_fused_body: bool = False,
        t2w_npu_fused_variant: str = "full",
        t2w_hift_graphs: bool = False,
        t2w_hift_decode_tf32=False,
        t2w_encoder_graphs: bool = False,
        t2w_threaded: bool = False,
        t2w_batch_wait_ms: float = 0.0,
        t2w_batch_wait_min_active: int = _T2W_BATCH_WAIT_MIN_ACTIVE,
        t2w_batch_min_ready: int | None = None,
        t2w_timesteps: int = 10,
        talker_device_blocks: bool = False,
        talker_post_graphs: bool = False,
        timeline_markers: bool = False,
        profile_stages: bool = False,
        stage_trace_dir: str | None = None,
        talker_full_block_graphs: bool = False,
        talker_full_block_warmup: bool = False,
        thinker_prefix_cache: bool = False,
        thinker_prefix_cache_min_tokens: int = 64,
        thinker_prefix_cache_min_live: int = 1,
        t2w_first_audio_priority: bool = False,
        thinker_prompt_lookup_max_tokens: int = 0,
        t2w_startup_graph_rows: tuple = (),
        t2w_startup_extended: bool = False,
        decoder_host_qo_lengths: bool = False,
        npu_core_fraction: float = 1.0,
        **_future_track_s_options,
    ):
        if torch.device(device).type != "npu":
            raise ValueError("AscendTrackSBackend requires an npu device")
        if int(t2w_timesteps) <= 0:
            raise ValueError("t2w_timesteps must be positive")
        if float(t2w_batch_wait_ms) < 0:
            raise ValueError("t2w_batch_wait_ms must be non-negative")
        if int(t2w_batch_wait_min_active) < 1:
            raise ValueError("t2w_batch_wait_min_active must be positive")
        if t2w_batch_min_ready is not None and int(t2w_batch_min_ready) < 1:
            raise ValueError("t2w_batch_min_ready must be positive when set")
        if t2w_npu_fused_body and not t2w_euler_graphs:
            raise ValueError("t2w_npu_fused_body requires t2w_euler_graphs")
        if t2w_npu_fused_variant not in ("full", "qkv", "hybrid"):
            raise ValueError("t2w_npu_fused_variant must be 'full', 'qkv', or 'hybrid'")
        if float(prompt_setup_wait_ms) < 0:
            raise ValueError("prompt_setup_wait_ms must be non-negative")
        if int(thinker_prefix_cache_min_tokens) < 1:
            raise ValueError("thinker_prefix_cache_min_tokens must be positive")
        if int(thinker_prefix_cache_min_live) < 1:
            raise ValueError("thinker_prefix_cache_min_live must be positive")
        startup_rows = tuple(dict.fromkeys(int(row) for row in t2w_startup_graph_rows))
        if any(row not in (1, 2, 4, 8) for row in startup_rows):
            raise ValueError("t2w_startup_graph_rows must use bounded request rows 1/2/4/8")
        if startup_rows and not (t2w_threaded and t2w_euler_graphs and t2w_hift_graphs):
            raise ValueError("media startup requires the threaded Euler and HiFT graph paths")
        import torch_npu  # noqa: F401
        fraction = float(npu_core_fraction)
        if not 0 < fraction <= 1:
            raise ValueError("npu_core_fraction must be in (0,1]")
        self.npu_core_policy = None
        if fraction < 1:
            device_id = torch.device(device).index or 0
            torch.npu.set_device(device_id)
            original_limit = torch.npu.get_device_limit(device_id)
            cube = max(1, int(original_limit["cube_core_num"] * fraction))
            vector = max(1, int(original_limit["vector_core_num"] * fraction))
            # Current process only, before any model operators or graph
            # capture. Every private stream inherits the same process limit;
            # this is no reservation, global device change or MPS daemon.
            torch.npu.set_device_limit(device_id, cube_num=cube, vector_num=vector)
            self.npu_core_policy = dict(fraction=fraction, before=original_limit,
                                       after=torch.npu.get_device_limit(device_id))

        from transformers import AutoTokenizer

        from engine.models.multimodal import MultimodalFrontend
        from engine.models.talker import Talker
        from engine.models.thinker import Thinker
        from engine.models.token2wav import Token2Wav
        from engine.runtime.npu_paged import NpuDecoderRunner, NpuPagedKV
        from engine.runtime.paged import DecoderSpec, fuse_weights
        from engine.runtime.graph_gate import GraphLaunchGate
        from engine.runtime.token2wav_batch import BatchedToken2Wav
        from engine.talker_sampler import TalkerSampling

        torch.npu.set_device(torch.device(device).index or 0)
        self._graph_gate = GraphLaunchGate()
        self.tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        self.thinker, self.talker = Thinker.load(model_dir, device=device), Talker.load(model_dir, device=device)
        c, t = self.thinker.cfg, self.talker.cfg
        self.tspec = DecoderSpec(c.num_layers, c.hidden_size, c.num_heads, c.num_kv_heads, c.head_dim, c.rope_theta, c.rms_eps, True)
        self.aspec = DecoderSpec(t.num_layers, t.hidden_size, t.num_heads, t.num_kv_heads, t.head_dim, t.rope_theta, t.rms_eps, False)
        thinker_layers = fuse_weights(self.thinker.w, self.tspec)
        talker_layers = fuse_weights(self.talker.w, self.aspec)
        # ``NpuDecoderRunner`` owns fused QKV and gate/up matrices.  The source
        # projections are not used by the serving path; retaining them doubles
        # almost every transformer weight and prevents the C32 KV arena from
        # fitting on a 64 GiB 910B1.
        for weights in (self.thinker.w, self.talker.w):
            for name in [name for name in weights if ".layers." in name and "_proj.weight" in name]:
                del weights[name]
        gc.collect()
        torch.npu.empty_cache()
        self.t2w = Token2Wav(
            model_dir / "assets" / "token2wav",
            float16=float16_flow,
            n_timesteps=int(t2w_timesteps),
            spk_threads=spk_threads,
            device=device,
        )
        self.t2w_batch = BatchedToken2Wav(
            self.t2w,
            use_graphs=False,
            hift_decode_tf32=t2w_hift_decode_tf32,
        )
        if t2w_hift_graphs:
            from engine.runtime.hift_graph import HiFTGraphs

            # Steady shapes benefit on 910B1.  The short final-shape graphs
            # measured slower than eager, so keep their exact eager fallback.
            self.t2w_batch.hift_graphs = HiFTGraphs(
                self.t2w.hift,
                final_frames={},
                decode_tf32=t2w_hift_decode_tf32,
            )
        if t2w_encoder_graphs:
            from engine.runtime.dit_graph import EncoderGraphs
            from engine.runtime.token2wav_batch import _encoder_core

            self.t2w_batch.enc_graphs = EncoderGraphs(
                self.t2w.flow.encoder, _encoder_core, self.t2w_batch.dtype
            )
        if t2w_euler_graphs:
            from engine.runtime.dit_graph import FINAL_T, SETUP_T, STEADY_T, EulerGraphs

            est = self.t2w.flow.decoder.estimator
            estimator_fn = BatchedToken2Wav._estimator_temb
            if t2w_npu_fused_body:
                from engine.runtime.dit_fused_npu import (
                    estimator_temb_fused_npu,
                    estimator_temb_hybrid_npu,
                    estimator_temb_qkv_npu,
                    prepare_estimator,
                )

                prepare_estimator(est, pack_convs=t2w_npu_fused_variant == "full")
                estimator_fn = {
                    "full": estimator_temb_fused_npu,
                    "hybrid": estimator_temb_hybrid_npu,
                    "qkv": estimator_temb_qkv_npu,
                }[t2w_npu_fused_variant]
            self.t2w_batch.euler = EulerGraphs(
                self.t2w.flow.decoder,
                estimator_fn,
                next(est.parameters()).dtype,
                self.t2w_batch.dtype,
                self.t2w.n_timesteps,
                graph_t=tuple(dict.fromkeys(STEADY_T + FINAL_T + SETUP_T)),
            )
        # Eager flow caches are large at long reference lengths.  Eight rows
        # keep one Euler step's workspace bounded while preserving useful NPU
        # batching; completed microbatches retain only request-owned caches.
        self.t2w_batch.max_flow_streams = 16 if float16_flow else 8
        self.t2w_npu_fused_body = bool(t2w_npu_fused_body)
        self.t2w_npu_fused_variant = str(t2w_npu_fused_variant) if t2w_npu_fused_body else "reference"
        self.t2w_batch.offload_flow_cache = bool(t2w_cache_offload)
        self.frontend = MultimodalFrontend(model_dir, device=device)
        gc.collect()
        torch.npu.empty_cache()
        self.tkv = NpuPagedKV(self.tspec, thinker_pages, page_size, device=device)
        self.akv = NpuPagedKV(self.aspec, talker_pages, page_size, device=device)
        self.tkv.host_qo_lengths = bool(decoder_host_qo_lengths)
        self.akv.host_qo_lengths = bool(decoder_host_qo_lengths)
        self.trun = NpuDecoderRunner(
            self.thinker.w,
            self.tspec,
            self.tkv,
            layers=thinker_layers,
            fused_ops=fused_decoder_ops,
            fused_mode=thinker_decoder_op_mode,
            use_graphs=use_graphs,
            graph_max_batch=graph_max_batch,
            graph_max_blocks=64,
            batch_append_prefill=thinker_prefix_cache,
        )
        self.arun = NpuDecoderRunner(
            self.talker.w,
            self.aspec,
            self.akv,
            layers=talker_layers,
            fused_ops=fused_decoder_ops,
            fused_mode=talker_decoder_op_mode,
            use_graphs=use_graphs,
            graph_max_batch=graph_max_batch,
            graph_max_blocks=32,
            static_shape_attention=talker_static_shape_attention,
            static_shape_capacity=talker_static_kv_capacity,
        )
        self.trun.capture_context = self._graph_capture_context
        self.arun.capture_context = self._graph_capture_context
        if self.t2w_batch.euler is not None:
            self.t2w_batch.euler.capture_context = self._graph_capture_context
        if self.t2w_batch.hift_graphs is not None:
            self.t2w_batch.hift_graphs.capture_context = self._graph_capture_context
        torch.npu.empty_cache()
        if int(thinker_prompt_lookup_max_tokens) < 0:
            raise ValueError("thinker_prompt_lookup_max_tokens must be non-negative")
        self.thinker_prompt_lookup_max_tokens = int(thinker_prompt_lookup_max_tokens)
        self.thinker_prompt_lookup_stats = dict(verification_calls=0,draft_tokens=0,matched_draft_tokens=0,target_tokens=0,kv_fallbacks=0)
        self.device, self.max_live = device, max_live
        if int(talker_min_new_tokens) < 0:
            raise ValueError("talker_min_new_tokens must be >= 0")
        if talker_sampling_semantics not in ("hf", "omni"):
            raise ValueError("talker_sampling_semantics must be 'hf' or 'omni'")
        if talker_sampling_rng not in ("auto", "native", "counter"):
            raise ValueError("talker_sampling_rng must be 'auto', 'native', or 'counter'")
        if talker_sampling_rng == "native" and talker_sampling_semantics != "omni":
            raise ValueError("native Talker RNG is only supported with Omni sampling semantics")
        self.cfg = TalkerSampling(
            min_new_tokens=int(talker_min_new_tokens),
            max_new_tokens=1 << 30,
            semantics=talker_sampling_semantics,
            rng=talker_sampling_rng,
        )
        self.talker_k, self.talker_k_max = int(talker_k), int(talker_k_max)
        self.talker_device_blocks = bool(talker_device_blocks)
        self.talker_post_graphs = None
        if talker_post_graphs:
            if self.cfg.uses_native_rng:
                raise ValueError("talker_post_graphs do not preserve native per-request RNG state")
            from engine.runtime.talker_post_graph import NpuTalkerPostGraphs

            self.talker_post_graphs = NpuTalkerPostGraphs(self.talker.head, self.talker.cfg.eos, self.cfg)
            self.talker_post_graphs.capture_context = self._graph_capture_context
        self.first_chunk = int(first_chunk)
        self.steady_chunk = int(steady_chunk)
        # Track-S helper methods use these capacity views.  The NPU runners
        # keep the actual paged state; no CUDA decoder object is constructed.
        self.tdec = SimpleNamespace(max_len=8192)
        self.adec = SimpleNamespace(max_len=8192)
        self.talker_pad_pages = 0
        self.talker_commit: dict[int, int] = {}
        self.admitted = 0
        self.stage_profile_enabled = bool(profile_stages)
        self.profile = self.stage_profile_enabled or bool(timeline_markers)
        self.t2w_batch.profile = {} if self.stage_profile_enabled else None
        self.t2w.profile_features = self.stage_profile_enabled
        self.t2w.feature_profile = {}
        self.prompt_profile: dict[str, list] = {}
        self.stage_trace_dir = Path(stage_trace_dir) if stage_trace_dir else None
        if self.stage_trace_dir is not None:
            self.stage_trace_dir.mkdir(parents=True, exist_ok=True)
        self.timelines = {}
        self.tick_profile = {}
        self.talker_full_block_graphs_enabled = bool(talker_full_block_graphs)
        # Exact row=3 full-block replay regresses the native C4 workload even
        # though it is numerically correct.  Keep that cohort on the original
        # eager graph-lane path; row=5/6/7 remove the measured C8/C16 fallback.
        self.talker_full_block_graph_batches = tuple(
            batch for batch in (1, 2, 4, 5, 6, 7, 8)
            if batch <= self.arun.graph_max_batch
        )
        self.talker_full_block_graphs = {}
        self.talker_full_block_replays = {}
        self.talker_full_block_warmup_s = {}
        self.split_threads = False
        self.tick_overlap = False
        self.t2wp = None
        self.audio_live = {}
        self.thinker_prefix_cache = bool(thinker_prefix_cache)
        self.thinker_prefix_cache_min_tokens = int(thinker_prefix_cache_min_tokens)
        self.thinker_prefix_cache_min_live = int(thinker_prefix_cache_min_live)
        # Negative graph-padding sequences start at -1.  Keep the retained
        # prompt KV well outside that namespace.
        self._thinker_prefix_seq = -1_000_000_001
        self._thinker_prefix_ids: tuple[int, ...] | None = None
        self._thinker_prefix_stats = {
            "hits": 0,
            "misses": 0,
            "tokens_reused": 0,
            "tokens_computed": 0,
            "requests_bypassed_below_min_live": 0,
        }
        self.sched = {"prefill_token_budget": None, "prefill_tokens_tick_max": 0,
                      "prefill_chunked_ticks": 0, "prefill_kv_wait_ticks": 0,
                      "talker_starts_deferred_ticks": 0, "talker_kv_wait_ticks": 0}
        self.pool = ThreadPoolExecutor(max_workers=max(1, int(prepare_workers)), thread_name_prefix="npu-t2w-prepare")
        self._t2w_lock = threading.Lock()
        self.prompt_setup_batching = bool(prompt_setup_batching)
        self.prompt_setup_wait_s = float(prompt_setup_wait_ms) / 1000.0
        self.prompt_setup_inbox: queue.Queue = queue.Queue()
        self.prompt_setup_batches: dict[int, int] = {}
        self.t2w_threaded = bool(t2w_threaded)
        self.t2w_batch_wait_s = float(t2w_batch_wait_ms) / 1000.0
        self.t2w_batch_wait_min_active = int(t2w_batch_wait_min_active)
        self.t2w_batch_min_ready = int(t2w_batch_min_ready) if t2w_batch_min_ready is not None else None
        self.prompt_stream = torch.npu.Stream(device=device)
        self.t2w_inbox: queue.Queue = queue.Queue()
        self.t2w_live: dict[int, LiveS] = {}
        self.t2w_wake = threading.Event()
        self.t2w_stream = torch.npu.Stream(device=device) if self.t2w_threaded else None
        self._prompt_cache_lock = threading.Lock()
        self._prompt_futures = {}
        self._prompt_refs = {}
        self.inbox: queue.Queue = queue.Queue()
        self.live: dict[int, LiveS] = {}
        self._next = 0
        self._lock = threading.Lock()
        if talker_full_block_warmup:
            self._warm_full_talker_blocks()
        self.t2w_startup_graph_rows = startup_rows
        self.t2w_startup_extended = bool(t2w_startup_extended)
        self.t2w_startup_ready = threading.Event()
        self.t2w_startup_error = None
        self.t2w_startup_stats = {"request_rows": list(startup_rows), "keys": [], "duration_s": 0.0}
        self.thread = threading.Thread(target=self._run, daemon=True, name="npu-engine")
        self.t2w_first_audio_priority = bool(t2w_first_audio_priority)
        self.t2w_first_audio_priority_stats = {"batches": 0, "reordered": 0}
        self.thread.start()
        if self.prompt_setup_batching:
            self.prompt_setup_thread = threading.Thread(
                target=self._run_prompt_setups,
                daemon=True,
                name="npu-prompt-setup",
            )
            self.prompt_setup_thread.start()
        if self.t2w_threaded:
            self.t2w_thread = threading.Thread(target=self._run_t2w, daemon=True, name="npu-token2wav")
            self.t2w_thread.start()
            if startup_rows:
                if not self.t2w_startup_ready.wait(180):
                    raise RuntimeError("media graph initialization exceeded 180 seconds")
                if self.t2w_startup_error is not None:
                    raise RuntimeError("media graph initialization failed") from self.t2w_startup_error
                if self.t2w_startup_extended:
                    self.pool.submit(self._warm_media_setup_graphs).result(timeout=180)



    def _prime_euler_key(self, rows, frames, cache, lane):
        import time

        euler = self.t2w_batch.euler
        if not euler.supports(rows, frames, cache):
            raise ValueError("unsupported extended Euler initialization shape")
        then = time.perf_counter()
        w = euler.views(rows, frames, cache)
        for name in ("x0", "mu", "spks", "cond", "cnn", "att"):
            w[name].zero_()
        euler.capture(rows, frames, cache)
        euler.graphs[(rows, frames, cache)][0].replay()
        torch.npu.synchronize()
        record = {"owner": "EulerGraphs", "key": [rows, frames, cache], "lane": lane, "seconds": time.perf_counter() - then}
        self.t2w_startup_stats["keys"].append(record)
        print({"media_graph_startup": record}, flush=True)

    def _warm_media_setup_graphs(self):
        # Prompt setup owns the same arena under the existing scratch lock,
        # but replays on prompt_stream, not the audio stream. Initialize its
        # bounded keys on that same stream in the actual preparation pool.
        import time

        started = time.perf_counter()
        torch.npu.set_device(torch.device(self.device).index or 0)
        with self._t2w_lock, torch.npu.stream(self.prompt_stream), torch.inference_mode():
            for frames in (256, 512):
                self._prime_euler_key(2, frames, 0, "prompt_setup")
        self.t2w_startup_stats["duration_s"] += time.perf_counter() - started
        self.t2w_startup_stats["allocated_after_bytes"] = int(torch.npu.memory_allocated())
        self.t2w_startup_stats["reserved_after_bytes"] = int(torch.npu.memory_reserved())

    def _warm_media_graphs_on_owner(self) -> None:
        """Capture a bounded generic first-window set on its eventual owner.

        Unlike prior main-thread warmup, capture/replay uses the audio worker's
        real NPU stream before health readiness and before admitting requests.
        Only deterministic zero inputs are used: no dataset or global RNG is
        consumed. All real inputs/caches are rebound by the existing solver.
        """
        import time

        started = time.perf_counter()
        before = int(torch.npu.memory_allocated())
        with self._t2w_lock, torch.npu.stream(self.t2w_stream), torch.inference_mode():
            for requests in self.t2w_startup_graph_rows:
                rows, frames, cache = 2 * requests, 50, 512
                euler = self.t2w_batch.euler
                if not euler.supports(rows, frames, cache):
                    raise ValueError("unsupported bounded Euler initialization shape")
                then = time.perf_counter()
                w = euler.views(rows, frames, cache)
                for name in ("x0", "mu", "spks", "cond", "cnn", "att"):
                    w[name].zero_()
                euler.capture(rows, frames, cache)
                euler.graphs[(rows, frames, cache)][0].replay()
                torch.npu.synchronize()
                record = {"owner": "EulerGraphs", "key": [rows, frames, cache], "seconds": time.perf_counter() - then}
                self.t2w_startup_stats["keys"].append(record)
                print({"media_graph_startup": record}, flush=True)
                hift = self.t2w_batch.hift_graphs
                if requests not in hift._rows(50):
                    raise ValueError("unsupported bounded HiFT initialization shape")
                then = time.perf_counter()
                hift._capture(requests, 50)
                hift.graphs[(requests, 50)][0].replay()
                torch.npu.synchronize()
                record = {"owner": "HiFTGraphs", "key": [requests, 50], "seconds": time.perf_counter() - then}
                self.t2w_startup_stats["keys"].append(record)
                print({"media_graph_startup": record}, flush=True)
                if self.t2w_startup_extended:
                    then = time.perf_counter()
                    hift._capture(requests, 58)
                    hift.graphs[(requests, 58)][0].replay()
                    torch.npu.synchronize()
                    record = {"owner": "HiFTGraphs", "key": [requests, 58], "seconds": time.perf_counter() - then}
                    self.t2w_startup_stats["keys"].append(record)
                    print({"media_graph_startup": record}, flush=True)
            if self.t2w_startup_extended:
                self._prime_euler_key(2, 56, 512, "audio_tail")
        self.t2w_startup_stats.update(duration_s=time.perf_counter() - started,
            allocated_before_bytes=before, allocated_after_bytes=int(torch.npu.memory_allocated()),
            reserved_after_bytes=int(torch.npu.memory_reserved()))

    def _warm_full_talker_blocks(self) -> None:
        """Build configured static buckets before starting service workers.

        These are deterministic dummy graph inputs, not dataset requests. No
        global RNG is consumed and all temporary sequence pages are released
        after synchronizing their last replay. Real inputs are rebound before
        every subsequent graph use, including their block tables and sampler
        state. Unsupported shapes still use the original correctness path.
        """
        if not (self.talker_full_block_graphs_enabled and self.cfg.rng == "counter"
                and self.arun.use_graphs and self.arun.static_shape_attention):
            return
        import time

        from engine.runtime.talker_block_graph import NpuTalkerBlockGraph
        from engine.talker_sampler import DecodeState

        with torch.inference_mode():
            for batch in self.talker_full_block_graph_batches:
                seqs = [-10_000 - index for index in range(batch)]
                if not self.akv.can_fit({seq: self.talker_k for seq in seqs}):
                    continue
                started = time.perf_counter()
                try:
                    for seq in seqs:
                        self.akv.reserve(seq, self.talker_k)
                    state = DecodeState.new(list(range(batch)), self.cfg, self.device)
                    feeds = torch.zeros(batch, dtype=torch.long, device=self.device)
                    graph = NpuTalkerBlockGraph(self.arun, self.talker.code_embed, self.talker.head,
                                                self.talker.cfg.eos, self.cfg, batch, self.talker_k)
                    graph.run(feeds, state, [0] * batch, [list(self.akv.pages[seq]) for seq in seqs])
                    torch.npu.synchronize()
                    self.talker_full_block_graphs[(batch, self.talker_k)] = graph
                    self.talker_full_block_warmup_s[batch] = time.perf_counter() - started
                finally:
                    # No replay may retain a write to a returned physical page.
                    torch.npu.synchronize()
                    for seq in seqs:
                        if seq in self.akv.pages:
                            self.akv.release(seq)

    def _prepare(self, ref: bytes):
        """Prepare one immutable reference prompt for a shared future."""
        # Reference feature extraction does not touch Token2Wav's shared flow
        # scratch.  Queue it before taking the decode lock so CPU preprocessing
        # and independent NPU kernels can overlap ready-window decoding.
        with self._graph_gate.launch(), torch.npu.stream(self.prompt_stream):
            if self.stage_profile_enabled:
                import time

                self.prompt_stream.synchronize()
                started = time.perf_counter()
                feats = self.t2w.features(ref, defer_speaker=True)
                self.prompt_stream.synchronize()
                self._record_prompt_stage("features", time.perf_counter() - started)
            else:
                feats = self.t2w.features(ref, defer_speaker=True)

        if not self.prompt_setup_batching:
            return self._setup_prompts([feats])[0]

        item = SimpleNamespace(feats=feats, prompt=None, error=None, ready=threading.Event())
        self.prompt_setup_inbox.put(item)
        item.ready.wait()
        if item.error is not None:
            raise item.error
        return item.prompt

    def _setup_prompts(self, feats_list: list[tuple]) -> list:
        """Run lossless batched prompt setup under the shared flow-scratch lock."""
        with self._t2w_lock:
            with self._graph_gate.launch(), torch.npu.stream(self.prompt_stream):
                if self.stage_profile_enabled:
                    import time

                    self.prompt_stream.synchronize()
                    started = time.perf_counter()
                    prompts = self.t2w_batch.setup_many(feats_list)
                    self.prompt_stream.synchronize()
                    self._record_prompt_stage("setup", time.perf_counter() - started)
                else:
                    prompts = self.t2w_batch.setup_many(feats_list)
                    self.prompt_stream.synchronize()
            if self.t2w_batch.offload_flow_cache:
                # A Seed-TTS wave has different reference voices.  Keeping
                # every prepared prompt cache on the NPU would scale memory
                # with concurrency before its first window is decoded.  CPU
                # cache tensors are copied back by BatchedToken2Wav per flow
                # microbatch, just like the request-owned continuation cache.
                for prompt in prompts:
                    prompt.flow_cache = {
                        name: value.cpu() if isinstance(value, torch.Tensor) else value
                        for name, value in prompt.flow_cache.items()
                    }
        return prompts

    def _run_prompt_setups(self) -> None:
        """Coalesce ready reference features by graph bucket and row limit."""
        import time

        from engine.runtime.dit_graph import ROW_LIMIT, SETUP_T

        torch.npu.set_device(torch.device(self.device).index or 0)
        while True:
            first = self.prompt_setup_inbox.get()
            items = [first]
            deadline = time.perf_counter() + self.prompt_setup_wait_s
            while True:
                left = deadline - time.perf_counter()
                if left <= 0:
                    break
                try:
                    items.append(self.prompt_setup_inbox.get(timeout=left))
                except queue.Empty:
                    break
            while True:
                try:
                    items.append(self.prompt_setup_inbox.get_nowait())
                except queue.Empty:
                    break

            try:
                groups: dict[int, list] = {}
                for item in items:
                    frames = item.feats[3].shape[1]
                    bucket = next((value for value in SETUP_T if value >= frames), -frames)
                    groups.setdefault(bucket, []).append(item)
            except Exception as exc:
                for item in items:
                    item.error = exc
                    item.ready.set()
                continue
            for bucket, group in groups.items():
                per = max(1, ROW_LIMIT.get(bucket, 2) // 2)
                for offset in range(0, len(group), per):
                    part = group[offset:offset + per]
                    try:
                        prompts = self._setup_prompts([item.feats for item in part])
                        if len(prompts) != len(part):
                            raise RuntimeError(
                                f"prompt setup returned {len(prompts)} prompts for {len(part)} inputs"
                            )
                    except Exception as exc:
                        for item in part:
                            item.error = exc
                            item.ready.set()
                        continue
                    self.prompt_setup_batches[len(part)] = self.prompt_setup_batches.get(len(part), 0) + 1
                    for item, prompt in zip(part, prompts):
                        item.prompt = prompt
                        item.ready.set()

    @contextmanager
    def _graph_capture_context(self):
        """Drain other launchers around GLOBAL NPUGraph capture only."""
        with self._graph_gate.capture():
            torch.npu.synchronize()
            try:
                yield
            finally:
                torch.npu.synchronize()

    def _record_prompt_stage(self, name: str, seconds: float) -> None:
        acc = self.prompt_profile.setdefault(name, [0, 0.0])
        acc[0] += 1
        acc[1] += seconds

    def _timed_stage(self, name: str, fn, *args):
        """Synchronizing diagnostic timing, completely disabled in normal serving."""
        if not self.stage_profile_enabled:
            return fn(*args)
        import time

        torch.npu.current_stream().synchronize()
        started = time.perf_counter()
        result = fn(*args)
        torch.npu.current_stream().synchronize()
        acc = self.tick_profile.setdefault(name, [0, 0.0])
        acc[0] += 1
        acc[1] += time.perf_counter() - started
        return result

    def _decoder_forward(self, runner, x, layout):
        """Forward through the decoder; its lazy captures upgrade the launch gate."""
        return runner.forward(x, layout)

    def _acquire_prompt(self, ref: bytes):
        """Return ``(key, shared future)`` and account one waiting request."""
        key = hashlib.sha256(ref).digest()
        with self._prompt_cache_lock:
            future = self._prompt_futures.get(key)
            if future is None:
                future = self.pool.submit(self._prepare, ref)
                self._prompt_futures[key] = future
            self._prompt_refs[key] = self._prompt_refs.get(key, 0) + 1
        return key, future

    def _submit_prompt(self, ref: bytes):
        return self._acquire_prompt(ref)

    def _release_prompt_key(self, key: bytes | None) -> None:
        if key is None:
            return
        with self._prompt_cache_lock:
            refs = self._prompt_refs[key] - 1
            if refs:
                self._prompt_refs[key] = refs
            else:
                self._prompt_refs.pop(key, None)
                self._prompt_futures.pop(key, None)

    def _release_prompt(self, lv: LiveS) -> None:
        key, lv.prompt_key = lv.prompt_key, None
        lv.prompt_future = None
        self._release_prompt_key(key)

    def _admit(self, lv: LiveS) -> None:
        import time

        from engine.runtime.t2w_sched import AudioState
        from engine.token2wav_chunker import StreamChunker

        lv.t_admit, lv.audio = time.perf_counter(), AudioState()
        lv.mark("admitted", lv.t_admit)
        if lv.wants_audio:
            lv.chunker = StreamChunker(first_chunk=self.first_chunk, chunk=self.steady_chunk)
        self.live[lv.seq] = lv
        if self.t2w_threaded and lv.wants_audio:
            self.t2w_inbox.put(lv)
            self.t2w_wake.set()

    def _release(self, lv: LiveS) -> None:
        self._release_prompt(lv)
        self.tkv.release(lv.seq)
        self.akv.release(lv.seq)
        self.talker_commit.pop(lv.seq, None)
        self.live.pop(lv.seq, None)
        lv.finished = True

    def stats(self) -> dict:
        import resource

        from engine.runtime.backend import _rss_kib

        with self._prompt_cache_lock:
            prompt_entries, prompt_refs = len(self._prompt_futures), sum(self._prompt_refs.values())
        profile_stats = {}
        if self.stage_profile_enabled:
            timelines = {}
            for request_id, events in self.timelines.items():
                origin = events[0][1] if events else 0.0
                timelines[request_id] = [[name, timestamp - origin] for name, timestamp in events]
            profile_stats = {
                "profile_tick": dict(self.tick_profile),
                "profile_prompt": dict(self.prompt_profile),
                "profile_t2w": dict(self.t2w_batch.profile or {}),
                "profile_features": dict(self.t2w.feature_profile),
                "profile_timelines": timelines,
            }
        return {"live": len(self.live), "queued": self.inbox.qsize(), "admitted": self.admitted,
                "max_live": self.max_live, "first_chunk": self.first_chunk, "steady_chunk": self.steady_chunk,
                "t2w_cache_offload": self.t2w_batch.offload_flow_cache,
                "t2w_timesteps": self.t2w.n_timesteps,
                "t2w_euler_graph_failures": ({str(k): v for k, v in self.t2w_batch.euler.failures.items()}
                                               if self.t2w_batch.euler else {}),
                "t2w_npu_fused_body": self.t2w_npu_fused_body,
                "t2w_npu_fused_variant": self.t2w_npu_fused_variant,
                "t2w_hift_graph_failures": ({str(k): v for k, v in self.t2w_batch.hift_graphs.failures.items()}
                                              if self.t2w_batch.hift_graphs else {}),
                "t2w_encoder_graph_failures": ({str(k): v for k, v in self.t2w_batch.enc_graphs.failures.items()}
                                                 if self.t2w_batch.enc_graphs else {}),
                "t2w_offload_stats": dict(self.t2w_batch.offload_stats),
                "t2w_flow_batch_hist": {str(k): v for k, v in sorted(self.t2w_batch.flow_batch_hist.items())},
                "t2w_flow_cache_bucket_hist": {
                    str(k): v for k, v in sorted(self.t2w_batch.flow_cache_bucket_hist.items())
                },
                "t2w_threaded": self.t2w_threaded, "t2w_live": len(self.t2w_live),
                "t2w_batch_wait_ms": self.t2w_batch_wait_s * 1000.0,
                "t2w_batch_wait_min_active": self.t2w_batch_wait_min_active,
                "t2w_batch_min_ready": self.t2w_batch_min_ready,
                "prompt_setup_batches": dict(self.prompt_setup_batches),
                "talker_graph_ring_failures": {str(k): v for k, v in self.arun.graph_ring_failures.items()},
                "talker_graph_ring_capture_retries": {
                    str(k): v for k, v in self.arun.graph_ring_capture_retries.items()
                },
                "talker_graph_ring_capture_retry_errors": {
                    str(k): v for k, v in self.arun.graph_ring_capture_retry_errors.items()
                },
                "talker_static_shape_attention": self.arun.static_shape_attention,
                "talker_static_kv_capacity": self.arun.static_shape_capacity,
                "talker_static_graph_buckets": sorted(self.arun.static_decode_graphs),
                "talker_static_graph_failures": {
                    str(k): v for k, v in self.arun.static_graph_failures.items()
                },
                "talker_post_graph_failures": (dict(self.talker_post_graphs.failures)
                                                if self.talker_post_graphs is not None else {}),
                "talker_post_graph_replays": (dict(self.talker_post_graphs.replays)
                                               if self.talker_post_graphs is not None else {}),
                "talker_full_block_graphs_enabled": self.talker_full_block_graphs_enabled,
                "talker_full_block_graph_batches": list(self.talker_full_block_graph_batches),
                "talker_full_block_graphs": [list(key) for key in self.talker_full_block_graphs],
                "talker_full_block_replays": dict(self.talker_full_block_replays),
                "talker_full_block_warmup_s": dict(self.talker_full_block_warmup_s),
                "t2w_first_audio_priority": self.t2w_first_audio_priority,
                "t2w_first_audio_priority_stats": dict(self.t2w_first_audio_priority_stats),
                "thinker_graph_buckets": sorted(self.trun.decode_graphs),
                "thinker_prompt_lookup_stats": dict(self.thinker_prompt_lookup_stats),
                "t2w_startup_stats": dict(self.t2w_startup_stats),
                "npu_core_policy": self.npu_core_policy,
                "thinker_prefix_cache_enabled": self.thinker_prefix_cache,
                "thinker_prefix_cache_tokens": len(self._thinker_prefix_ids or ()),
                "thinker_prefix_cache_min_live": self.thinker_prefix_cache_min_live,
                "thinker_prefix_cache_stats": dict(self._thinker_prefix_stats),
                "thinker_batch_append_prefill": self.trun.batch_append_prefill,
                "thinker_batch_append_prefill_failures": list(self.trun.batch_append_prefill_failures),
                "talker_graph_buckets": sorted(self.arun.decode_graphs),
                "prompt_cache_entries": prompt_entries, "prompt_cache_refs": prompt_refs,
                "thinker_kv_pages_used": self.tkv.used_pages,
                "talker_kv_pages_used": self.akv.used_pages, "thinker_kv_pages": self.tkv.num_pages,
                "talker_kv_pages": self.akv.num_pages, "npu_memory_allocated_bytes": int(torch.npu.memory_allocated()),
                "npu_memory_reserved_bytes": int(torch.npu.memory_reserved()),
                "rss_max_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, "rss_kib": _rss_kib(),
                "live_requests": self._live_diagnostics(),
                **profile_stats, **self.sched}

    def _run(self) -> None:
        torch.npu.set_device(torch.device(self.device).index or 0)
        while True:
            try:
                while True:
                    live = self.inbox.get(block=not self.live, timeout=None if not self.live else 0)
                    try:
                        self._admit(live)
                    except Exception as exc:
                        live.emit(RuntimeError(f"engine admission error: {type(exc).__name__}: {exc}"))
            except queue.Empty:
                pass
            try:
                with torch.inference_mode():
                    self._tick()
            except Exception as exc:
                traceback.print_exc()
                for live in list(self.live.values()):
                    live.emit(RuntimeError(f"engine error: {type(exc).__name__}: {exc}"))
                    self._release(live)

    def _tick(self) -> None:
        with self._graph_gate.launch():
            self._tick_launched()

    def _tick_launched(self) -> None:
        for lv in [x for x in self.live.values() if x.cancelled and not x.finished]:
            self._release(lv)
        self._timed_stage("thinker", self._thinker_step, [x for x in self.live.values() if not x.thinker_done])
        self._timed_stage("talker", self._talker_step,
                          [x for x in self.live.values() if x.wants_audio and not x.utterance_done])
        if not self.t2w_threaded:
            self._timed_stage("token2wav", self._drain_windows)
            for lv in [x for x in list(self.live.values()) if x.utterance_done and not x.windows]:
                if lv.stream is None and lv.prompt_future is not None and not lv.prompt_future.done():
                    continue
                lv.emit(self._done(lv))
                self._release(lv)
        else:
            # Audio completion belongs to the Token2Wav thread.  Compute KV
            # can be released as soon as the Talker has queued its final window.
            for lv in [x for x in list(self.live.values()) if x.utterance_done]:
                self.tkv.release(lv.seq)
                self.akv.release(lv.seq)
                self.talker_commit.pop(lv.seq, None)
                self.live.pop(lv.seq, None)
                self.t2w_wake.set()

    @staticmethod
    def _done(lv: LiveS):
        from engine.server import Done

        return Done(lv.finish_reason, {"prompt_tokens": len(lv.prompt_ids),
                                      "completion_tokens": len(lv.generated),
                                      "audio_frames": lv.frames})

    @staticmethod
    def _common_prefix_length(left: tuple[int, ...], right: list[int]) -> int:
        limit = min(len(left), len(right))
        for index in range(limit):
            if left[index] != right[index]:
                return index
        return limit

    def _thinker_step_group(self, rows: list[LiveS], prefix_lengths: dict[int, int] | None = None, *, _skip_prompt_lookup: bool = False) -> None:
        if not rows:
            return
        if (self.thinker_prompt_lookup_max_tokens and not _skip_prompt_lookup
                and not prefix_lengths and not any(lv.pending_prefill for lv in rows)):
            from engine.runtime.thinker_lookup import try_prompt_lookup_step
            if try_prompt_lookup_step(self, rows):
                return
        prefix_lengths = prefix_lengths or {}
        feeds = [
            lv.prompt_ids[prefix_lengths.get(lv.seq, 0):] if lv.pending_prefill else [lv.generated[-1]]
            for lv in rows
        ]
        lay = self.tkv.layout([lv.seq for lv in rows], [len(f) for f in feeds])
        embs = []
        for lv, feed in zip(rows, feeds):
            if lv.pending_prefill and lv.prompt_embeds is not None:
                start = prefix_lengths.get(lv.seq, 0)
                embs.append(lv.prompt_embeds[start:].to(self.thinker.dtype))
            else:
                embs.append(self.thinker.embed(torch.tensor(feed, dtype=torch.int64, device=self.device)))
        # torch-npu's GLOBAL graph-capture mode forbids host/device copies on
        # other threads.  Reference preparation uses the same lock, so the
        # one-time capture cannot race Token2Wav setup.
        hidden = self._decoder_forward(self.trun, torch.cat(embs), lay)
        last = hidden[lay.last_token_index]
        nxt = torch.nn.functional.linear(last, self.thinker.w["lm_head.weight"]).float().argmax(-1).tolist()
        from engine.server import TextDelta
        from engine.runtime.backend import IM_END, TTS_EOS

        # Seed the generic cache from the first ordinary text prompt.  Its KV
        # is copied before any generated token is appended.  Later prompts
        # discover their reusable length by token LCP; no request IDs or
        # benchmark samples participate in the decision.
        if self.thinker_prefix_cache and self._thinker_prefix_ids is None:
            source = next((lv for lv in rows if lv.pending_prefill and lv.prompt_embeds is None), None)
            if source is not None:
                self.tkv.copy_prefix(source.seq, self._thinker_prefix_seq, len(source.prompt_ids))
                self._thinker_prefix_ids = tuple(source.prompt_ids)

        for lv, h, tok in zip(rows, last, nxt):
            if lv.pending_prefill:
                lv.pending_prefill = False
                lv.prefilled = len(lv.prompt_ids)
                lv.mark("thinker_prefilled")
            else:
                lv.hidden.append(h)
            if tok in (TTS_EOS, IM_END):
                self._close_thinker(lv)
                continue
            lv.generated.append(tok)
            lv.emit(TextDelta(self.tok.decode([tok])))
            boundary = THINKER_CHUNK + 1 + THINKER_CHUNK * lv.chunk_index
            if len(lv.generated) == boundary:
                final = lv.chunk_index == -(-lv.req.max_tokens // THINKER_CHUNK) - 1
                self._pass_chunk(lv, final=final)
                lv.chunk_index += 1
                if final:
                    self._close_thinker(lv, passed_final=True)

    def _thinker_step(self, rows: list[LiveS]) -> None:
        if not rows:
            return
        if not self.thinker_prefix_cache:
            self._thinker_step_group(rows)
            return
        if len(self.live) < self.thinker_prefix_cache_min_live:
            # Below the crossover point, preserve r246's original combined
            # prefill/decode step and eager append fallback exactly.  The
            # retained cache may still be seeded during warmup for a later
            # high-active cohort.
            self._thinker_prefix_stats["requests_bypassed_below_min_live"] += sum(
                lv.pending_prefill for lv in rows
            )
            batch_append = self.trun.batch_append_prefill
            cache_enabled = self.thinker_prefix_cache
            self.trun.batch_append_prefill = False
            self.thinker_prefix_cache = False
            try:
                self._thinker_step_group(rows)
            finally:
                self.trun.batch_append_prefill = batch_append
                self.thinker_prefix_cache = cache_enabled
            return

        prefills = [lv for lv in rows if lv.pending_prefill]
        decodes = [lv for lv in rows if not lv.pending_prefill]
        prefix_lengths: dict[int, int] = {}
        cached = self._thinker_prefix_ids
        for lv in prefills:
            prefix = 0
            if cached is not None and lv.prompt_embeds is None and len(lv.prompt_ids) > 1:
                prefix = min(self._common_prefix_length(cached, lv.prompt_ids), len(lv.prompt_ids) - 1)
            if prefix >= self.thinker_prefix_cache_min_tokens:
                self.tkv.copy_prefix(self._thinker_prefix_seq, lv.seq, prefix)
                prefix_lengths[lv.seq] = prefix
                self._thinker_prefix_stats["hits"] += 1
                self._thinker_prefix_stats["tokens_reused"] += prefix
                self._thinker_prefix_stats["tokens_computed"] += len(lv.prompt_ids) - prefix
            else:
                self._thinker_prefix_stats["misses"] += 1
                self._thinker_prefix_stats["tokens_computed"] += len(lv.prompt_ids)

        # Keep prefill and one-token decode in their native attention modes.
        # Mixing them would route every decode row through append-prefill.
        self._thinker_step_group(prefills, prefix_lengths)
        self._thinker_step_group(decodes)

    def _talker_step(self, rows: list[LiveS]) -> None:
        if not rows:
            return
        from engine.talker_sampler import PAD

        eos = self.talker.cfg.eos
        starts = [lv for lv in rows if not lv.talking and lv.chunks]
        if starts:
            feeds = []
            for lv in starts:
                ch = lv.chunks[0]
                ids = torch.tensor(ch.ids, dtype=torch.int64, device=self.device)
                cond = self.talker.condition(ids, ch.hidden)
                if not ch.final:
                    cond = torch.cat([cond[:-2], cond[-1:]], dim=0)
                if lv.feed_last:
                    cond = torch.cat([self.talker.code_embed(lv.ts["codes"]), cond], dim=0)
                    lv.feed_last = False
                feeds.append(cond)
            lay = self.akv.layout([lv.seq for lv in starts], [f.shape[0] for f in feeds])
            hidden = self._decoder_forward(self.arun, torch.cat(feeds), lay)
            logits = torch.nn.functional.linear(hidden[lay.last_token_index], self.talker.head).float()
            for lv in starts:
                if lv.ts is None:
                    generator = None
                    if self.cfg.uses_native_rng:
                        generator = torch.Generator(device=self.device)
                        generator.manual_seed(request_seed(lv.req))
                    lv.ts = {
                        "codes": torch.zeros(1, dtype=torch.int64, device=self.device),
                        "seeds": torch.tensor([request_seed(lv.req)], dtype=torch.int64, device=self.device),
                        "steps": torch.zeros(1, dtype=torch.int64, device=self.device),
                        "history": torch.full((1, self.cfg.window), PAD, dtype=torch.int64, device=self.device),
                        "live": torch.ones(1, dtype=torch.bool, device=self.device),
                        "generator": generator,
                    }
            tokens = self._sample_talker_rows(starts, logits, eos)
            for lv, tok in zip(starts, tokens):
                lv.chunk_frames = 0
                self._take_frame(lv, tok, fed_upto=None)

        if self.talker_device_blocks:
            self._talker_decode_block(rows, eos)
            return

        # K eager/graph replays per scheduling tick.  Rows that finish a text
        # chunk leave the batch immediately; the next chunk starts next tick.
        for _ in range(self.talker_k):
            active = [lv for lv in rows if lv.talking and not lv.cancelled]
            if not active:
                break
            feeds = [self.talker.code_embed(lv.ts["codes"]) for lv in active]
            lay = self.akv.layout([lv.seq for lv in active], [1] * len(active))
            hidden = self._decoder_forward(self.arun, torch.cat(feeds), lay)
            logits = torch.nn.functional.linear(hidden[lay.last_token_index], self.talker.head).float()
            tokens = self._sample_talker_rows(active, logits, eos)
            for lv, tok in zip(active, tokens):
                self._take_frame(lv, tok, fed_upto=self.akv.lengths[lv.seq])

    def _sample_talker_rows(self, rows: list[LiveS], logits: torch.Tensor, eos: int) -> list[int]:
        """Sample all active Talker rows with one device-to-host synchronization.

        ``talker_sampler.step`` is row-independent.  Counter RNG keeps the
        draw itself batched; native Omni RNG intentionally retains one
        request-local generator per row.  Both paths use one token copy back
        to the host instead of one queue drain per request and frame.
        """
        from engine.talker_sampler import DecodeState, step

        n = len(rows)
        state = DecodeState(
            torch.cat([lv.ts["seeds"] for lv in rows]),
            torch.cat([lv.ts["steps"] for lv in rows]),
            torch.cat([lv.ts["history"] for lv in rows]),
            torch.zeros(n, dtype=torch.bool, device=self.device),
            torch.ones(n, dtype=torch.bool, device=self.device),
            [lv.ts["generator"] for lv in rows] if self.cfg.uses_native_rng else None,
        )
        tokens = step(state, logits, eos, self.cfg)
        values = tokens.to("cpu").tolist()
        for i, lv in enumerate(rows):
            lv.ts.update(steps=state.steps[i:i + 1], history=state.history[i:i + 1])
        return values

    def _talker_decode_block(self, rows: list[LiveS], eos: int) -> None:
        """Run K dependent Talker frames with one device-to-host token copy.

        Rows that end inside the block are handled by ``_take_frame`` in the
        original order; speculative KV after EOS/a chunk boundary is trimmed
        with the same ``fed_upto`` contract used by the CUDA K-block path.
        """
        self._talker_decode_block_impl(rows, eos)

    def _talker_decode_block_impl(self, rows: list[LiveS], eos: int) -> None:
        from engine.talker_sampler import DecodeState, PAD, restore_multinomial_states_after_block, step

        active = [lv for lv in rows if lv.talking and not lv.cancelled]
        if not active:
            return
        starts_len = [self.akv.lengths[lv.seq] for lv in active]
        full_block_base = (
            self.talker_full_block_graphs_enabled
            and self.cfg.rng == "counter"
            and self.arun.use_graphs
            and self.arun.static_shape_attention
            and max(starts_len) + self.talker_k <= self.arun.static_shape_capacity
            and self.akv.can_fit({lv.seq: self.talker_k for lv in active})
        )
        # The counter sampler and decoder rows are request-independent.  Split
        # cohorts larger than the captured graph limit into stable, ordered
        # microbatches instead of sending the whole 9--32 row cohort through
        # ten host-issued graph lanes.  Avoid a trailing row=3 cohort by
        # changing the preceding 8+3 split to 7+4.  Processing groups in
        # request order preserves the original host window-publication order.
        if full_block_base and len(active) > self.arun.graph_max_batch:
            remaining = len(active)
            offset = 0
            while remaining:
                width = min(self.arun.graph_max_batch, remaining)
                if (remaining - width == 3 and width - 1 in self.talker_full_block_graph_batches
                        and 4 in self.talker_full_block_graph_batches):
                    width -= 1
                self._talker_decode_block_impl(active[offset:offset + width], eos)
                offset += width
                remaining -= width
            return
        state = DecodeState(
            torch.cat([lv.ts["seeds"] for lv in active]),
            torch.cat([lv.ts["steps"] for lv in active]),
            torch.cat([lv.ts["history"] for lv in active]),
            torch.zeros(len(active), dtype=torch.bool, device=self.device),
            torch.ones(len(active), dtype=torch.bool, device=self.device),
            [lv.ts["generator"] for lv in active] if self.cfg.uses_native_rng else None,
        )
        feeds = torch.cat([lv.ts["codes"] for lv in active])
        if full_block_base and len(active) in self.talker_full_block_graph_batches:
            from engine.runtime.talker_block_graph import NpuTalkerBlockGraph

            for lv in active:
                self.akv.reserve(lv.seq, self.talker_k)
            key = (len(active), self.talker_k)
            graph = self.talker_full_block_graphs.get(key)
            if graph is None:
                graph = NpuTalkerBlockGraph(self.arun, self.talker.code_embed, self.talker.head,
                                            eos, self.cfg, len(active), self.talker_k)
                self.talker_full_block_graphs[key] = graph
            tokens, state = graph.run(feeds, state, starts_len,
                                      [list(self.akv.pages[lv.seq]) for lv in active])
            host = tokens.cpu()
            self.talker_full_block_replays[len(active)] = self.talker_full_block_replays.get(len(active), 0) + 1
            for i, lv in enumerate(active):
                self.akv.lengths[lv.seq] = starts_len[i] + self.talker_k
                lv.ts.update(steps=state.steps[i:i + 1], history=state.history[i:i + 1])
                for j in range(host.shape[0]):
                    tok = int(host[j, i])
                    if tok == PAD:
                        break
                    self._take_frame(lv, tok, fed_upto=starts_len[i] + j + 1)
                    if not lv.talking:
                        break
            return
        tokens = []
        rng_states = []
        use_post_graph = self.talker_post_graphs is not None and self.talker_post_graphs.begin(state, frames=self.talker_k)
        for index in range(self.talker_k):
            lay = self.akv.layout([lv.seq for lv in active], [1] * len(active))
            hidden = self.arun.forward_graph_lane(
                self.talker.code_embed(feeds), lay, index, self.talker_k
            )
            if use_post_graph:
                token, feeds = self.talker_post_graphs.step(hidden[lay.last_token_index])
            else:
                logits = torch.nn.functional.linear(hidden[lay.last_token_index], self.talker.head).float()
                token = step(state, logits, eos, self.cfg)
                feeds = token.clamp_min(0)  # finished rows are ignored and later truncated; PAD is not an embedding id
            tokens.append(token)
            if state.generators is not None:
                rng_states.append([generator.get_state().clone() for generator in state.generators])
        if use_post_graph:
            self.talker_post_graphs.finish()
        host = torch.stack(tokens).cpu()
        if state.generators is not None:
            restore_multinomial_states_after_block(state.generators, rng_states, host)
        for i, lv in enumerate(active):
            lv.ts.update(steps=state.steps[i:i + 1], history=state.history[i:i + 1])
            for j in range(host.shape[0]):
                tok = int(host[j, i])
                if tok == PAD:
                    break
                self._take_frame(lv, tok, fed_upto=starts_len[i] + j + 1)
                if not lv.talking:
                    break

    def _queue_window(self, lv: LiveS, window) -> None:
        import time

        window.t_ready = time.perf_counter()
        lv.mark(f"window_ready:{window.index}", window.t_ready)
        lv.windows.append(window)
        if self.t2w_threaded:
            self.t2w_wake.set()

    def _run_t2w(self) -> None:
        """Decode ready windows on a dedicated NPU stream while the engine feeds Thinker/Talker."""
        torch.npu.set_device(torch.device(self.device).index or 0)
        if self.t2w_startup_graph_rows:
            try:
                self._warm_media_graphs_on_owner()
            except BaseException as error:
                traceback.print_exc()
                self.t2w_startup_error = error
            finally:
                self.t2w_startup_ready.set()
            if self.t2w_startup_error is not None:
                return
        while True:
            worked = False
            try:
                while True:
                    lv = self.t2w_inbox.get_nowait()
                    self.t2w_live[lv.seq] = lv
                    worked = True
            except queue.Empty:
                pass
            try:
                with torch.npu.stream(self.t2w_stream), torch.inference_mode():
                    for lv in list(self.t2w_live.values()):
                        if lv.cancelled:
                            self._release_prompt(lv)
                            self.t2w_live.pop(lv.seq, None)
                            continue
                        if lv.stream is None and lv.prompt_future is not None and lv.prompt_future.done():
                            try:
                                with self._t2w_lock:
                                    lv.stream = self.t2w_batch.new_stream(
                                        lv.prompt_future.result(), noise_seed=request_seed(lv.req), share_prompt=True)
                                    lv.mark("prompt_adopted")
                            except Exception as exc:
                                from engine.server import BadRequest

                                lv.emit(BadRequest(f"reference audio: {type(exc).__name__}: {exc}"))
                                lv.cancelled = True
                            finally:
                                self._release_prompt(lv)
                            worked = True
                    ready = [lv for lv in self.t2w_live.values()
                             if not lv.cancelled and lv.stream is not None and lv.windows]
                    # The engine thread can publish all C16/C32 windows within
                    # a few milliseconds.  Starting Token2Wav on the first
                    # singleton immediately turns that tiny skew into a
                    # multi-second head-of-line stall.  Keep coalescing through
                    # the configured C16/C32 tail threshold, always respecting
                    # the bounded deadline.  C1 is unchanged and no request
                    # data or model math is altered.
                    active = sum(not lv.cancelled and lv.stream is not None
                                 for lv in self.t2w_live.values())
                    target = min(self.t2w_batch.max_flow_streams, active)
                    wait_target = min(target, self.t2w_batch_min_ready or target)
                    if (ready and target >= self.t2w_batch_wait_min_active and len(ready) < wait_target
                            and self.t2w_batch_wait_s):
                        import time

                        deadline = time.perf_counter() + self.t2w_batch_wait_s
                        while len(ready) < wait_target:
                            left = deadline - time.perf_counter()
                            if left <= 0:
                                break
                            self.t2w_wake.clear()
                            self.t2w_wake.wait(min(left, 0.001))
                            ready = [lv for lv in self.t2w_live.values()
                                     if not lv.cancelled and lv.stream is not None and lv.windows]
                    if ready:
                        self._decode_windows_threaded(ready)
                        worked = True
                    for lv in list(self.t2w_live.values()):
                        if lv.utterance_done and not lv.windows and not lv.cancelled:
                            self.t2w_batch.release_stream(lv.stream)
                            lv.mark("done")
                            lv.emit(self._done(lv))
                            lv.finished = True
                            self.t2w_live.pop(lv.seq, None)
                            worked = True
            except Exception as exc:
                traceback.print_exc()
                for lv in list(self.t2w_live.values()):
                    lv.emit(RuntimeError(f"token2wav error: {type(exc).__name__}: {exc}"))
                    lv.cancelled = True
                    self.t2w_live.pop(lv.seq, None)
            if not worked:
                self.t2w_wake.wait(0.002)
                self.t2w_wake.clear()

    def _decode_windows_threaded(self, ready: list[LiveS]) -> None:
        from engine.server import AudioChunk
        import time

        if self.t2w_first_audio_priority:
            self.t2w_first_audio_priority_stats["batches"] += 1
            ordered = sorted(ready, key=lambda lv: lv.sample_offset != 0)
            if any(a is not b for a, b in zip(ready, ordered)):
                self.t2w_first_audio_priority_stats["reordered"] += 1
            ready = ordered
        items = []
        for lv in ready:
            window = lv.windows.pop(0)
            lv.mark(f"t2w_start:{window.index}")
            items.append((lv.stream, window.tokens, window.last))
        with self._t2w_lock, self._graph_gate.launch():
            wavs = self.t2w_batch.decode(items)
        now = time.perf_counter()
        for lv, wav in zip(ready, wavs):
            lv.mark("t2w_end", now)
            wav = wav.astype("float32", copy=False)
            if len(wav):
                if lv.sample_offset == 0:
                    lv.mark("first_audio", now)
                lv.audio.sent(len(wav), now)
                lv.emit(AudioChunk(wav, lv.sample_offset))
                lv.sample_offset += len(wav)

    def _drain_windows(self) -> None:
        from engine.server import AudioChunk, BadRequest

        for lv in list(self.live.values()):
            if not lv.wants_audio or lv.stream is not None or lv.prompt_future is None or not lv.prompt_future.done():
                continue
            try:
                lv.stream = self.t2w_batch.new_stream(lv.prompt_future.result(), noise_seed=request_seed(lv.req), share_prompt=True)
            except Exception as exc:
                lv.emit(BadRequest(f"reference audio: {type(exc).__name__}: {exc}"))
                lv.cancelled = True
            finally:
                self._release_prompt(lv)
        ready = [lv for lv in self.live.values() if not lv.cancelled and lv.stream is not None and lv.windows]
        if not ready:
            return
        items = []
        for lv in ready:
            window = lv.windows.pop(0)
            items.append((lv.stream, window.tokens, window.last))
        with self._t2w_lock:
            wavs = self.t2w_batch.decode(items)
        import time

        now = time.perf_counter()
        for lv, wav in zip(ready, wavs):
            wav = wav.astype("float32", copy=False)
            if len(wav):
                lv.audio.sent(len(wav), now)
                lv.emit(AudioChunk(wav, lv.sample_offset))
                lv.sample_offset += len(wav)
