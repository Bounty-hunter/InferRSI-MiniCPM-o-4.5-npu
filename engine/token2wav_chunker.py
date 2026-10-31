"""Token2Wav stream chunking of Talker codec tokens, as in the HF reference stream driver.

Reference: ``MiniCPMO.streaming_generate`` in ``modeling_minicpmo.py`` (the
``s3tokenizer_step_audio`` branch): the buffer starts with three silence tokens
(4218); whenever it holds ``CHUNK + PRE_LOOKAHEAD`` = 28 tokens, those 28 are
decoded (25 new tokens plus 3 lookahead) and the buffer keeps the last 3
(``buffer = buffer[25:]``); at the end the remaining buffer (3 carried tokens
plus fewer than 25 new ones) is decoded as the last chunk.  The default remains
bit-for-bit window-compatible with that driver.  ``chunk`` may be increased by
the dedicated server to coalesce continuation windows after the first audio
event, reducing launch overhead without delaying TTFA.
"""

from __future__ import annotations

from dataclasses import dataclass, field

SILENCE_TOKEN = 4218
SILENCE_PREFIX = 3
PRE_LOOKAHEAD = 3
CHUNK = 25


@dataclass
class Window:
    tokens: list[int]
    last: bool
    index: int


@dataclass
class StreamChunker:
    """Split codec tokens into an initial window and continuation windows.

    ``first_chunk`` controls readiness of the first audio event.  ``chunk`` is
    the number of new codec frames in every continuation window; its default of
    25 exactly matches the HF stream driver.
    """

    buffer: list[int] = field(default_factory=lambda: [SILENCE_TOKEN] * SILENCE_PREFIX)
    emitted: int = 0
    closed: bool = False
    first_chunk: int = CHUNK
    chunk: int = CHUNK

    def __post_init__(self):
        if self.chunk < CHUNK:
            raise ValueError(f"chunk must be >= {CHUNK}")
        if not SILENCE_PREFIX < self.first_chunk <= self.chunk:
            raise ValueError(f"first_chunk must be in ({SILENCE_PREFIX}, {self.chunk}]")

    def push(self, tokens: list[int]) -> list[Window]:
        """Add codec tokens (never EOS); returns the full windows that became ready."""
        if self.closed:
            raise RuntimeError("chunker already flushed")
        out = []
        for t in tokens:
            self.buffer.append(int(t))
            size = self.first_chunk if self.emitted == 0 else self.chunk
            if len(self.buffer) >= size + PRE_LOOKAHEAD:
                out.append(Window(self.buffer[: size + PRE_LOOKAHEAD], False, self.emitted))
                self.emitted += 1
                self.buffer = self.buffer[size:]
        return out

    def finish(self) -> Window | None:
        """The final flush: carried lookahead plus the tail, decoded as the last chunk."""
        if self.closed:
            raise RuntimeError("chunker already flushed")
        self.closed = True
        if not self.buffer:
            return None
        w = Window(list(self.buffer), True, self.emitted)
        self.emitted += 1
        self.buffer = []
        return w
