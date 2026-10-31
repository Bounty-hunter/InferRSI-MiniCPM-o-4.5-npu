"""Reader/exclusive gate for lazy accelerator graph capture.

Ordinary launches from the engine, prompt-preparation, and Token2Wav streams
are readers and may overlap.  A lazy graph capture upgrades its own reader to
an exclusive writer, waits for the other launchers to drain, and restores the
reader on exit.  The upgrade is required because capture is discovered inside
an otherwise ordinary decoder call.
"""

from __future__ import annotations

import threading
from contextlib import contextmanager


class GraphLaunchGate:
    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._active_launches = 0
        self._capturing = False
        self._capture_owner: int | None = None
        self._local = threading.local()

    @contextmanager
    def launch(self):
        """Enter an ordinary accelerator-launch region (shared/read mode)."""
        owner = threading.get_ident()
        with self._condition:
            while self._capturing and self._capture_owner != owner:
                self._condition.wait()
            self._active_launches += 1
            self._local.launch_depth = getattr(self._local, "launch_depth", 0) + 1
        try:
            yield
        finally:
            with self._condition:
                self._active_launches -= 1
                self._local.launch_depth -= 1
                if self._active_launches == 0:
                    self._condition.notify_all()

    @contextmanager
    def capture(self):
        """Upgrade the caller's launch region to exclusive graph capture."""
        owner = threading.get_ident()
        nested = False
        own_launches = 0
        with self._condition:
            if self._capturing and self._capture_owner == owner:
                nested = True
                self._local.capture_depth = getattr(self._local, "capture_depth", 1) + 1
            else:
                own_launches = getattr(self._local, "launch_depth", 0)
                self._active_launches -= own_launches
                while self._capturing or self._active_launches:
                    self._condition.wait()
                self._capturing = True
                self._capture_owner = owner
                self._local.capture_depth = 1
        try:
            yield
        finally:
            with self._condition:
                if nested:
                    self._local.capture_depth -= 1
                else:
                    self._local.capture_depth = 0
                    self._capturing = False
                    self._capture_owner = None
                    self._active_launches += own_launches
                    self._condition.notify_all()
