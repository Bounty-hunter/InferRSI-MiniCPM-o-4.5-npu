"""Content fingerprints of everything an experiment's result depends on."""

from __future__ import annotations

import functools
import hashlib
import os
import re
import subprocess
from pathlib import Path


def _run(cmd: list[str], cwd: str | None = None) -> str:
    try:
        return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=120, check=True).stdout
    except (subprocess.SubprocessError, OSError):
        return ""


def file_sha256(path: str | Path) -> str | None:
    p = Path(path)
    if not p.is_file():
        return None
    h = hashlib.sha256()
    with p.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def tree_sha256(root: str | Path, subdirs: tuple[str, ...], suffixes: tuple[str, ...] = (".py", ".sh", ".yaml", ".json")) -> str:
    """Content hash of every source file under ``root/subdir`` (path + bytes), ignoring caches."""
    h = hashlib.sha256()
    root = Path(root)
    for sub in subdirs:
        for p in sorted((root / sub).rglob("*")):
            if p.is_file() and p.suffix in suffixes and "__pycache__" not in p.parts:
                h.update(str(p.relative_to(root)).encode())
                h.update(b"\0")
                h.update(p.read_bytes())
                h.update(b"\0")
    return h.hexdigest()


def repo_fingerprint(path: str | Path, paths: tuple[str, ...] = ()) -> dict:
    """Commit plus a hash of the tracked working-tree diff (dirty content, not just a flag), optionally limited to
    ``paths`` (the parts a measurement depends on)."""
    path = str(path)
    commit = _run(["git", "rev-parse", "HEAD"], cwd=path).strip()
    diff = _run(["git", "diff", "HEAD", "--binary", *(["--", *paths] if paths else [])], cwd=path)
    return {"commit": commit or None, "dirty": bool(diff.strip()), "diff_sha256": hashlib.sha256(diff.encode()).hexdigest()}


def installed_distributions_sha256() -> str:
    """The installed distributions (name==version, sorted) of the running interpreter.  The engine server reports it at
    /info, and ``env_distributions_sha256`` computes the same for a declared interpreter, by this same function."""
    import importlib.metadata as md

    dists = sorted({f"{(d.metadata['Name'] or '').lower()}=={d.version}" for d in md.distributions()})
    return hashlib.sha256("\n".join(dists).encode()).hexdigest()


@functools.lru_cache(maxsize=16)
def env_distributions_sha256(python: str) -> str | None:
    """``installed_distributions_sha256`` of the interpreter at ``python`` (run there); None if it cannot run."""
    root = str(Path(__file__).resolve().parents[1])
    out = _run([python, "-c", f"import sys; sys.path.insert(0, {root!r}); from bench.fingerprint import installed_distributions_sha256 as f; print(f())"])
    return out.strip() or None


@functools.lru_cache(maxsize=16)
def env_fingerprint(python: str) -> dict:
    frozen = _run(["uv", "pip", "freeze", "--python", python])
    if not frozen:
        frozen = _run([python, "-m", "pip", "freeze"])
    return {"python": python, "freeze_sha256": hashlib.sha256(frozen.encode()).hexdigest(), "packages": len(frozen.splitlines())}


def ascend_physical_device_index(index: int) -> int | None:
    """Map a process-local Ascend index to the physical ``npu-smi`` index.

    canhazgpu exposes an assigned physical card through
    ``ASCEND_RT_VISIBLE_DEVICES`` (for example ``2``), while frameworks address
    that single visible card as device 0.  ``npu-smi`` ignores the visibility
    mask and prints host-physical indices, so evidence and cleanup queries must
    translate the logical index before parsing its rows.
    """
    raw = os.environ.get("ASCEND_RT_VISIBLE_DEVICES")
    if raw is None:
        return index
    try:
        visible = [int(x) for x in re.split(r"[,;\s]+", raw.strip()) if x]
    except ValueError:
        return None
    if 0 <= index < len(visible):
        return visible[index]
    # Also accept an explicitly supplied physical index when it belongs to the
    # scheduler's visibility mask.  This keeps one-card diagnostic tools usable.
    return index if index in visible else None


def _npu_identity(index: int) -> dict:
    """Return a stable Ascend device identity from ``npu-smi info``.

    The scheduler exports the physical device id in
    ``ASCEND_RT_VISIBLE_DEVICES``.  ``npu-smi`` still prints the whole host, so
    parse the requested physical row and use its PCI bus id as the stable
    per-card identity.  Keeping the legacy ``gpu_*`` keys avoids invalidating
    the frozen result schema; ``accelerator`` makes their meaning explicit.
    """
    physical_index = ascend_physical_device_index(index)
    if physical_index is None:
        return {
            "gpu_index": index,
            "accelerator": "Ascend",
            "error": f"logical NPU {index} is outside ASCEND_RT_VISIBLE_DEVICES",
        }
    out = _run(["npu-smi", "info"])
    if not out:
        return {"gpu_index": index, "accelerator": "Ascend", "error": "npu-smi query failed"}
    version = re.search(r"npu-smi\s+(\S+)\s+Version:", out)
    header = re.search(
        rf"^\|\s*{physical_index}\s+(\S+)\s*\|\s*(\S+)\s*\|\s*([0-9.]+)\s+([0-9.]+)",
        out,
        re.MULTILINE,
    )
    detail = None
    if header:
        tail = out[header.end() :]
        detail = re.search(
            r"^\|\s*\d+\s*\|\s*([0-9A-Fa-f:.]+)\s*\|\s*([0-9.]+)\s+\d+\s*/\s*\d+\s+(\d+)\s*/\s*(\d+)",
            tail,
            re.MULTILINE,
        )
    if not header or not detail:
        return {
            "gpu_index": index,
            "physical_device_index": physical_index,
            "accelerator": "Ascend",
            "error": f"physical NPU {physical_index} not found in npu-smi output",
        }
    name, health, power, temperature = header.groups()
    bus_id, aicore, hbm_used, hbm_total = detail.groups()
    return {
        "gpu_index": index,
        "physical_device_index": physical_index,
        "gpu_sku": f"Ascend {name}",
        "gpu_uuid": f"ascend-pci-{bus_id.lower()}",
        "driver": version.group(1) if version else "unknown",
        "memory_mib": hbm_total,
        "accelerator": "Ascend",
        "pci_bus_id": bus_id.lower(),
        "health": health,
        "power_draw_w": float(power),
        "temperature_c": float(temperature),
        "aicore_percent": float(aicore),
        "memory_used_mib": float(hbm_used),
    }


def gpu_identity(index: int) -> dict:
    if os.environ.get("ASCEND_RT_VISIBLE_DEVICES") is not None:
        return _npu_identity(index)
    out = _run(["nvidia-smi", "-i", str(index), "--query-gpu=name,uuid,driver_version,memory.total", "--format=csv,noheader,nounits"]).strip()
    parts = [p.strip() for p in out.split(",")] if out else []
    if len(parts) < 4:
        return {"gpu_index": index, "error": "nvidia-smi query failed"}
    return {"gpu_index": index, "gpu_sku": parts[0].replace("NVIDIA ", ""), "gpu_uuid": parts[1], "driver": parts[2], "memory_mib": parts[3]}
