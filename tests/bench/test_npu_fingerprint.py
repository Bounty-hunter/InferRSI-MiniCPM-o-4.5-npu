import hashlib
from types import SimpleNamespace

from bench import fingerprint, servers


NPU_SMI = """
| npu-smi 25.5.0                   Version: 25.5.0                                               |
| 1     910B1               | OK            | 97.6        49                0    / 0             |
| 0                         | 0000:01:00.0  | 7           0    / 0          3459 / 65536         |
"""

NPU_SMI_CARD2 = """
| npu-smi 25.5.0                   Version: 25.5.0                                               |
| 2     910B1               | OK            | 88.1        47                0    / 0             |
| 0                         | 0000:03:00.0  | 9           0    / 0          4567 / 65536         |
"""

NPU_SMI_PROCESSES = """
| 2     0    | 111      | python                 | 2048       |
| 1     0    | 222      | other                  | 1024       |
| 2     0    | 333      | worker                 | 4096       |
"""


def test_npu_identity_uses_pci_bus_as_stable_card_id(monkeypatch):
    monkeypatch.setattr(fingerprint, "_run", lambda _cmd: NPU_SMI)

    got = fingerprint._npu_identity(1)

    assert got == {
        "gpu_index": 1,
        "physical_device_index": 1,
        "gpu_sku": "Ascend 910B1",
        "gpu_uuid": "ascend-pci-0000:01:00.0",
        "driver": "25.5.0",
        "memory_mib": "65536",
        "accelerator": "Ascend",
        "pci_bus_id": "0000:01:00.0",
        "health": "OK",
        "power_draw_w": 97.6,
        "temperature_c": 49.0,
        "aicore_percent": 7.0,
        "memory_used_mib": 3459.0,
    }


def test_npu_identity_maps_scheduler_visible_logical_index(monkeypatch):
    monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", "2")
    monkeypatch.setattr(fingerprint, "_run", lambda _cmd: NPU_SMI_CARD2)

    got = fingerprint.gpu_identity(0)

    assert got["gpu_index"] == 0
    assert got["physical_device_index"] == 2
    assert got["gpu_uuid"] == "ascend-pci-0000:03:00.0"
    assert fingerprint.ascend_physical_device_index(0) == 2
    assert fingerprint.ascend_physical_device_index(2) == 2
    assert fingerprint.ascend_physical_device_index(1) is None


def test_npu_process_and_memory_queries_use_physical_visible_card(monkeypatch):
    monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", "2")
    monkeypatch.setattr(
        servers.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(stdout=NPU_SMI_PROCESSES),
    )

    assert servers.gpu_memory_used_mib(0) == 6144.0
    assert servers.gpu_processes(0) == [
        {"pid": 111, "type": "C", "process_name": "python", "used_memory": "2048"},
        {"pid": 333, "type": "C", "process_name": "worker", "used_memory": "4096"},
    ]


def test_env_fingerprint_falls_back_when_uv_cannot_run(monkeypatch):
    fingerprint.env_fingerprint.cache_clear()
    calls = []

    def fake_run(cmd, cwd=None):
        calls.append((cmd, cwd))
        return "" if cmd[0] == "uv" else "a==1\nb==2\n"

    monkeypatch.setattr(fingerprint, "_run", fake_run)
    got = fingerprint.env_fingerprint("/python")

    assert [call[0][0] for call in calls] == ["uv", "/python"]
    assert got["packages"] == 2
    assert got["freeze_sha256"] == hashlib.sha256(b"a==1\nb==2\n").hexdigest()
