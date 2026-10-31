import json
import subprocess
import sys
import tarfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "tools" / "run_frozen_performance_validator.py"
FROZEN = ("validator/compare.py", "bench/stats.py", "bench/verify.py")
FROZEN_ARCHIVE = ROOT / "outputs" / "minicpmo-performance-validator-r40.tar.gz"


def _extract_frozen(tmp_path: Path) -> Path:
    with tarfile.open(FROZEN_ARCHIVE, "r:gz") as bundle:
        members = bundle.getmembers()
        names = {member.name for member in members if member.isfile()}
        assert names == {
            *FROZEN,
            "tools/run_frozen_performance_validator.py",
            "self-check.json",
        }
        for member in members:
            path = Path(member.name)
            assert not path.is_absolute()
            assert ".." not in path.parts
        bundle.extractall(tmp_path)
    return tmp_path


def _run(freeze_root: Path):
    return subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            "--freeze-root",
            str(freeze_root),
            "--source-root",
            str(ROOT),
            "--self-check",
        ],
        capture_output=True,
        text=True,
    )


def test_frozen_loader_keeps_the_measured_checkout_as_source_root(tmp_path):
    freeze_root = _extract_frozen(tmp_path)
    proc = _run(freeze_root)
    assert proc.returncode == 0, proc.stderr
    result = json.loads(proc.stdout)
    assert Path(result["source_root"]) == ROOT
    assert Path(result["loaded"]["compare"]) == freeze_root / "validator" / "compare.py"
    assert Path(result["loaded"]["stats"]) == freeze_root / "bench" / "stats.py"
    assert Path(result["loaded"]["verify"]) == freeze_root / "bench" / "verify.py"


def test_frozen_loader_rejects_a_changed_validator_file(tmp_path):
    _extract_frozen(tmp_path)
    with (tmp_path / "bench" / "stats.py").open("a", encoding="utf-8") as stream:
        stream.write("\n# changed\n")

    proc = _run(tmp_path)
    assert proc.returncode != 0
    assert "missing or changed" in proc.stderr
