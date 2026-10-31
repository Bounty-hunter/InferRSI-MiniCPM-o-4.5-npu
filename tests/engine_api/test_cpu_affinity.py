from unittest import mock

import pytest

from engine.server import apply_cpu_affinity_from_env, parse_cpu_affinity


def test_parse_cpu_affinity_accepts_linux_cpu_lists():
    assert parse_cpu_affinity("0-3,8,10-11") == {0, 1, 2, 3, 8, 10, 11}


@pytest.mark.parametrize("spec", ["", "1,,2", "3-1", "-1", "x", "1-x"])
def test_parse_cpu_affinity_rejects_invalid_lists(spec):
    with pytest.raises(ValueError):
        parse_cpu_affinity(spec)


def test_apply_cpu_affinity_checks_the_cpuset_and_applies_exactly(monkeypatch):
    monkeypatch.setenv("ENGINE_CPU_AFFINITY", "4-7")
    get_affinity = mock.Mock(side_effect=[set(range(8)), {4, 5, 6, 7}])
    with mock.patch("engine.server.os.sched_getaffinity", get_affinity, create=True), \
            mock.patch("engine.server.os.sched_setaffinity", create=True) as set_affinity:
        assert apply_cpu_affinity_from_env() == [4, 5, 6, 7]
    set_affinity.assert_called_once_with(0, {4, 5, 6, 7})


def test_apply_cpu_affinity_rejects_cpus_outside_the_current_cpuset(monkeypatch):
    monkeypatch.setenv("ENGINE_CPU_AFFINITY", "7-8")
    with mock.patch("engine.server.os.sched_getaffinity", return_value=set(range(8)), create=True), \
            mock.patch("engine.server.os.sched_setaffinity", create=True):
        with pytest.raises(ValueError, match=r"unavailable CPUs: \[8\]"):
            apply_cpu_affinity_from_env()
