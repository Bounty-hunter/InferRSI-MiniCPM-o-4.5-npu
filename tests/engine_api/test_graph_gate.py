import threading
import time

from engine.runtime.graph_gate import GraphLaunchGate


def test_capture_upgrades_own_reader_and_waits_for_other_reader():
    gate = GraphLaunchGate()
    other_entered = threading.Event()
    release_other = threading.Event()
    capture_entered = threading.Event()

    def other_reader():
        with gate.launch():
            other_entered.set()
            assert release_other.wait(2)

    thread = threading.Thread(target=other_reader)
    thread.start()
    assert other_entered.wait(2)
    with gate.launch():
        def release():
            time.sleep(0.05)
            assert not capture_entered.is_set()
            release_other.set()

        releaser = threading.Thread(target=release)
        releaser.start()
        with gate.capture():
            capture_entered.set()
        releaser.join(2)
    thread.join(2)
    assert capture_entered.is_set() and not thread.is_alive()


def test_new_reader_waits_during_capture_and_nested_capture_is_reentrant():
    gate = GraphLaunchGate()
    reader_entered = threading.Event()

    def reader():
        with gate.launch():
            reader_entered.set()

    with gate.launch(), gate.capture(), gate.capture():
        thread = threading.Thread(target=reader)
        thread.start()
        time.sleep(0.05)
        assert not reader_entered.is_set()
    thread.join(2)
    assert reader_entered.is_set() and not thread.is_alive()
