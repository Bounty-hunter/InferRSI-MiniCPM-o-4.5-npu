"""Device-independent contracts for the Ascend DiT graph selector."""

from engine.runtime.dit_graph import EulerGraphs


def test_euler_graph_long_cache_falls_back_to_eager():
    graphs = object.__new__(EulerGraphs)
    graphs.graph_t = (50,)
    graphs.mem_rows = {512: 32, 1024: 16}
    graphs.failures = {}

    assert graphs.supports(rows=2, T=50, L=1024)
    assert not graphs.supports(rows=2, T=50, L=1025)
    assert not graphs.supports(rows=2, T=50, L=1796)
