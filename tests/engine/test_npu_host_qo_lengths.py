"""CPU checks of the published NPU host-length fast path and device fallback."""
import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

SOURCE = Path(__file__).resolve().parents[2] / 'engine/runtime/npu_paged.py'
tree = ast.parse(SOURCE.read_text(encoding='utf-8'))
runner = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'NpuDecoderRunner')
method = next(node for node in runner.body if isinstance(node, ast.FunctionDef) and node.name == '_query_ends')
namespace = {}
exec(compile(ast.Module(body=[method], type_ignores=[]), str(SOURCE), 'exec'), namespace)
query_ends = namespace['_query_ends']


class Pointer:
    def __init__(self, values, calls):
        self.values = values
        self.calls = calls

    def __getitem__(self, index):
        return Pointer(self.values[index], self.calls)

    def to(self, device):
        assert device == 'cpu'
        self.calls.append('device-observation')
        return self

    def tolist(self):
        return self.values


@pytest.mark.parametrize('counts', ([1], [1, 1, 1], [2, 9, 3], [128, 1, 17, 3], [1] * 8))
@pytest.mark.parametrize('host_present', (True, False))
def test_valid_lengths_and_missing_metadata(counts, host_present):
    cumulative = [0]
    for count in counts:
        cumulative.append(cumulative[-1] + count)
    calls = []
    layout = SimpleNamespace(
        seqs=list(range(len(counts))), new_tokens=counts,
        host_qo_indptr=tuple(cumulative) if host_present else None,
        qo_indptr=Pointer(cumulative, calls))
    assert query_ends(None, layout, cumulative[-1]) == cumulative[1:]
    assert len(calls) == int(not host_present)


@pytest.mark.parametrize('host,counts,seqs,query_rows', (
    ((1, 2, 3), [1, 1], [0, 1], 3),
    ((0, 1, 4), [1, 2], [0, 1], 4),
    ((0, 1, 3), [1, 2], [0], 3),
    ((0, 1, 3), [1, 2], [0, 1], 4),
    ((0, 1, 3), [True, 2], [0, 1], 3),
    ((0, 0, 3), [0, 3], [0, 1], 3),
))
def test_malformed_metadata_uses_original_device_observation(host, counts, seqs, query_rows):
    calls = []
    layout = SimpleNamespace(seqs=seqs, new_tokens=counts, host_qo_indptr=host,
                             qo_indptr=Pointer([0, 2, 5], calls))
    assert query_ends(None, layout, query_rows) == [2, 5]
    assert calls == ['device-observation']
