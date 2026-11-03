"""Audit the fixed-work component and 2x2 experiments without NPU dependencies."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics

parser = argparse.ArgumentParser()
parser.add_argument('root', type=Path)
parser.add_argument('--require-complete', action='store_true')
args = parser.parse_args()
root = args.root
results = {}
input_hashes = None
for tag in ('A1', 'B1', 'B2', 'A2', 'A3', 'B3'):
    folder = root / f'replay-{tag}'
    path = folder / 'results.json'
    exit_path = root / f'replay-{tag}-controller-exit.json'
    if not path.exists() or not exit_path.exists():
        continue
    assert json.loads(exit_path.read_text())['exit_code'] == 0
    current = json.loads(path.read_text())
    assert current['input_templates_unchanged']
    assert all(row['same_input_repeated_exact'] for row in current['isolated'].values())
    hashes = json.loads((folder / 'input-files.json').read_text())
    if input_hashes is None:
        input_hashes = hashes
    assert hashes == input_hashes, 'Variants used different fixed inputs'
    assert current['quota']['cube_core_num'] == (24 if tag[0] == 'A' else 12)
    assert current['quota']['vector_core_num'] == (48 if tag[0] == 'A' else 24)
    results[tag] = current
if args.require_complete:
    assert len(results) == 6, list(results)
pairs = []
for index in (1, 2, 3):
    if f'A{index}' not in results or f'B{index}' not in results:
        continue
    a, b = results[f'A{index}'], results[f'B{index}']
    isolated = {}
    for component in ('thinker', 'talker', 'audio'):
        isolated[component] = {}
        for window in ('host_completion', 'stream_elapsed'):
            control = a['isolated'][component][window]['median_ms']
            limited = b['isolated'][component][window]['median_ms']
            isolated[component][window] = dict(control_ms=control, limited_ms=limited,
                                              reduction_pct=(1-limited/control)*100)
    matrix = {}
    for loops in ('1', '4'):
        table = {role: {mode: value['matrix'][loops][mode]['median_ms'] for mode in ('serial','parallel')}
                 for role, value in (('A',a), ('B',b))}
        matrix[loops] = dict(table_ms=table,
            quota_reduction_pct={mode:(1-table['B'][mode]/table['A'][mode])*100 for mode in ('serial','parallel')},
            parallel_reduction_pct={role:(1-table[role]['parallel']/table[role]['serial'])*100 for role in ('A','B')})
    pairs.append(dict(pair=index, isolated=isolated, matrix=matrix))
aggregate = {}
if pairs:
    for component in ('thinker', 'talker', 'audio'):
        values = [row['isolated'][component]['host_completion']['reduction_pct'] for row in pairs]
        aggregate[component] = dict(mean_paired_reduction_pct=statistics.mean(values),
                                    min_pct=min(values), max_pct=max(values))
report = dict(scope='Fixed-work component and2x2 diagnostics; native service timing excluded',
              complete=len(results)==6, server_pairs=len(pairs), fixture_sha256=input_hashes,
              aggregate_isolated=aggregate, pairs=pairs,
              raw_result_sha256={tag:hashlib.sha256((root/f'replay-{tag}/results.json').read_bytes()).hexdigest() for tag in results})
(root / 'fixed-work-analysis.json').write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(report, indent=2))
