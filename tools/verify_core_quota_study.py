"""Recompute compact fixed-work evidence without loading NPU dependencies."""
import hashlib
import json
import math
from pathlib import Path
import re
import statistics

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / 'evidence/20261103-core-quota'
COMPONENTS = ('thinker', 'talker', 'audio')
TAGS = ('A1', 'B1', 'B2', 'A2', 'A3', 'B3')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close(actual, expected):
    assert math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-10), (actual, expected)


def check_samples(row, count):
    samples = row['samples_ms']
    assert row['n'] == len(samples) == count
    assert all(math.isfinite(value) and value > 0 for value in samples)
    for key, value in (('mean_ms', statistics.mean(samples)),
                       ('median_ms', statistics.median(samples)),
                       ('min_ms', min(samples)), ('max_ms', max(samples))):
        close(row[key], value)


def verify_source():
    manifest = read(EVIDENCE / 'source-manifest.json')
    tree = hashlib.sha256()
    for name, expected in sorted(manifest['runtime_files'].items()):
        path = ROOT / manifest['overrides'].get(name, name)
        assert sha256(path) == expected, name
        tree.update(name.encode() + b'\0' + path.read_bytes() + b'\0')
    assert tree.hexdigest() == manifest['engine_tree_sha256']
    configs = {role: read(EVIDENCE / f'config-{role}.json') for role in 'AB'}
    assert configs['A'].pop('npu_core_fraction') == 1.0
    assert configs['B'].pop('npu_core_fraction') == 0.5
    assert configs['A'] == configs['B']
    for role in 'AB':
        assert sha256(EVIDENCE / f'config-{role}.json') == manifest['config_hashes'][role]
    assert manifest['speech_gate'] == 'NOT_RUN' and not manifest['lossless_claim']
    assert (manifest['uid'], manifest['gid']) == (1033, 1000)
    return manifest


def verify_runs(manifest):
    fixtures = read(EVIDENCE / 'capture/manifest.json')['cases']
    reference = read(EVIDENCE / 'fixture-reference.json')
    assert sha256(EVIDENCE / 'capture/manifest.json') == reference['manifest_sha256']
    capture = read(EVIDENCE / 'capture/requests-results.json')['rows']
    assert len(capture) == 16 and all(row['ok'] for row in capture)
    plan = read(EVIDENCE / 'component-plan.json')
    assert [row[0] for row in plan['tasks']] == [f'replay-{tag}' for tag in TAGS] + ['profile-A', 'profile-B']
    for name in ('fixed_common.py', 'capture_fixed.py', 'replay_fixed.py', 'component_controller.py'):
        assert sha256(EVIDENCE / 'scripts' / name) == plan['programs'][name]
    rows = {}
    digests = {role: {} for role in 'AB'}
    distributions = set()
    processes = set()
    for tag in (*TAGS, 'profile-A', 'profile-B'):
        profile = tag.startswith('profile-')
        role = tag[-1] if profile else tag[0]
        folder = EVIDENCE / ('profiles' if profile else 'trials') / (role if profile else tag)
        assert read(folder / 'exit.json')['exit_code'] == 0
        info = read(folder / 'info.json')
        assert info['engine_tree_sha256'] == manifest['engine_tree_sha256']
        assert info['config_sha256'] == manifest['config_hashes'][role]
        assert info['device_name'] == 'Ascend910B1' and info['t2w_worker_pid'] is None
        assert info['visible_devices'] == plan['visible_devices']
        processes.add(info['pid'])
        distributions.add(info['python_dists_sha256'])
        audit = read(folder / 'runtime-import-audit.json')
        assert not audit['forbidden_import_attempts'] and not audit['forbidden_modules_loaded']
        assert read(folder / 'input-files.json') == {name: row['sha256'] for name, row in fixtures.items()}
        quota = dict(cube_core_num=24, vector_core_num=48) if role == 'A' else dict(cube_core_num=12, vector_core_num=24)
        assert read(folder / 'device-limit.json') == quota
        for owner in ('engine', 'audio'):
            observed = read(folder / f'resource-{owner}.json')
            assert observed['device_limit'] == observed['stream_limit'] == quota
        row = read(folder / 'results.json')
        assert row['role'] == role and row['quota'] == quota
        assert row['input_templates_unchanged'] and row['diagnostic_only']
        for name in COMPONENTS:
            measured = row['isolated'][name]
            assert measured['same_input_repeated_exact']
            for window in ('host_completion', 'stream_elapsed'):
                check_samples(measured[window], 3 if profile else 20)
            assert measured['versus_capture']['exact'] == (role == 'A')
            if role == 'A':
                assert measured['digest'] == fixtures[name]['expected_digest']
            previous = digests[role].setdefault(name, measured['digest'])
            assert previous == measured['digest']
        for modes in row['matrix'].values():
            for measured in modes.values():
                check_samples(measured, 2 if profile else 12)
        if profile:
            assert row['profile']['record_shapes'] and not row['profile']['with_stack']
            assert row['profile']['aic_metrics'] == 'PipeUtilization'
        else:
            rows[tag] = row
    assert len(processes) == 8 and len(distributions) == 1
    assert all(digests['A'][name] != digests['B'][name] for name in COMPONENTS)
    assert (EVIDENCE / 'receipt-exit-code.txt').read_text().strip() == '0'
    return rows


def verify_aggregates(rows):
    summary = read(EVIDENCE / 'summary.json')
    paired = read(EVIDENCE / 'fixed-work-analysis.json')
    assert summary['ordering'] == list(TAGS)
    assert paired['complete'] and paired['server_pairs'] == 3
    for tag in TAGS:
        assert sha256(EVIDENCE / 'trials' / tag / 'results.json') == paired['raw_result_sha256'][tag]
        assert rows[tag]['shapes'] == summary['shapes']
    for name in COMPONENTS:
        a, b = [statistics.mean(rows[f'{role}{i}']['isolated'][name]['host_completion']['median_ms']
                                for i in (1, 2, 3)) for role in 'AB']
        expected = summary['components'][name]
        close(expected['control_ms'], a)
        close(expected['limited_ms'], b)
        close(expected['reduction_pct'], (1 - b / a) * 100)
        assert summary['numerical_differences'][name] == rows['B1']['isolated'][name]['versus_capture']
    for loops, matrix in summary['matrix'].items():
        for role in 'AB':
            for mode in ('serial', 'parallel'):
                close(matrix['table_ms'][role][mode], statistics.mean(
                    rows[f'{role}{i}']['matrix'][loops][mode]['median_ms'] for i in (1, 2, 3)))
            close(matrix['stream_time_saved_ms'][role],
                  matrix['table_ms'][role]['serial'] - matrix['table_ms'][role]['parallel'])
        for mode in ('serial', 'parallel'):
            close(matrix['quota_reduction_pct'][mode],
                  (1 - matrix['table_ms']['B'][mode] / matrix['table_ms']['A'][mode]) * 100)
    for pair in paired['pairs']:
        a, b = [rows[f'{role}{pair["pair"]}'] for role in 'AB']
        for name in COMPONENTS:
            for window in ('host_completion', 'stream_elapsed'):
                values = pair['isolated'][name][window]
                close(values['control_ms'], a['isolated'][name][window]['median_ms'])
                close(values['limited_ms'], b['isolated'][name][window]['median_ms'])
                close(values['reduction_pct'], (1 - values['limited_ms'] / values['control_ms']) * 100)
        for loops, matrix in pair['matrix'].items():
            for role, raw in (('A', a), ('B', b)):
                for mode in ('serial', 'parallel'):
                    close(matrix['table_ms'][role][mode], raw['matrix'][loops][mode]['median_ms'])
                close(matrix['parallel_reduction_pct'][role],
                      (1 - matrix['table_ms'][role]['parallel'] / matrix['table_ms'][role]['serial']) * 100)
            for mode in ('serial', 'parallel'):
                close(matrix['quota_reduction_pct'][mode],
                      (1 - matrix['table_ms']['B'][mode] / matrix['table_ms']['A'][mode]) * 100)
    for name in COMPONENTS:
        values = [pair['isolated'][name]['host_completion']['reduction_pct'] for pair in paired['pairs']]
        aggregate = paired['aggregate_isolated'][name]
        close(aggregate['mean_paired_reduction_pct'], statistics.mean(values))
        close(aggregate['min_pct'], min(values))
        close(aggregate['max_pct'], max(values))


def verify_profile_and_bundle():
    profile = read(EVIDENCE / 'profile-kernel-analysis.json')
    assert profile['paired_groups'] == profile['same_count_groups'] == 652
    assert profile['unmatched'] == dict(A=[], B=[])
    for role in 'AB':
        raw = profile['provenance'][role]
        assert raw['rows'] == 266538 and raw['groups'] == 652
        assert len(raw['sha256']) == 64 and raw['bytes'] > 0
    for key in ('largest_work_reductions', 'largest_work_regressions', 'largest_control_work'):
        for row in profile[key]:
            assert row['same_count'] and row['A']['count'] == row['B']['count']
            close(row['mean_reduction_pct'], (1 - row['B']['mean_us'] / row['A']['mean_us']) * 100)
    bundle_path = EVIDENCE / 'bundle-sha256.json'
    bundle = read(bundle_path)
    actual = {path.relative_to(EVIDENCE).as_posix(): sha256(path)
              for path in EVIDENCE.rglob('*') if path.is_file() and path != bundle_path}
    assert bundle == actual
    for path in (ROOT / 'docs/core-quota-causality.zh-CN.md', EVIDENCE / 'README.md'):
        for target in re.findall(r'\]\(([^)]+)\)', path.read_text(encoding='utf-8')):
            assert (path.parent / target.split('#')[0]).exists(), target


if __name__ == '__main__':
    verify_aggregates(verify_runs(verify_source()))
    verify_profile_and_bundle()
    print('PASS: source/config binding, 6 fixed-work trials, 2 profile receipts, samples, summaries and bundle hashes')
    print('Remote large inputs/CSV hashes are recorded identities; final speech gate remains NOT_RUN.')
