"""Verify published source, four-tier measurements and the 40-round audit bundle."""
import ast
import hashlib
import json
import math
from pathlib import Path
import re

from r337_comparison_data import CONCURRENCIES, REQUEST_COUNTS, load_direct_comparison

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / 'evidence/20261103-r337'
ARCHIVE = EVIDENCE / 'campaign'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalized(command):
    command = list(command)
    for key in ('--port', '--result-dir', '--result-filename'):
        command[command.index(key) + 1] = '<environment>'
    return command


def verify_source():
    manifest = read(EVIDENCE / 'source-manifest.json')
    actual = {path.relative_to(ROOT).as_posix(): path for path in (ROOT / 'engine').rglob('*')
              if path.is_file() and path.suffix in ('.py', '.sh', '.yaml', '.json') and '__pycache__' not in path.parts}
    assert set(actual) == set(manifest['runtime_files'])
    measured = hashlib.sha256()
    control = hashlib.sha256()
    for name, path in sorted(actual.items()):
        assert sha256(path) == manifest['runtime_files'][name], name
        content = path.read_bytes()
        if name == 'engine/server.py':
            exact = ROOT / manifest['exact_measured_server']
            assert sha256(exact) == manifest['measured_server_sha256']
            content = exact.read_bytes()
        measured.update(name.encode() + b'\0' + content + b'\0')
        override = manifest['control309_overrides'].get(name)
        control_content = (ROOT / override).read_bytes() if override else content
        control.update(name.encode() + b'\0' + control_content + b'\0')
        if path.suffix == '.py':
            for node in ast.walk(ast.parse(path.read_bytes())):
                modules = [alias.name for alias in node.names] if isinstance(node, ast.Import) else (
                    [node.module] if isinstance(node, ast.ImportFrom) and node.module else [])
                assert all(module.split('.')[0].lower() not in
                           ('vllm', 'vllm_omni', 'sglang', 'sglang_omni', 'inferrsi') for module in modules), name
    assert measured.hexdigest() == manifest['measured_engine_tree_sha256']
    assert control.hexdigest() == manifest['control309_tree_sha256']
    assert sha256(ROOT / manifest['experimental_config']) == manifest['config_sha256']
    return manifest


def verify_measurements(manifest):
    omni, engine, paths = load_direct_comparison()
    data = read(ROOT / 'docs/data/ascend-r337-20261103.json')
    assert read(ROOT / 'docs/current-results.json') == data
    assert data == read(ARCHIVE / '独立引擎vs-vLLM-Omni-绘图数据.json')
    assert data['concurrencies'] == list(CONCURRENCIES)
    assert data['engine_curves_shown'] == data['baseline_curves_shown'] == 1
    assert data['engine_repeats_shown'] == 0
    assert data['speech_gate'] == 'NOT_RUN' and not data['equal_quality_claim']
    for record in data['sources']:
        assert sha256(ARCHIVE / record['path']) == record['sha256']
    for row in data['rows']:
        c = row['concurrency']
        baseline_folder = ARCHIVE / 'results' / ('omni-c1' if c == 1 else 'omni-final')
        baseline_command = read(baseline_folder / f'command-c{c}.json')
        baseline = omni[c]
        assert read(baseline_folder / 'source.json')['commit'] == data['omni_server_sha']
        assert not read(baseline_folder / 'source.json')['dirty']
        assert read(baseline_folder / 'runtime.json')['device_name'] == 'Ascend910B1'
        assert read(baseline_folder / 'server-exit-code.json') == dict(returncode=0, cleanup_requested=True, forced=False)
        assert read(baseline_folder / 'benchmark-source.json')['commit'] == '5917730a4b02aedb44d374173cdc1b10484ccd52'
        assert read(baseline_folder / f'exit-c{c}.json')['exit_code'] == 0
        assert baseline['completed'] == REQUEST_COUNTS[c] and baseline['failed'] == 0
        for key, value in row['omni'].items():
            assert value == baseline[key]
        for repeat in (1, 2):
            tag = f'r337-c1B{repeat}' if c == 1 else f'r337-finalB{repeat}'
            folder = ARCHIVE / 'results' / tag
            result = engine[repeat][c]
            assert result['completed'] == REQUEST_COUNTS[c] and result['failed'] == 0
            assert result['total_input_tokens'] == baseline['total_input_tokens']
            assert normalized(read(folder / 'native' / f'c{c}-command.json')) == normalized(baseline_command)
            assert read(folder / 'native-server-exit.json') == dict(exit_code=0, forced=False)
            info = read(folder / 'native/info.json')
            assert info['engine_tree_sha256'] == manifest['measured_engine_tree_sha256']
            assert info['config_sha256'] == manifest['config_sha256']
            assert info['device_name'] == 'Ascend910B1'
            audit = read(folder / 'native/runtime-import-audit.json')
            assert not audit['forbidden_import_attempts'] and not audit['forbidden_modules_loaded']
            for key, value in row['engine_repeats'][repeat - 1].items():
                assert value == result[key]
        for key, value in row['engine_result'].items():
            expected = (engine[1][c][key] + engine[2][c][key]) / 2
            assert math.isclose(value, expected, abs_tol=1e-12)
    for filename in data['files']:
        assert (ARCHIVE / filename).stat().st_size > 1000
    return len(paths)


def verify_rounds():
    campaign = read(ARCHIVE / 'campaign-results.json')
    rows = campaign['completed']
    patches = {row['tag']: row for row in read(ARCHIVE / 'patches/manifest.json')['rows']}
    assert len(rows) == len(patches) == 40
    assert {row['tag'] for row in rows} == {f'r{i}' for i in range(312, 352)}
    failures = []
    for row in rows:
        tag = row['tag']
        receipt = ARCHIVE / 'results' / f'minicpmo-{tag}-20261102'
        assert int((receipt / 'exit-code.txt').read_text()) == row['receipt_exit']
        assert (receipt / 'receipt.log').stat().st_size > 0
        spec = read(ARCHIVE / 'results' / tag / 'spec.json')
        assert spec['engine_sha256'] == row['source_sha256'] == patches[tag]['candidate_sha256']
        assert sha256(ARCHIVE / patches[tag]['patch']) == patches[tag]['patch_sha256']
        assert read(ARCHIVE / spec['config']) == read(ARCHIVE / 'results' / tag / 'config.json')
        if row['receipt_exit']:
            failures.append(tag)
            assert list((ARCHIVE / 'results' / tag).rglob('*.log.txt'))
    assert failures == ['r322', 'r327', 'r329', 'r330', 'r331', 'r335', 'r339']
    return len(rows)


def verify_links():
    documents = [ROOT / 'README.md', ROOT / 'docs/current-results.md', ROOT / 'docs/rsi-r337-20261103.zh-CN.md',
                 ROOT / 'docs/optimization-journey.zh-CN.md', ROOT / 'docs/optimization-ledger-npu.zh-CN.md',
                 ROOT / 'docs/optimizations-npu.md', ROOT / 'docs/architecture-npu.md',
                 ROOT / 'docs/reproduce-benchmark.md', ROOT / 'docs/testing-npu.md', *EVIDENCE.rglob('*.md')]
    count = 0
    for document in documents:
        for link in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)', document.read_text(encoding='utf-8')):
            if '://' in link or link.startswith('#'):
                continue
            assert (document.parent / link.split('#')[0]).exists(), (document, link)
            count += 1
    return count


def main():
    manifest = verify_source()
    source_files = verify_measurements(manifest)
    rounds = verify_rounds()
    links = verify_links()
    for name, expected in read(EVIDENCE / 'bundle-sha256.json')['files'].items():
        assert sha256(EVIDENCE / name) == expected, name
    print(json.dumps(dict(direct_concurrencies=list(CONCURRENCIES), measured_source_files=source_files,
                          algorithm_attempts=rounds, local_links=links, aggregated_curve_verified=True,
                          measured_source_reconstructed=True, control_source_reconstructed=True,
                          speech_gate='NOT_RUN', quality_certified=False)))


if __name__ == '__main__':
    main()
