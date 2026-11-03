"""Read the verified four-tier direct comparison and retain supplement provenance."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parents[1] / 'evidence/20261103-r337/campaign'
RESULTS = HERE / 'results'
CONCURRENCIES = (1, 2, 4, 8)
REQUEST_COUNTS = {1: 32, 2: 32, 4: 64, 8: 128}


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def load_direct_comparison():
    assert read(HERE / 'omni-confirmation-verification.json')['verified']
    assert read(HERE / 'c1-completion-verification.json')['verified']
    paths = [RESULTS / 'omni-c1/summary.json', RESULTS / 'omni-final/summary.json']
    omni = {}
    for path in paths:
        omni.update({row['concurrency']: row['result'] for row in read(path)})
    engine = {1: {}, 2: {}}
    for repeat in (1, 2):
        for concurrency in CONCURRENCIES:
            tag = f'r337-c1B{repeat}' if concurrency == 1 else f'r337-finalB{repeat}'
            assert (RESULTS / f'minicpmo-{tag}-20261102/exit-code.txt').read_text().strip() == '0'
            path = RESULTS / tag / 'native' / f'c{concurrency}.json'
            paths.append(path)
            result = read(path)
            baseline = omni[concurrency]
            assert result['completed'] == baseline['completed'] == REQUEST_COUNTS[concurrency]
            assert result['failed'] == baseline['failed'] == 0
            assert result['total_input_tokens'] == baseline['total_input_tokens']
            engine[repeat][concurrency] = result
    return omni, engine, paths
