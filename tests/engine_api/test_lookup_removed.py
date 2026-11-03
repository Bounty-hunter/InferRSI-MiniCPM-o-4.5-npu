"""CPU-only regression checks for removal and preservation of ordinary decoding."""
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'engine/runtime/backend_npu.py'
TREE = ast.parse(SOURCE.read_text(encoding='utf-8'))
BACKEND = next(node for node in TREE.body if isinstance(node, ast.ClassDef) and node.name == 'AscendTrackSBackend')
METHODS = {node.name: node for node in BACKEND.body if isinstance(node, ast.FunctionDef)}
ARCHIVE = ROOT / 'evidence/20261103-lookup-removal'


def load_method(name, namespace):
    future = ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)
    module = ast.fix_missing_locations(ast.Module(body=[future, METHODS[name]], type_ignores=[]))
    exec(compile(module, str(SOURCE), 'exec'), namespace)
    return namespace[name]


class DeviceInitializationReached(Exception):
    pass


class LookupRemovalTests(unittest.TestCase):
    def test_legacy_nonzero_setting_fails_before_device_initialization(self):
        device = Mock(side_effect=DeviceInitializationReached)
        initialize = load_method('__init__', dict(MODEL_DIR=None, _T2W_BATCH_WAIT_MIN_ACTIVE=8,
                                                  torch=SimpleNamespace(device=device)))
        for value in (8, 1, -1, 0.5, None, '8'):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'prompt-lookup has been removed'):
                initialize(SimpleNamespace(), thinker_prompt_lookup_max_tokens=value)
        device.assert_not_called()

    def test_default_and_legacy_zero_continue_to_normal_initialization(self):
        device = Mock(side_effect=DeviceInitializationReached)
        initialize = load_method('__init__', dict(MODEL_DIR=None, _T2W_BATCH_WAIT_MIN_ACTIVE=8,
                                                  torch=SimpleNamespace(device=device)))
        for kwargs in ({}, {'thinker_prompt_lookup_max_tokens': 0}):
            with self.subTest(kwargs=kwargs), self.assertRaises(DeviceInitializationReached):
                initialize(SimpleNamespace(), **kwargs)
        self.assertEqual(device.call_count, 2)

    def test_all_serving_configs_omit_lookup(self):
        for path in (ROOT / 'configs/engine').glob('*.json'):
            with self.subTest(path=path.name):
                self.assertNotIn('thinker_prompt_lookup_max_tokens', json.loads(path.read_text(encoding='utf-8')))

    def test_normal_decode_body_is_preserved(self):
        historical = ast.parse((ARCHIVE / 'source/engine/runtime/backend_npu.py.txt').read_text(encoding='utf-8'))
        backend = next(node for node in historical.body if isinstance(node, ast.ClassDef) and node.name == BACKEND.name)
        previous = next(node for node in backend.body if isinstance(node, ast.FunctionDef) and node.name == '_thinker_step_group')
        self.assertIn('try_prompt_lookup_step', ast.unparse(previous.body[1]))
        expected = ast.Module(body=[previous.body[0], *previous.body[2:]], type_ignores=[])
        actual = ast.Module(body=METHODS['_thinker_step_group'].body, type_ignores=[])
        self.assertEqual(ast.dump(actual), ast.dump(expected))
        self.assertEqual(METHODS['_thinker_step_group'].args.kwonlyargs, [])

    def test_empty_batch_uses_ordinary_return_without_lookup_state(self):
        step = load_method('_thinker_step_group', {})
        self.assertIsNone(step(SimpleNamespace(), []))

    def test_draft_module_branch_and_counters_are_removed(self):
        self.assertFalse((ROOT / 'engine/runtime/thinker_lookup.py').exists())
        source = SOURCE.read_text(encoding='utf-8')
        for text in ('try_prompt_lookup_step', 'thinker_prompt_lookup_stats', '_skip_prompt_lookup'):
            self.assertNotIn(text, source)

    def test_archived_source_and_config_bytes_are_bound(self):
        mapping = json.loads((ARCHIVE / 'source-map.json').read_text(encoding='utf-8'))
        self.assertEqual(mapping['current_performance'], 'NOT_MEASURED')
        for name, row in mapping['files'].items():
            with self.subTest(name=name):
                self.assertEqual(hashlib.sha256((ROOT / row['path']).read_bytes()).hexdigest(), row['sha256'])


if __name__ == '__main__':
    unittest.main()
