"""Recount published activities and explicit/manual candidate groups for the retrospective."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'docs/data/optimization-retrospective.json'
LEDGER = ROOT / 'docs/optimization-ledger-npu.zh-CN.md'
CAMPAIGN = ROOT / 'evidence/20261103-r337/campaign/campaign-results.json'

# The early ledger records jobs, repairs and tests, without one uniform RoundSpec.
# These are disclosed human groupings, not recovered automatic algorithm rounds.
# Each tuple is (representative, mechanism, evidence IDs, retained in r337, reason).
EARLY = [
    ('r191', 'identity-gather', '191 192 193', False, 'micro_gain_service_regression'),
    ('r196', 'thinker-static-fia', '196', False, 'numerical_check_failed'),
    ('r198', 'talker-full-block', '198 199 200 201 202 204 206 207 208 210 212 214 215', True, 'corrected_block_retained'),
    ('r216', 'talker-add-rms', '216 217', True, 'retained_with_mixed_ttfp'),
    ('r218', 'flow-fixed-adaln', '218 220', False, 'local_gain_without_service_confirmation'),
    ('r221', 'causal-conv-gemm', '221', False, 'common_shapes_slower'),
    ('r222', 'thinker-full-fusion', '222 224 225 226', False, 'speech_screen_failed'),
    ('r227', 'cfm-full-fusion', '227 228 229', False, 'dtype_and_cache_numerical_failure'),
    ('r230', 'cfm-packed-only', '230 231 232 233 234 235 236 237', False, 'native_reliability_failure'),
    ('r238', 'cfm-hybrid', '238 239', True, 'qualified_hybrid_retained'),
]

PROFILE = [
    ('r243', 'talker-all-exact-rows', '243 244 244b 245', False, 'c4_regression'),
    ('r246', 'talker-selective-rows', '246 247 253', True, 'selective_route_retained'),
    ('r248', 'thinker-eight-plus-eight', '248 249', False, 'native_regression'),
    ('r250', 'warmed-eight-plus-eight', '250 251', False, 'native_regression'),
    ('r254', 'prefix-always-on', '254 254b', False, 'low_concurrency_regression'),
    ('r255', 'prefix-active-eight', '255', False, 'no_general_gain'),
    ('r256', 'prefix-active-twelve', '256', False, 'no_general_gain'),
    ('r257', 'prefix-live-twelve', '257', False, 'no_general_gain'),
    ('r258', 'prefix-live-nine', '258', True, 'conditional_pressure_route_retained'),
    ('r259', 'prefix-latch', '259', False, 'worse_than_parent'),
    ('r260', 'prefix-preseed', '260 260b', False, 'no_service_gain'),
    ('r261', 'prefix-preseed-latch-wait10', '261', False, 'no_general_gain'),
    ('r262', 'prefix-cohort-admission', '262', False, 'waiting_offsets_gain'),
    ('r263', 'prefix-wait50', '263 263b', False, 'longer_wait_worse'),
    ('r264', 'thinker-static-block', '264 264b', False, 'accumulated_numerical_drift'),
    ('r265', 'thinker-dynamic-block16', '265 266', False, 'zero_native_replay'),
    ('r267', 'thinker-dynamic-block9-16', '267 268', False, 'zero_native_replay'),
    ('r269', 'thinker-dynamic-block8', '269 270', False, 'zero_native_replay'),
    ('r271', 'thinker-dynamic-block3-8', '271 272', False, 'one_native_replay_and_slower'),
    ('r273', 'thinker-dynamic-block1-8', '273 274', False, 'zero_native_replay'),
    ('r275', 'thinker-block-routing-repair', '275 276', False, 'no_archived_service_result'),
    ('r280', 'mixed-tick-split', '280 280b', False, 'concurrency_tradeoff'),
    ('r281', 'thinker-graph9-10', '281 281b 281c', False, 'small_gain_unqualified'),
    ('r282', 'host-metadata-and-mask', '282', False, 'no_consistent_service_gain'),
    ('r283', 'busy-cycle-split-gating', '283 284', False, 'c16_regression'),
    ('r287', 'fair-capture-gate', '287', True, 'fairness_retained_not_uniform_gain'),
    ('r288', 'hift-startup', '288', False, 'not_selected_in_matched_candidate'),
    ('r289', 'euler-startup', '289', False, 'request_failure'),
    ('r290', 'gate-plus-hift', '290', False, 'combination_not_selected'),
    ('r291', 'pa-update-exclusive', '291', False, 'native_regression'),
    ('r292', 'gate-plus-first-audio', '292 294 295', True, 'mechanism_retained_matched_c8_c32_regressed'),
    ('r293', 'dense-page-id-cache', '293', False, 'not_selected_in_matched_candidate'),
]

SIX_LAYER = [
    ('r297', 'thinker-graph16', False, 'small_gain_not_qualified'),
    ('r298', 'thinker-graph32', False, 'mean_tail_tradeoff'),
    ('r299', 'grouped-pcm-delivery', False, 'high_concurrency_only_candidate'),
    ('r300', 'thinker-graph8-slices', False, 'native_regression'),
    ('r301', 'thinker-startup-buckets', False, 'native_regression'),
    ('r302', 'replay-event-wait', False, 'no_stable_incremental_gain'),
    ('r303', 'audio-edf', False, 'tail_and_continuity_tradeoff'),
    ('r304', 'adaln-outside-replay', False, 'native_regression'),
    ('r305', 'prompt-lookup', True, 'retained_with_control_confirmation'),
    ('r306', 'npu-pending-audio', False, 'api_repaired_mixed_native_result'),
]


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ledger_rows():
    rows = {}
    for line in LEDGER.read_text(encoding='utf-8').splitlines():
        cells = [cell.strip() for cell in line.split('|')[1:-1]]
        if cells and re.fullmatch(r'r\d+[a-z]?(?:/r\d+[a-z]?)*', cells[0]):
            for tag in cells[0].split('/'):
                if 182 <= int(re.search(r'\d+', tag)[0]) <= 295:
                    assert tag not in rows
                    rows[tag] = dict(cells=cells, row_sha256=hashlib.sha256(line.encode()).hexdigest())
    return rows


def manual_stage(name, definitions, rows):
    groups = []
    for representative, mechanism, evidence, retained, reason in definitions:
        tags = [f'r{tag}' for tag in evidence.split()]
        assert all(tag in rows for tag in tags)
        groups.append(dict(representative=representative, mechanism=mechanism, evidence_ids=tags,
                           retained_current_experimental=retained, reason=reason))
    return dict(stage=name, counting='retrospective human grouping, not original automatic round count', groups=groups)


def build():
    rows = ledger_rows()
    stages = [manual_stage('knowledge-led-r182-r239', EARLY, rows),
              manual_stage('profiling-and-autonomy-r240-r295', PROFILE, rows)]
    six_groups = []
    for tag, mechanism, retained, reason in SIX_LAYER:
        folder = 'r306b' if tag == 'r306' else tag
        spec = ROOT / 'evidence/20261102-r305/rounds' / folder / 'spec.json'
        assert spec.exists()
        six_groups.append(dict(representative=tag, mechanism=mechanism, retained_current_experimental=retained,
                               reason=reason, spec=str(spec.relative_to(ROOT)).replace('\\', '/')))
    stages.append(dict(stage='six-layer-r297-r306', counting='10 recorded algorithm rounds; r306b is a repair', groups=six_groups))
    followup = json.loads((ROOT / 'evidence/20261103-r337/followup/followup-results.json').read_text(encoding='utf-8'))
    assert {row['tag'] for row in followup['experiments']} == {
        'r307', 'r308', 'r309', 'r310-r309-confirm', 'r311-control-after'}
    stages.append(dict(stage='owner-followup-r307-r309', counting='3 recorded candidates; r310/r311 are confirmations/controls', groups=[
        dict(representative='r307', mechanism='owner-startup', retained_current_experimental=False,
             reason='narrow_candidate_superseded_by_r309', mechanism_reused=True),
        dict(representative='r308', mechanism='remove-head-fp32-cast', retained_current_experimental=False,
             reason='no_consistent_native_gain'),
        dict(representative='r309', mechanism='owner-startup-extended', retained_current_experimental=True,
             reason='retained_extended_candidate'),
    ]))
    campaign = json.loads(CAMPAIGN.read_text(encoding='utf-8'))
    attempts = campaign['completed']
    assert {row['tag'] for row in attempts} == {f'r{i}' for i in range(312, 352)}
    late_groups = []
    for row in attempts:
        late_groups.append(dict(representative=row['tag'], mechanism=row['kind'], parent=row['parent'],
                                retained_current_experimental=row['tag'] in ('r320', 'r337'),
                                reason=row['verdict'], receipt_exit=row['receipt_exit'],
                                mechanism_reused=row['tag'] == 'r319'))
    assert next(row for row in attempts if row['tag'] == 'r337')['parent'] == 'r320'
    assert next(row for row in attempts if row['tag'] == 'r320')['parent'] == 'control309'
    stages.append(dict(stage='continued-r312-r351', counting='40 recorded algorithm rounds; controls/retries excluded', groups=late_groups))
    summaries = []
    for stage in stages:
        count = len(stage['groups'])
        retained = sum(row['retained_current_experimental'] for row in stage['groups'])
        summaries.append(dict(stage=stage['stage'], groups=count, retained=retained, not_retained=count-retained,
                              not_retained_pct=(count-retained)/count*100))
    sources = [LEDGER, CAMPAIGN,
               ROOT / 'docs/rsi-profiling-20261102.zh-CN.md',
               ROOT / 'evidence/20261103-r337/followup/followup-results.json',
               ROOT / 'evidence/20261103-r337/campaign/final-confirmation-summary.json']
    coverage = []
    for lo, hi in ((182, 239), (240, 295)):
        selected = {tag: row for tag, row in rows.items() if lo <= int(re.search(r'\d+', tag)[0]) <= hi}
        primaries = {int(re.search(r'\d+', tag)[0]) for tag in selected}
        coverage.append(dict(range=[lo, hi], primary_slots=hi-lo+1, recorded_primary_ids=len(primaries),
                             expanded_evidence_ids=len(selected), missing_primary_ids=sorted(set(range(lo, hi+1))-primaries),
                             records=selected))
    assert '## r296' in LEDGER.read_text(encoding='utf-8')
    primary_ids = {int(re.search(r'\d+', tag)[0]) for tag in rows} | set(range(296, 352))
    missing_primary = sorted(set(range(182, 352)) - primary_ids)
    assert missing_primary == [203, 205, 209, 211, 213]
    return dict(schema='optimization-retrospective/v1', as_of='2026-11-03',
                scope='official-benchmark era r182-r351; earlier r32-r181 incomplete and excluded',
                unit_warning='Early manual proposal groups and later recorded rounds are exposed separately. No exact lifetime algorithm-run total is recoverable.',
                adoption_warning='Retained means a mechanism/configuration node inherited by experimental r337. It does not certify isolated gain or the speech gate.',
                failure_warning='Not retained includes negative, inconclusive, superseded and unfinished work. It is not a count of proven zero-value experiments.',
                primary_activities=dict(slots=170, recorded=len(primary_ids), missing=missing_primary),
                ledger_coverage=coverage, stages=stages, summary=summaries,
                reconstructed_group_total=sum(row['groups'] for row in summaries),
                retained_nodes=sum(row['retained'] for row in summaries),
                not_retained_groups=sum(row['not_retained'] for row in summaries),
                exact_later_algorithm_rounds=53,
                late_screening_counts=dict(Counter(row['verdict'] for row in attempts)),
                late_execution_counts=dict(successful_exit=sum(row['receipt_exit']==0 for row in attempts),
                                           nonzero_exit=sum(row['receipt_exit']!=0 for row in attempts)),
                sources=[dict(path=str(path.relative_to(ROOT)).replace('\\', '/'), sha256=file_hash(path)) for path in sources],
                infer_rsi_reference=dict(repository='https://github.com/InferMatrixLab/InferRSI',
                                         commit='a1ee3266466194c16f4e2159ab845b758ed9561b',
                                         scope='Current framework contract only; not proof of historical per-round execution'))


def main():
    data = build()
    assert [(s['groups'], s['retained'], s['not_retained']) for s in data['summary']] == [
        (10,3,7), (32,4,28), (10,1,9), (3,1,2), (40,2,38)]
    assert data['reconstructed_group_total'] == 95
    assert data['retained_nodes'] == 11 and data['not_retained_groups'] == 84
    assert [row['expanded_evidence_ids'] for row in data['ledger_coverage']] == [53,66]
    encoded = json.dumps(data, ensure_ascii=False, indent=2) + '\n'
    if '--verify-only' in sys.argv:
        assert OUTPUT.read_text(encoding='utf-8') == encoded
    else:
        OUTPUT.write_text(encoded, encoding='utf-8', newline='\n')
    print(json.dumps(dict(summary=data['summary'], reconstructed_groups=95, retained_nodes=11,
                          not_retained=84, exact_later_algorithm_rounds=53,
                          late_screening=data['late_screening_counts'], late_execution=data['late_execution_counts'])))


if __name__ == '__main__':
    main()
