"""Check the completed main run against the previously verified tag/client evidence."""
import hashlib
import json
import math
from pathlib import Path

HERE=Path(__file__).resolve().parent
PREVIOUS=HERE.parent/'omni-v0320-baseline-20261102'
MAIN_SHA='9990f10d9e898c3d38b1fa640513d55f771b8685'
CLIENT_SHA='5917730a4b02aedb44d374173cdc1b10484ccd52'
KEYS=('ttfp_ms','rtf','req_s')

def read(path):
    return json.loads(path.read_text(encoding='utf-8'))

def normalize(command):
    command=list(command)
    for key in ('--port','--result-dir','--result-filename'):
        command[command.index(key)+1]='<endpoint-or-output>'
    return command

def benefit(candidate,reference):
    return {k:((candidate[k]/reference[k]-1) if k=='req_s' else (1-candidate[k]/reference[k]))*100 for k in KEYS}

def metrics(raw):
    return dict(ttfp_ms=raw['mean_audio_ttfp_ms'],rtf=raw['mean_audio_rtf'],req_s=raw['request_throughput'],
                completed=raw['completed'],failed=raw['failed'],input_tokens=raw['total_input_tokens'],
                audio_duration_s=raw['total_audio_duration_s'])

def main():
    baseline=HERE/'baseline-main-20261102-0304'
    tag=PREVIOUS/'baseline-v0320-20261102'
    candidate=PREVIOUS/'candidate-v0320-client-20261102'
    assert int((HERE/'minicpmo-omni-main-20261102-0304/exit-code').read_text())==0
    assert int((PREVIOUS/'minicpmo-engine-v0320-client-20261102/exit-code').read_text())==0
    source=read(baseline/'source.json')
    assert source['commit']==MAIN_SHA and not source['dirty']
    assert read(baseline/'status.json')['phase']=='completed'
    assert read(baseline/'benchmark-source.json')['commit']==CLIENT_SHA
    assert read(baseline/'server-exit-code.json')==dict(returncode=0,cleanup_requested=True,forced=False)
    assert read(candidate/'server-exit-code.json')['exit_code']==0
    assert read(candidate/'runtime-import-audit.json')==dict(forbidden_import_attempts=[],forbidden_modules_loaded=[])
    info=read(candidate/'info.json')
    assert info['engine_tree_sha256']=='7d0c96eeadccf88c98872ea48f9149e60085c483b5c046c0a2063906db17f780'
    assert info['config_sha256']=='3fe1dd286a25caaf65053c860ce307589395fc1f1aeb10337e409e4fdac3a0d0'
    runtime=read(baseline/'runtime.json')
    assert runtime['omni_file'].startswith('/workspace/minicpmo-r182/vllm-omni-main-20261102-0304/')
    assert runtime['device_name']=='Ascend910B1'
    inputs=read(baseline/'input-sha256.json')
    for directory in (tag,candidate):
        other=read(directory/'input-sha256.json')
        assert all(other[p]==digest for p,digest in inputs.items() if p.startswith('/dataset/'))
    rows=[]
    for n,c in ((32,1),(64,4),(128,8),(256,16),(512,32)):
        a=read(baseline/f'chunk25_c{c}.json')
        t=read(tag/f'chunk25_c{c}.json')
        b=read(candidate/f'c{c}-w1.json')
        assert a['completed']==t['completed']==b['completed']==n
        assert a['failed']==t['failed']==b['failed']==0
        assert a['total_input_tokens']==t['total_input_tokens']==b['total_input_tokens']
        assert read(baseline/f'exit-c{c}.json')['exit_code']==0
        command=normalize(read(baseline/f'command-c{c}.json'))
        assert command==normalize(read(tag/f'command-c{c}.json'))==normalize(read(candidate/f'c{c}-w1-command.json'))
        base,previous,engine=metrics(a),metrics(t),metrics(b)
        assert all(math.isfinite(v[k]) and v[k]>0 for v in (base,previous,engine) for k in KEYS)
        rows.append(dict(concurrency=c,omni=base,previous_tag=previous,candidate=engine,
                         benefit_percent={'omni':benefit(engine,base)},main_benefit_over_tag=benefit(base,previous)))
    data=dict(schema=2,measurement_date='2026-11-02',hardware='One Ascend 910B1 (A2) per service',
        baseline_key='omni',baseline_label='vLLM-Omni main (9990f10d9)',
        baseline_identity=dict(git_branch='main',git_commit=MAIN_SHA,source_clean=True,
            vllm_runtime_version=runtime['versions']['vllm'],reported_source_version=runtime['source_version'],
            installed_distribution_versions=runtime['versions'],note=runtime['note']),
        candidate='Unchanged r292 candidate; reused completed November 2 same-client measurement',
        engine_identity=dict(engine_tree_sha256=info['engine_tree_sha256'],config_sha256=info['config_sha256'],promoted_to_default=False),
        client_git_commit=CLIENT_SHA,
        workload='Seed-TTS en; chunk25; C1/C4/C8/C16/C32; 32/64/128/256/512 measured requests; two native warmups per tier',
        comparison_measured_requests=1984,newly_measured_requests=992,measured_errors=0,
        measurement_timing=dict(omni_main=read(baseline/'status.json')['time_utc'],
            engine=read(candidate/'status.json')['utc'],
            note='Latest-main server is fresh; the unchanged engine result is reused from the earlier same-day official-tag-client run.'),
        internal_compute=dict(omni=dict(flow_precision='FP32',flow_steps=10),candidate=dict(flow_precision='FP16',flow_steps=6),
            scope='Different internal precision/solver budgets; not lossless same-precision attribution.'),
        uncertainty='Single-run points; no frozen confidence-bound or final speech-quality acceptance',
        plot_provenance='Omni: pinned main@9990f10d9, vLLM 0.32.0. Fixed v0.32.0 native benchmark client on both implementations.',
        plot_scope='Main server: new Nov 2 run; unchanged r292 engine: earlier Nov 2 run. Each: 992 requests, 0 errors; 2 warmups/tier.',
        plot_reference='Pinned main@9990f10d9; same-day engine evidence reused.',
        measurements=rows,
        evidence_sha256={name:hashlib.sha256((baseline/name).read_bytes()).hexdigest()
            for name in ('summary.json','source.json','benchmark-source.json','input-sha256.json','server-command.json','server-exit-code.json')})
    (HERE/'comparison.json').write_text(json.dumps(data,indent=2)+'\n',encoding='utf-8',newline='\n')
    lines=['# Latest main comparison','',f'Omni main `{MAIN_SHA}`; fixed native client `{CLIENT_SHA}`.','',
        '| C | Main TTFP / RTF / req/s | Engine TTFP / RTF / req/s | Engine gain TTFP / RTF / req/s | Main gain over tag TTFP / RTF / req/s |',
        '|---|---|---|---|---|']
    for row in rows:
        fmt=lambda v:f"{v['ttfp_ms']:.2f} / {v['rtf']:.5f} / {v['req_s']:.5f}"
        gains=lambda v:' / '.join(f'{v[k]:+.2f}%' for k in KEYS)
        lines.append(f"| C{row['concurrency']} | {fmt(row['omni'])} | {fmt(row['candidate'])} | {gains(row['benefit_percent']['omni'])} | {gains(row['main_benefit_over_tag'])} |")
    lines+=['','Zero measured errors. Main is a new 992-request run; engine evidence is the completed same-day 992-request run.',
            'Flow: Omni FP32/10 steps versus engine FP16/6. No speech-quality equivalence or statistical acceptance claimed.']
    (HERE/'results.md').write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n')
    print('\n'.join(lines))

if __name__=='__main__':
    main()
