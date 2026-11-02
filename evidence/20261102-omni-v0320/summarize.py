"""Verify completed tag baseline and same-client engine evidence before publishing."""
import hashlib
import json
import math
from pathlib import Path

HERE=Path(__file__).resolve().parent
SHA='5917730a4b02aedb44d374173cdc1b10484ccd52'
KEYS=('ttfp_ms','rtf','req_s')


def read(path):return json.loads(path.read_text(encoding='utf-8'))


def normalize(command):
    command=list(command)
    for key in ('--port','--result-dir','--result-filename'):
        command[command.index(key)+1]='<environment-output-path>'
    return command


def main():
    baseline=HERE/'baseline-v0320-20261102'
    candidate=HERE/'candidate-v0320-client-20261102'
    assert int((HERE/'minicpmo-omni-v0320-baseline-20261102/exit-code').read_text())==0
    assert int((HERE/'minicpmo-engine-v0320-client-20261102/exit-code').read_text())==0
    source=read(baseline/'source.json');assert source['commit']==SHA and not source['dirty']
    assert read(baseline/'status.json')['phase']=='completed'
    assert read(candidate/'status.json')['phase']=='completed'
    assert read(baseline/'server-exit-code.json')['returncode']==0
    assert not read(baseline/'server-exit-code.json')['forced']
    assert read(candidate/'server-exit-code.json')['exit_code']==0
    assert read(candidate/'runtime-import-audit.json')=={'forbidden_import_attempts':[],'forbidden_modules_loaded':[]}
    info=read(candidate/'info.json')
    assert info['engine_tree_sha256']=='7d0c96eeadccf88c98872ea48f9149e60085c483b5c046c0a2063906db17f780'
    assert info['config_sha256']=='3fe1dd286a25caaf65053c860ce307589395fc1f1aeb10337e409e4fdac3a0d0'
    assert read(candidate/'benchmark-source.json')['commit']==SHA
    config=read(candidate/'config.json')
    assert config['float16_flow'] is True and config['t2w_timesteps']==6
    runtime=read(baseline/'runtime.json')
    assert Path(runtime['omni_file']).is_relative_to('/workspace/minicpmo-r182/vllm-omni-v0320-20261102')
    inputs=read(baseline/'input-sha256.json');other=read(candidate/'input-sha256.json')
    for path,digest in inputs.items():
        if path.startswith('/dataset/'):
            assert other[path]==digest
    results=[]
    for n,c in ((32,1),(64,4),(128,8),(256,16),(512,32)):
        a=read(baseline/f'chunk25_c{c}.json');b=read(candidate/f'c{c}-w1.json')
        assert a['completed']==b['completed']==n
        assert a['failed']==b['failed']==0
        assert a['total_input_tokens']==b['total_input_tokens']
        assert read(baseline/f'exit-c{c}.json')['exit_code']==read(candidate/f'c{c}-w1-exit.json')['exit_code']==0
        server_metrics=read(candidate/f'c{c}-w1-metrics.json')
        assert server_metrics['failed_streams']==server_metrics['cancelled_streams']==0
        assert normalize(read(baseline/f'command-c{c}.json'))==normalize(read(candidate/f'c{c}-w1-command.json'))
        metrics=lambda r:dict(ttfp_ms=r['mean_audio_ttfp_ms'],rtf=r['mean_audio_rtf'],req_s=r['request_throughput'],
            completed=r['completed'],failed=r['failed'],input_tokens=r['total_input_tokens'],audio_duration_s=r['total_audio_duration_s'])
        base,engine=metrics(a),metrics(b)
        assert all(math.isfinite(v[k]) and v[k]>0 for v in (base,engine) for k in KEYS)
        gains={k:(engine[k]/base[k]-1)*100 if k=='req_s' else (1-engine[k]/base[k])*100 for k in KEYS}
        results.append(dict(concurrency=c,omni=base,candidate=engine,benefit_percent={'omni':gains}))
    data=dict(schema=2,measurement_date='2026-11-02',hardware='One Ascend 910B1 (A2) per service',
        candidate='Unchanged r292 benchmark candidate, measured with v0.32.0 CLI',
        baseline_key='omni',baseline_label='vLLM-Omni v0.32.0',
        baseline_identity=dict(git_tag='v0.32.0',git_tag_object=source['tag_object'],git_commit=SHA,
            vllm_runtime_version=runtime['versions']['vllm'],reported_source_version=runtime.get('source_version'),
            installed_distribution_versions=runtime['versions'],
            note=runtime['note'],source_clean=True),
        engine_identity=dict(engine_tree_sha256=info['engine_tree_sha256'],config_sha256=info['config_sha256'],promoted_to_default=False),
        workload='Seed-TTS en; chunk25; fixed C1/C4/C8/C16/C32 order; 32/64/128/256/512 measured requests; native two warmups per tier',
        official_client_git_commit=SHA,total_measured_requests=1984,measured_errors=0,
        internal_compute=dict(omni=dict(flow_precision='FP32 (unmodified tag defaults)',flow_steps=10),
            candidate=dict(flow_precision='FP16',flow_steps=6),
            scope='Same benchmark inputs and client, different internal precision and solver budget; not a lossless same-precision speedup.'),
        uncertainty='Single-run points; frozen statistical and speech-quality acceptance not completed',measurements=results,
        evidence_sha256={f'{variant}/{name}':hashlib.sha256((directory/name).read_bytes()).hexdigest()
            for variant,directory in (('baseline',baseline),('candidate',candidate))
            for name in ('summary.json','server-command.json','input-sha256.json','server-exit-code.json')})
    (HERE/'comparison.json').write_text(json.dumps(data,indent=2)+'\n',encoding='utf-8')
    lines=['# Official vLLM-Omni v0.32.0 — Ascend baseline remeasurement','',
        f'Git tag v0.32.0 → `{SHA}`; same native tag CLI on both services. No model/runtime patches or shared environment upgrades.', '',
        '| C | Omni TTFP / RTF / req/s | Engine TTFP / RTF / req/s | Engine improvement TTFP / RTF / req/s |',
        '|---|---|---|---|']
    for r in results:
        fmt=lambda v:f"{v['ttfp_ms']:.2f} / {v['rtf']:.5f} / {v['req_s']:.5f}"
        gain=' / '.join(f"{r['benefit_percent']['omni'][k]:+.2f}%" for k in KEYS)
        lines.append(f"| C{r['concurrency']} | {fmt(r['omni'])} | {fmt(r['candidate'])} | {gain} |")
    lines+=['','1,984 measured requests, zero errors. Point estimates only; no frozen LCB or final speech gate.', '',
        'Internal compute differs: stock Omni Flow FP32/10 steps; existing engine Flow FP16/6 steps. This is a system comparison, not proof of a lossless scheduling speedup. The previous 16-row quality screen compared with the older engine, not this official tag.', '',
        'The reported package metadata is retained verbatim from the reused editable environment; execution identity is verified by source path and the clean tag worktree.']
    (HERE/'results.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('\n'.join(lines))


if __name__=='__main__':main()
