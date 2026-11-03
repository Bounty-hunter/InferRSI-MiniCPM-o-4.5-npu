"""One isolated RSI round: native timing, separate host profiling, quality generation."""
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request

ROOT=Path('/workspace/minicpmo-r182')
HERE=Path(__file__).resolve().parent
CLIENT=ROOT/'vllm-omni-v0320-20261102'

def main():
    spec=json.loads(Path(sys.argv[1]).read_text())
    out=ROOT/'artifacts'/HERE.name/spec['tag']
    out.mkdir(parents=True,exist_ok=False)
    engine=(HERE/spec['engine']).resolve();config=HERE/spec['config'];port=spec['port']
    assert os.getuid()==1033 and os.getgid()==1000 and os.environ.get('ASCEND_RT_VISIBLE_DEVICES')
    def save(name,value): (out/name).write_text(json.dumps(value,indent=2)+'\n')
    def phase(name,**fields):
        row=dict(phase=name,utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**fields)
        save('status.json',row);print(json.dumps(row),flush=True)
    def get(route):
        with urllib.request.urlopen(f'http://127.0.0.1:{port}/{route}',timeout=90) as r:return json.load(r)
    def cleanup(process,name):
        forced=False
        if process is not None:
            if process.poll() is None:
                os.killpg(process.pid,signal.SIGTERM)
                try:process.wait(timeout=40)
                except subprocess.TimeoutExpired:
                    forced=True;os.killpg(process.pid,signal.SIGKILL);process.wait(timeout=10)
            save(name+'-exit.json',dict(exit_code=process.returncode,forced=forced))
    save('spec.json',spec);save('config.json',json.loads(config.read_text()))
    frozen=json.loads((ROOT/'artifacts/baseline-r185/input-sha256.json').read_text())
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in frozen.items())
    save('input-sha256.json',frozen)
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=CLIENT,text=True).strip()=='5917730a4b02aedb44d374173cdc1b10484ccd52'
    env=os.environ.copy()
    env.update(MINICPMO_ENGINE_ROOT=str(engine),MINICPMO_RESULT_DIR=str(out),MODEL_PATH='/dataset/models/MiniCPM-o-4_5',ENGINE_PYTHON=sys.executable)
    env['PYTHONPATH']=':'.join([str(engine),str(HERE),env.get('PYTHONPATH','')])
    client_env=os.environ.copy();client_env['PYTHONPATH']=':'.join([str(CLIENT),client_env.get('PYTHONPATH','')])
    summary=[];server=None;client=None
    signal.signal(signal.SIGTERM,lambda *_:sys.exit(143))
    try:
        # Separate fresh service processes prevent profiler perturbation of accepted timers.
        modes=[]
        if spec.get('native'):modes.append(('native',spec['native']))
        if spec.get('diagnostic'):modes.append(('diagnostic',spec['diagnostic']))
        if spec.get('quality'):modes.append(('quality',[]))
        if spec.get('quality_first'):
            modes.sort(key=lambda pair: pair[0] != 'quality')
        for mode,concurrencies in modes:
            group=out/mode;group.mkdir()
            env.update(MINICPMO_RESULT_DIR=str(group),RSI_DIAGNOSTIC=str(int(mode=='diagnostic')),RSI_QUALITY=str(int(mode=='quality')))
            if mode=='quality':
                (group/'wavs').mkdir();env['MINICPMO_QUALITY_DIR']=str(group/'wavs')
            command=[sys.executable,str(HERE/'entry_lowc.py'),'--host','127.0.0.1','--port',str(port),'--config',str(config)]
            save(mode+'-server-command.json',command)
            with (group/'server.log').open('w') as log:
                server=subprocess.Popen(command,cwd=engine,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                save(mode+'-server-pid.json',dict(pid=server.pid));phase('loading',mode=mode,pid=server.pid,device=os.environ['ASCEND_RT_VISIBLE_DEVICES'])
                deadline=time.monotonic()+600
                while True:
                    assert server.poll() is None,server.returncode
                    try:
                        with urllib.request.urlopen(f'http://127.0.0.1:{port}/health',timeout=2) as r:
                            if r.status==200:break
                    except (urllib.error.URLError,TimeoutError):pass
                    assert time.monotonic()<deadline,'health timeout'
                    time.sleep(1)
                info=get('info');(group/'info.json').write_text(json.dumps(info,indent=2)+'\n')
                assert info['engine_tree_sha256']==spec['engine_sha256'],info
                assert info['config_sha256']==hashlib.sha256(config.read_bytes()).hexdigest()
                if mode=='quality':
                    request=urllib.request.Request(f'http://127.0.0.1:{port}/diagnostics/quality',data=b'{"variant":"candidate"}',headers={'Content-Type':'application/json'})
                    phase('quality_generation')
                    with urllib.request.urlopen(request,timeout=600) as response:quality=json.load(response)
                    assert len(quality['rows'])==16 and all(row['ok'] for row in quality['rows'])
                    save('quality-generation.json',quality)
                    if spec.get('require_exact_thinker'):
                        reference=json.loads((ROOT/'artifacts'/HERE.name/(spec.get('quality_reference','control-lowc')+'/quality-generation.json')).read_text())
                        assert [r['request_id'] for r in reference['rows']]==[r['request_id'] for r in quality['rows']]
                        parity=[dict(request_id=a['request_id'],text_exact=a['thinker_text']==b['thinker_text'],
                                     tokens_exact=a['thinker_tokens']==b['thinker_tokens']) for a,b in zip(reference['rows'],quality['rows'])]
                        save('prebenchmark-thinker-parity.json',parity)
                        assert all(r['text_exact'] and r['tokens_exact'] for r in parity),'Thinker verification parity failed'

                for c in concurrencies:
                    tag=f'c{c}'
                    command=json.loads((HERE/'commands'/f'command-c{c}.json').read_text())
                    command[0]=sys.executable
                    command.extend(['--metric-percentiles','50,95,99'])
                    if mode=='diagnostic':
                        command=[command[0],str(ROOT/'rtf-profile-r296-20261102/client_entry.py'),*command[3:]]
                        client_env['RSI_CLIENT_TRACE']=str(group/(tag+'-chunks.json'))
                    for opt,val in (('--port',str(port)),('--result-dir',str(group)),('--result-filename',tag+'.json')):
                        command[command.index(opt)+1]=val
                    (group/(tag+'-command.json')).write_text(json.dumps(command,indent=2)+'\n')
                    phase('benchmark',mode=mode,concurrency=c)
                    with (group/(tag+'.log')).open('w') as log:
                        client=subprocess.Popen(command,cwd=CLIENT,env=client_env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                        rc=client.wait(timeout=900)
                    (group/(tag+'-exit.json')).write_text(json.dumps(dict(exit_code=rc))+'\n');assert rc==0
                    raw=json.loads((group/(tag+'.json')).read_text())
                    assert raw['failed']==0 and raw['completed']==int(command[command.index('--num-prompts')+1])
                    metrics=get('metrics');(group/(tag+'-metrics.json')).write_text(json.dumps(metrics,indent=2)+'\n')
                    assert metrics['failed_streams']==0 and metrics['cancelled_streams']==0
                    summary.append(dict(mode=mode,concurrency=c,ttfp_ms=raw['mean_audio_ttfp_ms'],rtf=raw['mean_audio_rtf'],req_s=raw['request_throughput'],completed=raw['completed'],failed=raw['failed']))
                    save('summary.json',summary);phase('benchmark_completed',**summary[-1])
                cleanup(server,mode+'-server');assert server.returncode==0
                audit=json.loads((group/'runtime-import-audit.json').read_text())
                assert not audit['forbidden_import_attempts'] and not audit['forbidden_modules_loaded']
                server=None
        phase('completed',measurements=summary,quality='generation only, not scored gate')
    except BaseException as error:
        phase('failed',error=repr(error));raise
    finally:
        cleanup(client,'client-final');cleanup(server,'server-final')

if __name__=='__main__':main()
