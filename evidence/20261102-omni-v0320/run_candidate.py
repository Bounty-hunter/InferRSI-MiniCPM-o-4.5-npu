"""Fresh-process native workload with explicit source/config identity and cleanup."""
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
CAMPAIGN=ROOT/'rsi-r286-r295'


def main():
    spec=json.loads(Path(sys.argv[1]).read_text())
    out=ROOT/'artifacts'/spec['artifact']
    engine=ROOT/spec['engine']
    config=ROOT/spec['config']
    port=spec['port']
    assert os.getuid()==1033 and os.getgid()==1000
    assert os.environ.get('ASCEND_RT_VISIBLE_DEVICES')
    out.mkdir(exist_ok=False)
    save=lambda name,value:(out/name).write_text(json.dumps(value,indent=2)+'\n')
    def phase(name,**fields):
        row=dict(phase=name,utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**fields)
        save('status.json',row);print(json.dumps(row),flush=True)
    def get(route):
        with urllib.request.urlopen(f'http://127.0.0.1:{port}/{route}',timeout=60) as response:
            return json.load(response)
    benchmark_repo=ROOT/'vllm-omni-v0320-20261102'
    sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=benchmark_repo,text=True).strip()
    assert sha==spec['benchmark_git_commit']
    assert subprocess.check_output(['git','status','--porcelain'],cwd=benchmark_repo,text=True).strip()==''
    save('benchmark-source.json',dict(git_tag=spec['benchmark_git_tag'],commit=sha,repo=str(benchmark_repo),pythonpath=os.environ.get('PYTHONPATH')))
    save('round-spec.json',spec)
    save('config.json',json.loads(config.read_text()))
    frozen=json.loads((ROOT/'artifacts/baseline-r185/input-sha256.json').read_text())
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==digest for p,digest in frozen.items())
    save('input-sha256.json',frozen)
    env=os.environ.copy()
    env.update(MINICPMO_ENGINE_ROOT=str(engine),MINICPMO_RESULT_DIR=str(out),
               MODEL_PATH='/dataset/models/MiniCPM-o-4_5',ENGINE_PYTHON=sys.executable)
    env['PYTHONPATH']=':'.join([str(CAMPAIGN),str(engine)]+[p for p in env.get('PYTHONPATH','').split(':')
                           if p and p!=str(ROOT/'vllm-omni')])
    if spec.get('tests'):
        with (out/'unit-tests.log').open('w') as log:
            rc=subprocess.run([sys.executable,'-m','pytest',*spec['tests'],'-q'],cwd=engine,env=env,
                              stdout=log,stderr=subprocess.STDOUT,timeout=120).returncode
        save('unit-tests-exit.json',dict(exit_code=rc));assert rc==0
    command=[sys.executable,str(CAMPAIGN/'entry.py'),'--host','127.0.0.1','--port',str(port),'--config',str(config)]
    save('server-command.json',command)
    server=None;client=None;results=[];boundaries=[]
    signal.signal(signal.SIGTERM,lambda *_:sys.exit(143))
    try:
        started=time.monotonic()
        with (out/'server.log').open('w') as log:
            server=subprocess.Popen(command,cwd=engine,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            phase('loading',pid=server.pid,device=os.environ['ASCEND_RT_VISIBLE_DEVICES'])
            deadline=time.monotonic()+600
            while True:
                assert server.poll() is None,server.returncode
                try:
                    with urllib.request.urlopen(f'http://127.0.0.1:{port}/health',timeout=2) as response:
                        if response.status==200:break
                except (urllib.error.URLError,TimeoutError):pass
                assert time.monotonic()<deadline
                time.sleep(1)
            startup=time.monotonic()-started
            info=get('info');save('info.json',info)
            assert info['engine_tree_sha256']==spec['engine_sha256']
            save('startup.json',dict(server_ready_s=startup))
            for concurrency in spec.get('concurrencies',[16]):
                source=ROOT/('artifacts/baseline-r185' if concurrency in (1,4,8) else 'artifacts/scaling-r240/omni')
                original=json.loads((source/f'command-c{concurrency}.json').read_text())
                for window in range(1,spec.get('repeat',1)+1):
                    tag=f'c{concurrency}-w{window}'
                    command=list(original)
                    for key,value in (('--port',str(port)),('--result-dir',str(out)),('--result-filename',tag+'.json')):
                        command[command.index(key)+1]=value
                    save(tag+'-command.json',command)
                    begin_ns=time.time_ns()
                    phase('benchmark',tag=tag,start_ns=begin_ns)
                    with (out/(tag+'.log')).open('w') as clientlog:
                        client=subprocess.Popen(command,cwd=ROOT/'vllm-omni-v0320-20261102',env=os.environ.copy(),stdout=clientlog,stderr=subprocess.STDOUT)
                        rc=client.wait(timeout=900)
                    save(tag+'-exit.json',dict(exit_code=rc));assert rc==0
                    data=json.loads((out/(tag+'.json')).read_text())
                    assert data['completed']==int(original[original.index('--num-prompts')+1])
                    assert data['failed']==0
                    boundaries.append(dict(tag=tag,concurrency=concurrency,window=window,
                                           begin_ns=begin_ns,end_ns=time.time_ns(),native_warmups=2))
                    save('window-boundaries.json',boundaries)
                    metrics=get('metrics');save(tag+'-metrics.json',metrics)
                    assert metrics['failed_streams']==0 and metrics['cancelled_streams']==0
                    results.append(dict(concurrency=concurrency,window=window,ttfp_ms=data['mean_audio_ttfp_ms'],
                        rtf=data['mean_audio_rtf'],req_s=data['request_throughput'],failed=data['failed'],
                        completed=data['completed'],input_tokens=data['total_input_tokens'],
                        audio_duration_s=data['total_audio_duration_s'],server_ready_s=startup))
                    save('summary.json',results);phase('benchmark_completed',**results[-1])
            phase('completed',measurements=results)
    except BaseException as error:
        phase('failed',error=repr(error));raise
    finally:
        if client is not None and client.poll() is None:
            client.terminate();client.wait(timeout=20)
        if server is not None:
            if server.poll() is None:
                os.killpg(server.pid,signal.SIGTERM)
                try:server.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(server.pid,signal.SIGKILL);server.wait(timeout=10)
            save('server-exit-code.json',dict(exit_code=server.returncode,cleanup_requested=True))


if __name__=='__main__':
    main()
