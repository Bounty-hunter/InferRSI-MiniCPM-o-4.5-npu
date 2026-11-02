"""Pinned latest-main baseline, native five-tier workload, no model patches."""
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request

ROOT = Path('/workspace/minicpmo-r182')
REPO = ROOT/'vllm-omni-main-20261102-0304'
CLIENT = ROOT/'vllm-omni-v0320-20261102'
OUT = ROOT/'artifacts/baseline-main-20261102-0304'
OLD = ROOT/'artifacts/baseline-r185'
SHA = '9990f10d9e898c3d38b1fa640513d55f771b8685'
PORT = 18334
MODEL = Path('/dataset/models/MiniCPM-o-4_5')
TIERS = ((32,1),(64,4),(128,8),(256,16),(512,32))


def save(name, value):
    (OUT/name).write_text(json.dumps(value,indent=2)+'\n')


def git(*args):
    return subprocess.check_output(['git',*args],cwd=REPO,text=True).strip()


def phase(name, **fields):
    value=dict(phase=name,time_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**fields)
    save('status.json',value)
    print(json.dumps(value),flush=True)


def main():
    assert os.getuid()==1033 and os.getgid()==1000
    assert os.environ.get('ASCEND_RT_VISIBLE_DEVICES')
    assert git('rev-parse','HEAD') == SHA
    assert git('rev-parse','refs/remotes/codex-main-20261102-0304') == SHA
    assert git('status','--porcelain') == ''
    OUT.mkdir(exist_ok=False)
    phase('preflight',ref='refs/heads/main',commit=SHA,device=os.environ['ASCEND_RT_VISIBLE_DEVICES'])
    frozen=json.loads((OLD/'input-sha256.json').read_text())
    unchanged={p:h for p,h in frozen.items() if '/vllm-omni/' not in p}
    for path,digest in unchanged.items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest,path
    deploy=REPO/'vllm_omni/deploy/minicpmo_4_5.yaml'
    ci=REPO/'tests/dfx/perf/tests/test_minicpmo_4_5.json'
    spec=json.loads(ci.read_text())[0]['benchmark_params'][0]
    assert spec['num_prompts']==[32,64,128,256,512]
    assert spec['max_concurrency']==[1,4,8,16,32]
    save('input-sha256.json',{**unchanged,str(deploy):hashlib.sha256(deploy.read_bytes()).hexdigest(),str(ci):hashlib.sha256(ci.read_bytes()).hexdigest()})
    (OUT/'deploy.yaml').write_bytes(deploy.read_bytes())
    (OUT/'ci-case.json').write_bytes(ci.read_bytes())
    save('source.json',dict(branch='main',fetched_at_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),commit=SHA,
         dirty=False,changes_from_previous=git('diff','--stat','5917730a4b02aedb44d374173cdc1b10484ccd52',SHA),
         previous_commit='5917730a4b02aedb44d374173cdc1b10484ccd52',
         original_checkout_preserved=True,pythonpath=os.environ.get('PYTHONPATH')))
    client_sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=CLIENT,text=True).strip()
    assert client_sha=='5917730a4b02aedb44d374173cdc1b10484ccd52'
    assert subprocess.check_output(['git','status','--porcelain'],cwd=CLIENT,text=True).strip()==''
    client_env=os.environ.copy()
    client_env['PYTHONPATH']=':'.join([str(CLIENT)]+[p for p in client_env.get('PYTHONPATH','').split(':') if p and p not in (str(REPO),str(CLIENT))])
    save('benchmark-source.json',dict(commit=client_sha,repo=str(CLIENT),pythonpath=client_env['PYTHONPATH'],scope='Fixed v0.32.0 client for a controlled latest-main server comparison'))
    env=os.environ.copy()
    env['OMNI_V032_RESULT_DIR']=str(OUT)
    env['OMNI_V032_REPO']=str(REPO)
    code='''
import importlib.metadata as m,json,os,sys
from pathlib import Path
import torch,torch_npu,vllm,vllm_omni
repo=Path(os.environ['OMNI_V032_REPO'])
assert Path(vllm_omni.__file__).resolve().is_relative_to(repo)
names=('vllm','vllm-ascend','vllm-omni','torch','torch-npu','transformers')
value=dict(python=sys.executable,visible_devices=os.environ['ASCEND_RT_VISIBLE_DEVICES'],
    device_name=torch.npu.get_device_name(0),omni_file=vllm_omni.__file__,vllm_file=vllm.__file__,
    source_version=getattr(vllm_omni,'__version__',None),versions={n:m.version(n) for n in names},
    note='Distribution metadata comes from the reused editable environment; the verified pinned main Git SHA identifies the executed source.')
assert value['versions']['vllm'].split('+')[0]=='0.32.0'
Path(os.environ['OMNI_V032_RESULT_DIR']).joinpath('runtime.json').write_text(json.dumps(value,indent=2)+'\\n')
print(json.dumps(value),flush=True)
'''
    with (OUT/'preflight.log').open('w') as log:
        result=subprocess.run([sys.executable,'-c',code],cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=180)
    save('preflight-exit-code.json',dict(exit_code=result.returncode))
    assert result.returncode==0,'See complete preflight.log'
    command=json.loads((OLD/'server-command.json').read_text())
    command[0]=sys.executable
    command[command.index('--port')+1]=str(PORT)
    command[command.index('--deploy-config')+1]=str(deploy)
    save('server-command.json',command)
    server=None;client=None;rows=[]
    signal.signal(signal.SIGTERM,lambda *_:sys.exit(143))
    try:
        with (OUT/'server.log').open('w') as serverlog:
            server=subprocess.Popen(command,cwd=REPO,env=env,stdout=serverlog,stderr=subprocess.STDOUT,start_new_session=True)
            save('server-pid.json',dict(pid=server.pid,pgid=server.pid))
            phase('server_starting',pid=server.pid)
            started=time.monotonic();deadline=started+1800
            while True:
                assert server.poll() is None,f'Server exited: {server.returncode}'
                try:
                    with urllib.request.urlopen(f'http://127.0.0.1:{PORT}/health',timeout=3) as response:
                        if response.status==200:break
                except (urllib.error.URLError,TimeoutError):pass
                assert time.monotonic()<deadline,'Health timeout after 1800 seconds'
                time.sleep(2)
            save('startup.json',dict(ready_s=time.monotonic()-started))
            with urllib.request.urlopen(f'http://127.0.0.1:{PORT}/v1/models',timeout=10) as response:save('models.json',json.load(response))
            for prompts,concurrency in TIERS:
                command=json.loads((OLD/'command-c8.json').read_text())
                command[0]=sys.executable
                filename=f'chunk25_c{concurrency}.json'
                for key,value in (('--port',str(PORT)),('--num-prompts',str(prompts)),('--max-concurrency',str(concurrency)),
                                  ('--result-dir',str(OUT)),('--result-filename',filename)):
                    command[command.index(key)+1]=value
                save(f'command-c{concurrency}.json',command)
                phase('benchmark_running',concurrency=concurrency,prompts=prompts,warmups=2)
                with (OUT/f'chunk25_c{concurrency}.log').open('w') as log:
                    client=subprocess.Popen(command,cwd=CLIENT,env=client_env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    deadline=time.monotonic()+1200
                    while client.poll() is None:
                        assert server.poll() is None,f'Server failed: {server.returncode}'
                        assert time.monotonic()<deadline,'Benchmark timeout after 1200 seconds'
                        time.sleep(1)
                save(f'exit-c{concurrency}.json',dict(exit_code=client.returncode))
                assert client.returncode==0,'See complete native benchmark log'
                raw=json.loads((OUT/filename).read_text())
                assert raw['completed']==prompts and raw['failed']==0,(raw['completed'],raw['failed'])
                for key in ('request_throughput','mean_audio_ttfp_ms','mean_audio_rtf','total_audio_duration_s'):
                    assert math.isfinite(raw[key]) and raw[key]>0,key
                rows.append(dict(concurrency=concurrency,prompts=prompts,result=raw))
                save('summary.json',rows)
                phase('benchmark_completed',concurrency=concurrency,completed=raw['completed'],failed=raw['failed'],
                    ttfp_ms=raw['mean_audio_ttfp_ms'],rtf=raw['mean_audio_rtf'],request_per_s=raw['request_throughput'])
                client=None
    except BaseException as error:
        phase('failed',error=repr(error),completed_cells=len(rows))
        raise
    finally:
        if client is not None and client.poll() is None:
            os.killpg(client.pid,signal.SIGTERM)
            try:client.wait(timeout=15)
            except subprocess.TimeoutExpired:os.killpg(client.pid,signal.SIGKILL);client.wait(timeout=10)
        if server is not None:
            cleanup=server.poll() is None;forced=False
            if cleanup:
                os.killpg(server.pid,signal.SIGTERM)
                try:server.wait(timeout=45)
                except subprocess.TimeoutExpired:
                    forced=True;os.killpg(server.pid,signal.SIGKILL);server.wait(timeout=10)
            save('server-exit-code.json',dict(returncode=server.returncode,cleanup_requested=cleanup,forced=forced))
    assert len(rows)==5
    assert git('rev-parse','HEAD')==SHA and git('status','--porcelain')==''
    phase('completed',cells=5,completed=sum(r['result']['completed'] for r in rows),failed=0,
          quality_gate='NOT_RUN',frozen_lcb='NOT_RUN')


if __name__=='__main__':main()
