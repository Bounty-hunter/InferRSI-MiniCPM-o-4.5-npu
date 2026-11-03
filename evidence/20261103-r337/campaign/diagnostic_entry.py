"""Temporary, external diagnostic instrumentation; original engine source/config stay unchanged."""
import atexit
import functools
import json
import os
from pathlib import Path
import sys
import threading
import time

ROOT=Path('/workspace/minicpmo-r182')
OUT=Path(os.environ['MINICPMO_RESULT_DIR'])
sys.path.insert(0,os.environ['MINICPMO_ENGINE_ROOT'])
entry=ROOT/'run_engine_standalone.py'
prefix,separator,_=entry.read_text().partition('from engine.server import main')
assert separator
exec(compile(prefix,str(entry),'exec'),{'__name__':'r296_import_audit','__file__':str(entry)})
sys.path.insert(0,str(ROOT/'rsi-r286-r295'))
from capture_hooks import install
install()

import torch
from torch_npu.npu.mstx import mstx
from engine.runtime.backend_npu import AscendTrackSBackend as Backend
from engine.runtime.backend_s import LiveS

spans=[]
samples=[]
observations=[]
active=threading.Event()
flag=OUT/'profile.flag'

def stack(frame,limit=14):
    result=[]
    while frame is not None and len(result)<limit:
        result.append([frame.f_code.co_filename,frame.f_lineno,frame.f_code.co_name])
        frame=frame.f_back
    return result

def sampler():
    while True:
        if flag.exists():
            active.set()
            names={t.ident:(t.name,t.native_id) for t in threading.enumerate()}
            samples.append(dict(mono_ns=time.perf_counter_ns(),wall_ns=time.time_ns(),threads=[
                dict(name=names.get(k,('unknown',None))[0],tid=names.get(k,('unknown',None))[1],stack=stack(v))
                for k,v in sys._current_frames().items() if k!=threading.get_ident()]))
            time.sleep(.01)
        else:
            active.clear()
            time.sleep(.1)

threading.Thread(target=sampler,name='r296-python-stack-sampler',daemon=True).start()

for method in ('_tick','_thinker_step','_thinker_step_group','_talker_step','_talker_decode_block_impl',
               '_prepare','_setup_prompts','_decode_windows_threaded','_drain_windows','_admit','_pass_chunk'):
    original=getattr(Backend,method)
    def decorate(fn,name):
        @functools.wraps(fn)
        def wrapped(self,*args,**kwargs):
            begin=time.perf_counter_ns()
            fields={}
            if args and isinstance(args[0],list):
                fields['rows']=len(args[0])
                if args[0] and isinstance(args[0][0],LiveS):
                    fields['seqs']=[lv.seq for lv in args[0]]
                    fields['prefills']=sum(lv.pending_prefill for lv in args[0])
            profiling=active.is_set()
            marker=mstx.range_start('RSI:'+name) if profiling else None
            try:
                return fn(self,*args,**kwargs)
            finally:
                if profiling:
                    mstx.range_end(marker)
                spans.append(dict(name=name,start_ns=begin,end_ns=time.perf_counter_ns(),
                    tid=threading.get_native_id(),**fields))
        return wrapped
    setattr(Backend,method,decorate(original,method))

for name in ('cpu','tolist','item','copy_'):
    original=getattr(torch.Tensor,name)
    def decorate_observation(fn,name):
        @functools.wraps(fn)
        def wrapped(self,*args,**kwargs):
            if not active.is_set():
                return fn(self,*args,**kwargs)
            begin=time.perf_counter_ns()
            source=stack(sys._getframe(1))
            try:
                return fn(self,*args,**kwargs)
            finally:
                observations.append(dict(name=name,start_ns=begin,end_ns=time.perf_counter_ns(),
                    tid=threading.get_native_id(),stack=source))
        return wrapped
    setattr(torch.Tensor,name,decorate_observation(original,name))

def save():
    for name,value in (('stage-spans',spans),('python-samples',samples),('host-observations',observations)):
        (OUT/(name+'.json')).write_text(json.dumps(value)+'\n')
    (OUT/'diagnostic-clocks.json').write_text(json.dumps(dict(mono_ns=time.perf_counter_ns(),wall_ns=time.time_ns(),
        native_stack_capture=False,stack_source='10ms Python samples plus explicit host-observation stacks',
        profiler='External CANN msprof dynamic collection; no torch profiler start/stop'))+'\n')

original_stats=Backend.stats
def stats(self):
    save()
    return original_stats(self)
Backend.stats=stats
atexit.register(save)


# Supplemental successful graph capture/replay counters; diagnostic process only.
from collections import Counter
from engine.runtime.npu_paged import NpuDecodeGraph
graph_calls=Counter()
graph_original=NpuDecodeGraph.run
@functools.wraps(graph_original)
def graph_run(self,*args,**kwargs):
    phase='capture' if self.graph is None else 'replay'
    label=f'{self.runner.spec.prefix}:batch{self.batch}:{phase}'
    result=graph_original(self,*args,**kwargs)
    graph_calls[label]+=1
    return result
NpuDecodeGraph.run=graph_run
stats_before_graph=Backend.stats
def stats_graph(self):
    result=stats_before_graph(self)
    result['rsi_successful_graph_calls']=dict(graph_calls)
    result['rsi_thinker_graph_failures']=dict(self.trun.graph_failures)
    result['rsi_thinker_static_graph_failures']={str(k):v for k,v in self.trun.static_graph_failures.items()}
    return result
Backend.stats=stats_graph

from engine.server import main
raise SystemExit(main())
