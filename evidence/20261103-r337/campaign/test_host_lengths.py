"""CPU functional gate for metadata provenance and conservative fallback."""
import ast
import json
from pathlib import Path
from types import SimpleNamespace

HERE=Path(__file__).resolve().parent
path=HERE/'engine-r319/engine/runtime/npu_paged.py'
module=ast.parse(path.read_text())
cls=next(n for n in module.body if isinstance(n,ast.ClassDef) and n.name=='NpuDecoderRunner')
method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_query_ends')
namespace={}
exec(compile(ast.Module(body=[method],type_ignores=[]),str(path),'exec'),namespace)
query_ends=namespace['_query_ends']

class Pointer:
    def __init__(self,values,calls):self.values=values;self.calls=calls
    def __getitem__(self,index):return Pointer(self.values[index],self.calls)
    def to(self,device):assert device=='cpu';self.calls.append('device-observation');return self
    def tolist(self):return self.values

records=[]
for counts in ([1],[1,1,1],[2,9,3],[128,1,17,3],[1]*8):
    qo=[0]
    for n in counts:qo.append(qo[-1]+n)
    calls=[]
    layout=SimpleNamespace(seqs=list(range(len(counts))),new_tokens=counts,host_qo_indptr=tuple(qo),qo_indptr=Pointer(qo,calls))
    assert query_ends(None,layout,qo[-1])==qo[1:] and not calls
    records.append(dict(case='valid_host',counts=counts,device_reads=len(calls)))
    layout.host_qo_indptr=None
    assert query_ends(None,layout,qo[-1])==qo[1:] and len(calls)==1
    records.append(dict(case='missing_host_fallback',counts=counts,device_reads=len(calls)))

for host,counts,seqs,query_rows in (
    ((1,2,3),[1,1],[0,1],3),
    ((0,1,4),[1,2],[0,1],4),
    ((0,1,3),[1,2],[0],3),
    ((0,1,3),[1,2],[0,1],4),
    ((0,1,3),[True,2],[0,1],3),
    ((0,0,3),[0,3],[0,1],3),
):
    calls=[];original=[0,2,5]
    layout=SimpleNamespace(seqs=seqs,new_tokens=counts,host_qo_indptr=host,qo_indptr=Pointer(original,calls))
    assert query_ends(None,layout,query_rows)==[2,5] and len(calls)==1
    records.append(dict(case='malformed_host_fallback',host=host,device_reads=len(calls)))

result=dict(passed=len(records),records=records,scope='CPU metadata/fallback contract only; real NPU generation parity remains mandatory')
(HERE/'host-lengths-cpu-tests.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(dict(passed=len(records),source=str(path))))
