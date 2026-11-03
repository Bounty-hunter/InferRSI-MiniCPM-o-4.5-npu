"""Run only fixed-input component and 2x2 experiments after scope clarification."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
OUT = Path('/workspace/minicpmo-r182/artifacts') / HERE.name
assert os.getuid() == 1033 and os.getgid() == 1000 and os.environ.get('ASCEND_RT_VISIBLE_DEVICES')
tasks = []
reference = OUT / 'fixture-reference.json'
if reference.exists():
    fixture = json.loads(reference.read_text())
    assert hashlib.sha256((Path(fixture['directory']) / 'manifest.json').read_bytes()).hexdigest() == fixture['manifest_sha256']
else:
    tasks.append(('fixed-inputs-r2', [str(HERE / 'capture_fixed.py'), '--tag', 'fixed-inputs-r2']))
for tag in ('A1', 'B1', 'B2', 'A2', 'A3', 'B3'):
    tasks.append((f'replay-{tag}', [str(HERE / 'replay_fixed.py'), '--tag', f'replay-{tag}', '--role', tag[0]]))
for role in ('A', 'B'):
    tasks.append((f'profile-{role}', [str(HERE / 'replay_fixed.py'), '--tag', f'profile-{role}', '--role', role, '--profile']))


def save(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2) + '\n')


save('component-plan.json', dict(scope='Fixed-work component and serial/multistream 2x2 only', tasks=tasks,
     visible_devices=os.environ['ASCEND_RT_VISIBLE_DEVICES'],
     programs={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in HERE.glob('*.py')}))
try:
    for tag, arguments in tasks:
        command = [sys.executable, *arguments]
        save('component-status.json', dict(phase='running', tag=tag, command=command,
                                           utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())))
        with (OUT / f'{tag}-controller.log').open('w') as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            save(f'{tag}-controller-pid.json', dict(pid=process.pid))
            code = process.wait()
        save(f'{tag}-controller-exit.json', dict(exit_code=code))
        assert code == 0, (tag, code)
    save('component-status.json', dict(phase='completed'))
except BaseException as error:
    save('component-status.json', dict(phase='failed', error=repr(error)))
    raise
