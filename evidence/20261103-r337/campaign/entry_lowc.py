"""Import audit, optional diagnostics. No source mutations at runtime."""
import os
from pathlib import Path
import sys

ROOT=Path('/workspace/minicpmo-r182')
HERE=Path(__file__).resolve().parent
sys.path.insert(0,os.environ['MINICPMO_ENGINE_ROOT'])
if os.environ.get('RSI_DIAGNOSTIC')=='1':
    import runpy
    runpy.run_path(str(HERE/'diagnostic_entry.py'),run_name='__main__')
    raise SystemExit('Unexpected return')
entry=ROOT/'run_engine_standalone.py'
prefix,sep,_=entry.read_text().partition('from engine.server import main')
assert sep
exec(compile(prefix,str(entry),'exec'),{'__name__':'r297_import_audit','__file__':str(entry)})
if os.environ.get('RSI_QUALITY')=='1':
    sys.path.insert(0,str(ROOT/'rsi-r286-r295'))
    sys.path.insert(0,str(HERE))
    from quality_hooks_lowc import install
    install()
from engine.server import main
raise SystemExit(main())
