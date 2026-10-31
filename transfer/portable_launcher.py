"""Tiny frozen entry point; all application modules run in the shipped Python."""
import ctypes
import os
from pathlib import Path
import subprocess
import sys

def main():
    root=Path(sys.executable).resolve().parent
    job=any(value in sys.argv for value in ('--portable-smoke-test','--hub-sync-job'))
    python=root/'python_runtime'/('python.exe' if job else 'pythonw.exe')
    script=root/'_internal/portable_bootstrap.py'
    if not python.is_file() or not script.is_file():
        raise FileNotFoundError('Установите полный комплект CV-1 вместе с Python и моделями. Один EXE отдельно не запускается.')
    log=root/'Kern Analyzer Data/logs';log.mkdir(parents=True,exist_ok=True)
    with (log/'launcher.log').open('ab') as output:
        child=subprocess.Popen([str(python),'-B',str(script),*sys.argv[1:]],cwd=root,
            stdin=subprocess.DEVNULL,stdout=output,stderr=output,creationflags=0x08000000)
        return child.wait() if job else 0

if __name__=='__main__':
    try:raise SystemExit(main())
    except SystemExit:raise
    except Exception as exc:
        if not any(value in sys.argv for value in ('--portable-smoke-test','--hub-sync-job')):
            ctypes.windll.user32.MessageBoxW(0,str(exc),'CV-1',0x10)
        raise SystemExit(1)
