"""One audit per process, preserving model memory isolation and exit codes."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--start',type=int,default=11);p.add_argument('--end',type=int,default=16);a=p.parse_args()
 for n in range(a.start,a.end+1):
  start=time.monotonic();print(json.dumps(dict(event='audit_start',round=n)),flush=True)
  done=subprocess.run([sys.executable,'-u',str(ROOT/'scripts/audit_transfer_facies_round.py'),'--round',str(n)],cwd=ROOT,check=False)
  print(json.dumps(dict(event='audit_exit',round=n,exit_code=done.returncode,seconds=time.monotonic()-start)),flush=True)
  if done.returncode:raise SystemExit(done.returncode)
if __name__=='__main__':main()
