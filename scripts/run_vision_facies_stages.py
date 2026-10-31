"""Run vision-training folds in separate processes with durable exit receipts."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--audit', action='store_true', help='Replay each saved adapter in a fresh process, then aggregate receipts.')
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(ROOT.parent / 'vlm_core_v1/outputs'):
        raise ValueError('Choose a local experiment output')
    if args.audit:
        if not (output / 'result.json').is_file():
            raise ValueError('Finish training before audit')
        stages = ('fold0', 'fold1', 'fold2', 'final', 'aggregate')
        script, flag = 'audit_vision_facies_v9.py', '--run'
    else:
        if output.exists():
            raise ValueError('Choose a new local experiment output')
        stages = ('fold0', 'fold1', 'fold2', 'finalize')
        script, flag = 'train_vision_facies_v9.py', '--output'
    journal = output.with_suffix('.audit_stages.jsonl' if args.audit else '.stages.jsonl')
    if journal.exists():
        raise ValueError('Stage journal already exists')
    for stage in stages:
        start = time.monotonic()
        event = dict(event='stage_started', stage=stage)
        print(json.dumps(event), flush=True)
        with journal.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(event) + '\n')
        process = subprocess.run(
            [sys.executable, '-u', str(ROOT / 'scripts' / script),
             flag, str(output), '--stage', stage], cwd=ROOT, check=False)
        event = dict(event='stage_exited', stage=stage, exit_code=process.returncode,
                     seconds=time.monotonic() - start)
        with journal.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(event) + '\n')
        print(json.dumps(event), flush=True)
        if process.returncode:
            raise SystemExit(process.returncode)
    expected = 'audit_result.json' if args.audit else 'result.json'
    if not (output / expected).is_file():
        raise RuntimeError('All stages exited but final result is missing: ' + expected)


if __name__ == '__main__':
    main()
