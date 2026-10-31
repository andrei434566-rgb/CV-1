"""Verify a separate installation layout without duplicating multi-GB files."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from build_offline_release import APP, BUILD, OUT, freeze, portable_file_map

stage=Path('\\\\?\\'+str(BUILD/'portable_verification')) if os.name=='nt' else BUILD/'portable_verification'
stage.mkdir(parents=True,exist_ok=True)
launcher=OUT/'Kern_Analyzer.exe'
OUT.mkdir(parents=True,exist_ok=True)
launcher=freeze(APP/'transfer/portable_launcher.py','Kern_Analyzer')
mapping=portable_file_map(launcher)
for index,(relative,source) in enumerate(mapping.items(),1):
    target=stage/relative
    target.parent.mkdir(parents=True,exist_ok=True)
    if relative=='Kern_Analyzer.exe' and target.exists():target.unlink()
    if not target.exists():
        try:os.link(source,target)
        except OSError:shutil.copyfile(source,target)
    if index%5000==0:print('Stage files',index,flush=True)
print('Portable staged files',len(mapping),flush=True)
python=stage/'python_runtime/python.exe'
check="import sys,torch,transformers,peft,cv2,openpyxl,docx,pypdf,ultralytics;from PySide6.QtWidgets import QApplication;print('OFFLINE_RUNTIME_OK',sys.executable,torch.__version__)"
result=subprocess.run([str(python),'-B','-c',check],capture_output=True,text=True,timeout=120,creationflags=0x08000000)
print(result.stdout,flush=True)
if result.returncode:
    print(result.stderr[-6000:],flush=True);raise SystemExit(result.returncode)
env=dict(os.environ,QT_QPA_PLATFORM='offscreen')
result=subprocess.run([str(stage/'Kern_Analyzer.exe'),'--portable-smoke-test'],cwd=stage,env=env,timeout=120,creationflags=0x08000000)
print('PACKAGED_APP_EXIT',result.returncode,flush=True)
log=stage/'launch_error.log'
if result.returncode and log.exists():print(log.read_text(encoding='utf-8')[-5000:],flush=True)
if result.returncode:raise SystemExit(result.returncode)
sys.path.insert(0,str(APP))
from app.infrastructure.model_hub import initialize_hub, make_registry_model, publish_model
fixture=BUILD/'launcher_hub_test';fixture.mkdir(exist_ok=True)
source=fixture/'source';source.mkdir(exist_ok=True);(source/'fixture.pt').write_bytes(b'fixture for file transport only')
hub=fixture/'hub';initialize_hub(hub)
if not (hub/'registry/models.json').exists():
    model=make_registry_model(source,model_id='fixture',role='geometry',version='v1',entrypoint='fixture.pt',approval={'approved':True})
    publish_model(hub,source,model,baseline=True)
job=fixture/'job.json';answer=fixture/'result.json'
job.write_text(json.dumps({'operation':'sync','hub':str(hub),'cache':str(fixture/'cache'),'result':str(answer)}),encoding='utf8')
result=subprocess.run([str(stage/'Kern_Analyzer.exe'),'--hub-sync-job',str(job)],cwd=stage,env=env,timeout=45,creationflags=0x08000000)
if result.returncode or not json.loads(answer.read_text(encoding='utf8'))['ok']:raise RuntimeError('Packaged model-hub subprocess failed')
print('PACKAGED_HUB_JOB_OK',flush=True)
(BUILD/'portable_verification_passed.json').write_text(json.dumps({'ok':True,'files':len(mapping),'stage':str(stage)}),encoding='utf-8')
