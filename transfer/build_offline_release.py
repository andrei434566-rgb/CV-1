"""Reproducible offline client + separate server models, all files below 2 GiB."""
from __future__ import annotations
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

APP = Path(__file__).resolve().parents[1]
VLM = APP.parent / "vlm_core_v1"
BUILD = APP / "outputs/build_cv1_210"
OUT = APP / "transfer/out/cv1-v2.1.0"
REVISION = "89644892e4d85e24eaac8bacfd4f463576704203"
BASE = VLM / "model_cache/hub/models--Qwen--Qwen3-VL-2B-Instruct/snapshots" / REVISION
VERSION = "2.1.0"
SKIP = {"__pycache__", ".git", "node_modules", "tests", "test", ".pytest_cache"}


def digest(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024),b''):value.update(block)
    return value.hexdigest()


def files_below(root):
    root=Path(root)
    for current,dirs,names in os.walk(root,followlinks=False):
        dirs[:]=[name for name in dirs if name not in SKIP and not name.endswith('.egg-info')]
        for name in sorted(names):
            path=Path(current)/name
            if path.is_file() and not path.is_symlink() and path.suffix not in {'.pyc','.pyo'}:
                yield path


class VolumeWriter(io.RawIOBase):
    def __init__(self, root, prefix, limit=1_500_000_000):
        self.root,self.prefix,self.limit=Path(root),prefix,limit
        self.position,self.in_part=0,0
        self.stream,self.hash=None,None
        self.parts=[]
    def writable(self):return True
    def seekable(self):return False
    def tell(self):return self.position
    def seek(self,*args):raise OSError('streaming archive')
    def _finish_part(self):
        if self.stream:
            self.stream.close()
            self.parts[-1].update(size=self.in_part,sha256=self.hash.hexdigest())
            self.stream=None
    def write(self,data):
        size=len(data);view=memoryview(data)
        while view:
            if self.stream is None:
                name=f'{self.prefix}.part{len(self.parts)+1:03d}'
                self.stream=(self.root/name).open('wb');self.hash=hashlib.sha256();self.in_part=0
                self.parts.append({'name':name})
            length=min(len(view),self.limit-self.in_part)
            chunk=view[:length];self.stream.write(chunk);self.hash.update(chunk)
            self.in_part+=length;self.position+=length;view=view[length:]
            if self.in_part==self.limit:self._finish_part()
        return size
    def finish(self):self._finish_part();return self.parts


def freeze(script, name, data=None):
    spec=BUILD/'specs';spec.mkdir(parents=True,exist_ok=True)
    command=[sys.executable,'-m','PyInstaller','--noconfirm','--clean','--onefile','--windowed',
        '--name',name,'--distpath',str(OUT),'--workpath',str(BUILD/'freeze'/name),'--specpath',str(spec)]
    for module in ('torch','torchvision','transformers','peft','PySide6','cv2','numpy','scipy','pandas'):
        command+=['--exclude-module',module]
    if data:command+=['--add-data',str(data)+os.pathsep+'.']
    command.append(str(script))
    log=BUILD/(name+'.build.log')
    print('BUILD',name,flush=True)
    with log.open('w',encoding='utf-8') as output:
        result=subprocess.run(command,cwd=APP,stdout=output,stderr=subprocess.STDOUT)
    if result.returncode:raise RuntimeError('PyInstaller failed. See '+str(log))
    return OUT/(name+'.exe')


def portable_file_map(launcher):
    result={'Kern_Analyzer.exe':launcher}
    for source in files_below(APP/'app'):
        result['_internal/app/'+source.relative_to(APP/'app').as_posix()]=source
    result['_internal/run.py']=APP/'run.py'
    result['_internal/portable_bootstrap.py']=APP/'transfer/portable_bootstrap.py'
    studio=APP/'excel_photo_model_studio/src'
    for source in files_below(studio):
        result['_internal/excel_photo_model_studio/src/'+source.relative_to(studio).as_posix()]=source
    for source in files_below(BUILD/'python_runtime'):
        result['python_runtime/'+source.relative_to(BUILD/'python_runtime').as_posix()]=source
    site=VLM/'.venv/Lib/site-packages'
    for source in files_below(site):
        # Build tools are not required on the geologist's computer.
        parts=source.relative_to(site).parts
        if parts[0].lower().startswith(('pyinstaller','_pyinstaller','pip-','pip','altgraph','pefile','pywin32_ctypes')):continue
        if parts[0]=='PySide6' and len(parts)>1 and parts[1] in {'qml','include','typesystems','lib','glue','scripts','doc','examples'}:continue
        if source.suffix.lower() in {'.obj','.lib','.pdb','.h','.hpp','.cpp','.a'}:continue
        result['python_runtime/Lib/site-packages/'+source.relative_to(site).as_posix()]=source
    for source in files_below(BASE):
        if not source.name.startswith('.'):
            result['bundled_models/vlm/Qwen3-VL-2B-Instruct/'+source.relative_to(BASE).as_posix()]=source
    result['bundled_models/geometry/best.pt']=APP/'models/best.pt'
    for index,source in enumerate(sorted((APP/'models/trained').rglob('best.pt')),1):
        result[f'bundled_models/optional_candidates/geometry_candidate_{index}.pt']=source
    for source in files_below(VLM/'data/knowledge'):
        result['_internal/knowledge/'+source.relative_to(VLM/'data/knowledge').as_posix()]=source
    ocr=Path(r'C:\Program Files\Tesseract-OCR')
    for source in files_below(ocr):
        if source.suffix.lower() in {'.dll','.exe','.traineddata','.txt'}:
            result['_internal/tools/tesseract/'+source.relative_to(ocr).as_posix()]=source
    russian=BUILD/'ocr/rus.traineddata'
    if not russian.is_file():raise RuntimeError('Download official Russian OCR data before packaging')
    result['_internal/tools/tesseract/tessdata/rus.traineddata']=russian
    for source in files_below(APP/'docs'):
        result['_internal/docs/'+source.relative_to(APP/'docs').as_posix()]=source
    manual=OUT/'CV1_User_Manual_RU.docx'
    if not manual.is_file():raise RuntimeError('Create and render the user manual before packaging')
    result['_internal/docs/CV1_User_Manual_RU.docx']=manual
    for source in (APP/'README.md',APP/'README_PRIMARY_SOURCE_RU.md',APP/'README_WORD_PHOTO_TRAINING_RU.md',APP/'VERSION.txt',APP/'transfer/QWEN_APACHE_2.0.txt'):
        if source.is_file():result['_internal/'+source.name]=source
    for source in files_below(APP/'scripts'):
        if source.name in {'model_hub_admin.py','model_hub_train.py'}:result['_internal/scripts/'+source.name]=source
    defaults=BUILD/'bundle_runtime.json'
    defaults.write_text(json.dumps({'schema':'cv1-offline-runtime-v1','python':'python_runtime/python.exe',
        'base':'bundled_models/vlm/Qwen3-VL-2B-Instruct','geometry':'bundled_models/geometry/best.pt','knowledge':'_internal/knowledge'}),encoding='utf-8')
    result['bundle_runtime.json']=defaults
    return result


def pack(mapping, name, kind):
    destination=BUILD/name;destination.mkdir(parents=True,exist_ok=True)
    control={'schema':'cv1-offline-installer-v1','version':VERSION,'kind':kind,'files':[],'parts':[],'unpacked_bytes':0}
    writer=VolumeWriter(OUT,name)
    total=len(mapping)
    with zipfile.ZipFile(writer,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1,allowZip64=True) as archive:
        for index,(relative,source) in enumerate(sorted(mapping.items()),1):
            source=Path(source)
            size=source.stat().st_size;checksum=digest(source)
            archive.write(source,relative)
            control['files'].append({'path':relative,'size':size,'sha256':checksum})
            control['unpacked_bytes']+=size
            if index%500==0 or size>100_000_000:print(f'{kind}: {index}/{total} {relative} · {writer.position/1e9:.2f} GB',flush=True)
    control['parts']=writer.finish()
    manifest=destination/'release_control.json'
    manifest.write_text(json.dumps(control,ensure_ascii=False),encoding='utf-8')
    (OUT/(name+'.manifest.json')).write_text(json.dumps(control,ensure_ascii=False,indent=2),encoding='utf-8')
    freeze(APP/'transfer/offline_installer.py','CV1_Server_Models' if kind=='server' else 'CV1_Setup',manifest)
    print('PACKED',kind,control['unpacked_bytes'],len(control['parts']),flush=True)
    return control


def server_map(client):
    sys.path.insert(0,str(APP))
    from app.infrastructure.model_hub import make_registry_model
    folder=BUILD/'server_registry';folder.mkdir(parents=True,exist_ok=True)
    approval={'approved':True,'scope':'Базовый комплект. Проверка запуска, без аттестации геологической точности','independent_wells':[]}
    # The files of this baseline are exactly the installed, unmodified weights.
    base_model=make_registry_model(BASE,model_id='Qwen3-VL-2B-Instruct',role='vlm_base',version=REVISION,entrypoint='config.json',approval=approval)
    base_model['files']=[row for row in base_model['files'] if not Path(row['path']).name.startswith('.')]
    geometry_source=folder/'geometry';geometry_source.mkdir(exist_ok=True)
    shutil.copyfile(APP/'models/best.pt',geometry_source/'best.pt')
    geometry=make_registry_model(geometry_source,model_id='geometry',role='geometry',version='baseline-2.1.0',entrypoint='best.pt',approval=approval)
    registry={'schema':'cv1-model-registry-v1','release_id':'baseline-2.1.0','models':[geometry,base_model],
        'columns':{'method':'builtin_detector','weights_included':False},'note':'LoRA-кандидаты и модель точности 95% не заявляются.'}
    (folder/'models.json').write_text(json.dumps(registry,ensure_ascii=False,indent=2),encoding='utf-8')
    (folder/'hub.json').write_text(json.dumps({'schema':'cv1-network-hub-v1','initial_package':VERSION}),encoding='utf-8')
    result={'registry/models.json':folder/'models.json','hub.json':folder/'hub.json',
            'models/geometry/baseline-2.1.0/best.pt':APP/'models/best.pt'}
    result['documentation/QWEN_APACHE_2.0.txt']=APP/'transfer/QWEN_APACHE_2.0.txt'
    for source in files_below(BASE):
        if not source.name.startswith('.'):
            result[f'models/Qwen3-VL-2B-Instruct/{REVISION}/'+source.relative_to(BASE).as_posix()]=source
    for relative,source in client.items():
        if relative.startswith('bundled_models/optional_candidates/'):
            result['optional_candidates/'+Path(relative).name]=source
    for relative,source in client.items():
        if relative.startswith('_internal/docs/'):
            result['documentation/'+Path(relative).name]=source
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--kind',choices=['client','server','all'],default='all');args=parser.parse_args()
    BUILD.mkdir(parents=True,exist_ok=True);OUT.mkdir(parents=True,exist_ok=True)
    launcher=OUT/'Kern_Analyzer.exe'
    if not launcher.is_file():launcher=freeze(APP/'transfer/portable_launcher.py','Kern_Analyzer')
    client=portable_file_map(launcher)
    if args.kind in {'client','all'}:pack(client,'CV1_Client_Data','client')
    if args.kind in {'server','all'}:pack(server_map(client),'CV1_Server_Data','server')
    hashes=[]
    for path in sorted(OUT.iterdir()):
        if path.is_file() and path.name not in {'SHA256SUMS.txt','Kern_Analyzer.exe'} and not path.name.endswith('.verify.json'):
            hashes.append(digest(path)+'  '+path.name)
    (OUT/'SHA256SUMS.txt').write_text('\n'.join(hashes)+'\n',encoding='ascii')
    print('RELEASE READY',OUT,flush=True)


if __name__=='__main__':main()
