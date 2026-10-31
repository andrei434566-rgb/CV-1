"""Exercise full installation on identical hardlinked files without a 10 GB copy."""
import json,os,subprocess,sys,shutil
from pathlib import Path
from build_offline_release import APP,BUILD,OUT,portable_file_map
from offline_installer import install

target=BUILD/'installed_verification'
target.mkdir(exist_ok=True)
for relative,source in portable_file_map(OUT/'Kern_Analyzer.exe').items():
    path=target/relative;path.parent.mkdir(parents=True,exist_ok=True)
    if not path.exists():
        try:os.link(source,path)
        except PermissionError:
            if not str(source).startswith(os.environ.get('ProgramFiles','C:/Program Files')):raise
            shutil.copyfile(source,path)
control=json.loads((OUT/'CV1_Client_Data.manifest.json').read_text(encoding='utf8'))
print('INSTALL_VERIFY_START',len(control['files']),flush=True)
install(control,OUT,target)
if not (target/'python_runtime/sitecustomize.py').is_file():raise RuntimeError('Runtime DLL setup missing')
process=subprocess.run([str(target/'Kern_Analyzer.exe'),'--portable-smoke-test'],cwd=target,
    env=dict(os.environ,QT_QPA_PLATFORM='offscreen'),timeout=120,creationflags=0x08000000)
if process.returncode:raise RuntimeError('Installed executable failed')
(BUILD/'installed_client_verified.json').write_text(json.dumps({'ok':True,'version':control['version']}),encoding='utf8')
print('INSTALLED_CLIENT_OK',flush=True)
