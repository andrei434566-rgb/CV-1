"""Read every archived byte and compare it with the installation manifest."""
import hashlib,json,subprocess,zipfile,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from offline_installer import Volumes
from build_offline_release import OUT,BUILD
from app.infrastructure.model_hub import validate_registry

for prefix,exe in (('CV1_Client_Data','CV1_Setup'),('CV1_Server_Data','CV1_Server_Models')):
    control=json.loads((OUT/(prefix+'.manifest.json')).read_text(encoding='utf8'))
    with Volumes([OUT/p['name'] for p in control['parts']]) as joined,zipfile.ZipFile(joined) as archive:
        if {i.filename for i in archive.infolist()}!={r['path'] for r in control['files']}:raise ValueError('ZIP member mismatch')
        for index,item in enumerate(control['files'],1):
            digest=hashlib.sha256();size=0
            with archive.open(item['path']) as source:
                for chunk in iter(lambda:source.read(8*1024*1024),b''):digest.update(chunk);size+=len(chunk)
            if digest.hexdigest()!=item['sha256'] or size!=item['size']:raise ValueError('Archive checksum mismatch: '+item['path'])
            if index%3000==0:print('VERIFIED',prefix,index,flush=True)
        if prefix=='CV1_Server_Data':validate_registry(json.loads(archive.read('registry/models.json')))
    process=subprocess.run([str(OUT/(exe+'.exe')),'--verify-only'],timeout=600,creationflags=0x08000000)
    if process.returncode:raise ValueError('Frozen installer integrity check failed: '+exe)
    result=json.loads((OUT/(exe+'.verify.json')).read_text())
    if not result['ok']:raise ValueError(result)
    print('MEDIA_VERIFIED',prefix,len(control['files']),flush=True)
(BUILD/'offline_media_verified.json').write_text(json.dumps({'ok':True,'version':'2.1.0'}),encoding='utf8')
