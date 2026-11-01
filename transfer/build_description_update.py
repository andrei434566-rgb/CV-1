"""Small code-only update for the installed full 2.1.0 distribution."""
import json,zipfile
from build_offline_release import APP,OUT,BUILD,VERSION,freeze,VolumeWriter,digest

OUT.mkdir(parents=True,exist_ok=True)
files=['infrastructure/description_output.py','infrastructure/local_vlm_runner.py',
       'infrastructure/local_vlm_client.py','ui/windows/main_window.py']
control={'schema':'cv1-offline-installer-v1','kind':'update','version':VERSION,
         'compatible_versions':['2.1.0',VERSION],'files':[],'unpacked_bytes':0}
writer=VolumeWriter(OUT,'CV1_Update_Data')
with zipfile.ZipFile(writer,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
    for name in files:
        source=APP/'app'/name;relative='_internal/app/'+name
        archive.write(source,relative)
        control['files'].append({'path':relative,'size':source.stat().st_size,'sha256':digest(source)})
        control['unpacked_bytes']+=source.stat().st_size
control['parts']=writer.finish()
manifest=BUILD/'description_update/release_control.json'
manifest.parent.mkdir(parents=True,exist_ok=True)
manifest.write_text(json.dumps(control,ensure_ascii=False),encoding='utf8')
(OUT/'CV1_Update_Data.manifest.json').write_text(json.dumps(control,ensure_ascii=False,indent=2),encoding='utf8')
exe=freeze(APP/'transfer/offline_installer.py','CV1_Update_'+VERSION,manifest)
assets=[exe,OUT/'CV1_Update_Data.manifest.json',*[OUT/part['name'] for part in control['parts']]]
(OUT/'SHA256SUMS.txt').write_text('\n'.join(digest(p)+'  '+p.name for p in assets)+'\n',encoding='ascii')
print('CODE_ONLY_UPDATE_READY',OUT,'FILES',len(files),flush=True)
