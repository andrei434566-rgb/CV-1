"""Publish full offline media using the existing GitHub credential helper."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time
import urllib.request
import requests
from upload_release_assets import github_token

APP=Path(__file__).resolve().parents[1]
OUT=APP/'transfer/out/cv1-v2.1.0'
REPO='andrei434566-rgb/CV-1'
TAG='cv1-v2.1.0'

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--publish',action='store_true');args=parser.parse_args()
    manifests=[json.loads((OUT/(name+'.manifest.json')).read_text(encoding='utf8')) for name in ('CV1_Client_Data','CV1_Server_Data')]
    names=['CV1_Setup.exe','CV1_Server_Models.exe','CV1_User_Manual_RU.docx','SHA256SUMS.txt']
    for name,manifest in zip(('CV1_Client_Data','CV1_Server_Data'),manifests):
        names.append(name+'.manifest.json');names.extend(part['name'] for part in manifest['parts'])
    if not all((OUT/name).is_file() for name in names):raise ValueError('Incomplete offline release')
    commit=subprocess.run(['git','rev-parse','HEAD'],cwd=APP,text=True,capture_output=True,check=True).stdout.strip()
    s=requests.Session();s.proxies.update(urllib.request.getproxies())
    s.headers.update({'Authorization':'Bearer '+github_token(),'Accept':'application/vnd.github+json',
        'X-GitHub-Api-Version':'2022-11-28','User-Agent':'CV1-offline-release'})
    api=f'https://api.github.com/repos/{REPO}'
    r=s.get(api+'/releases?per_page=100',timeout=45);r.raise_for_status()
    release=next((v for v in r.json() if v['tag_name']==TAG),None)
    client_bytes=sum(p['size'] for p in manifests[0]['parts'])
    server_bytes=sum(p['size'] for p in manifests[1]['parts'])
    body=f'''Автономный CV-1 2.1.0 с локальной Qwen3-VL-2B, исходными весами геометрии и общим хранилищем моделей.

Для рабочего компьютера скачайте **CV1_Setup.exe и все CV1_Client_Data.partNNN** из этого релиза, положите рядом и запустите EXE. Объём частей {client_bytes/1e9:.2f} ГБ; после установки {manifests[0]['unpacked_bytes']/1e9:.2f} ГБ. Python, CUDA-библиотеки, Qwen и OCR входят в комплект; скачивание модели при запуске не требуется. Для VLM нужен совместимый NVIDIA GPU и драйвер.

Для общей серверной папки отдельно скачайте **CV1_Server_Models.exe и все CV1_Server_Data.partNNN** ({server_bytes/1e9:.2f} ГБ). Установщик распакует версионные модели и registry/models.json. Администратор открывает общий доступ средствами Windows; в приложении путь задаётся через ⚙ → Общие модели и данные… . Клиенты запускают проверенные модели из локального кеша, сохраняют автономную работу и поддерживают откат. Передаются только подтверждённые интервалы. Центральное обучение и публикация новой версии выполняются отдельно после проверки на независимых скважинах.

**CV1_User_Manual_RU.docx** — подробная инструкция на 16 страницах: установка, фото, глубины, редактор свойств, чат, типовой отчёт, приоритетные материалы, обучение по Word и фото, общий сервер и команды администратора. Встроенная инструкция открывается клавишей F1.

Excel: ярко-красный фон проблемных ячеек, чёрный текст, полные границы. Экспорт допускается без обязательного заполнения.

Проверено 84 автоматическими тестами, автономный запуск Qt/PyTorch и отдельный процесс синхронизации. Архивы и файлы проверяются SHA256. Это базовый комплект; принятый геологический LoRA-адаптер и точность 95% не заявляются. Старые двухклассовые веса находятся в optional_candidates и автоматически не включаются. Проверка на вашем втором ПК и настройка реального сетевого доступа выполняются после установки.

Исходники: main, коммит {commit}. Файлы частей разных выпусков не смешивайте. Самостоятельного переноса одного Kern_Analyzer.exe недостаточно: нужны соседние библиотеки и модели.
'''
    if release is None:
        r=s.post(api+'/releases',json={'tag_name':TAG,'target_commitish':commit,'name':'CV-1 2.1.0 — автономный клиент и общие модели','body':body,'draft':True,'prerelease':False},timeout=45)
        r.raise_for_status();release=r.json()
    elif not release['draft']:
        print('Already published',release['html_url']);return
    else:
        r=s.patch(api+f"/releases/{release['id']}",json={'body':body,'tag_name':TAG,'target_commitish':commit},timeout=45);r.raise_for_status()
    (APP/'outputs/build_cv1_210/github_release.json').write_text(json.dumps({'id':release['id'],'tag':TAG,'url':release['html_url']}),encoding='utf8')
    r=s.get(api+f"/releases/{release['id']}/assets?per_page=100",timeout=45);r.raise_for_status()
    existing={a['name']:a for a in r.json()}
    upload=release['upload_url'].split('{')[0]
    for name in names:
        path=OUT/name;size=path.stat().st_size
        if name in existing:
            if existing[name]['size']!=size or existing[name]['state']!='uploaded':raise ValueError('Inspect incomplete or conflicting asset: '+name)
            print('Already uploaded',name,flush=True);continue
        print('UPLOADING',name,size,flush=True);started=time.monotonic()
        with path.open('rb') as f:
            r=s.post(upload,params={'name':name},headers={'Content-Type':'application/octet-stream','Content-Length':str(size)},data=f,timeout=(45,7200))
        if r.status_code!=201:raise RuntimeError(f'Upload failed {name}: HTTP {r.status_code} '+r.text[:300])
        asset=r.json()
        with path.open('rb') as f:expected='sha256:'+hashlib.file_digest(f,'sha256').hexdigest()
        if asset.get('digest') and asset['digest']!=expected:raise ValueError('Remote digest mismatch '+name)
        print('UPLOADED',name,round(time.monotonic()-started,1),'seconds',flush=True)
    r=s.get(api+f"/releases/{release['id']}/assets?per_page=100",timeout=45);r.raise_for_status()
    remote={v['name']:v for v in r.json()}
    if any(name not in remote or remote[name]['size']!=(OUT/name).stat().st_size for name in names):raise RuntimeError('Final remote asset check failed')
    checksums={line.split('  ',1)[1]:line.split('  ',1)[0] for line in (OUT/'SHA256SUMS.txt').read_text().splitlines()}
    checksums['SHA256SUMS.txt']=hashlib.sha256((OUT/'SHA256SUMS.txt').read_bytes()).hexdigest()
    for name in names:
        if remote[name]['state']!='uploaded':raise ValueError('Asset is not complete: '+name)
        if remote[name].get('digest') not in (None,'sha256:'+checksums[name]):raise ValueError('Final remote checksum mismatch: '+name)
    if args.publish:
        r=s.patch(api+f"/releases/{release['id']}",json={'tag_name':TAG,'target_commitish':commit,'draft':False,'make_latest':'true'},timeout=45);r.raise_for_status()
        print('PUBLISHED',r.json()['html_url'],flush=True)
    else:print('DRAFT COMPLETE. Run with --publish after media verification.',flush=True)

if __name__=='__main__':main()
