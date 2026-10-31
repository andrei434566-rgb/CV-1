"""Fetch official LibreOffice administrative media for local document QA only."""
import hashlib,json
from pathlib import Path
import urllib.request
import requests
root=Path(__file__).resolve().parents[1]/'outputs/build_cv1_210'
url='https://download.documentfoundation.org/libreoffice/stable/26.8.2/win/x86_64/LibreOffice_26.8.2_Win_x86-64.msi'
s=requests.Session();s.proxies.update(urllib.request.getproxies())
p=root/'LibreOffice.msi'
with s.get(url,stream=True,timeout=(30,90)) as r:
    r.raise_for_status()
    with p.open('wb') as f:
        for block in r.iter_content(4*1024*1024):f.write(block)
checksum=hashlib.file_digest(p.open('rb'),'sha256').hexdigest()
check=s.get(url+'.sha256',timeout=45);check.raise_for_status()
if checksum not in check.text:raise ValueError('Official renderer checksum mismatch')
(root/'libreoffice_source.json').write_text(json.dumps({'url':url,'sha256':checksum,'bytes':p.stat().st_size}),encoding='utf8')
print('Verified official rendering media',p.stat().st_size,flush=True)
