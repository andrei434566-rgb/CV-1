"""Upload independent release volumes concurrently; never publish a partial release."""
from concurrent.futures import ThreadPoolExecutor,as_completed
import hashlib,json,time,urllib.request
import requests
from publish_offline_release import OUT,REPO,TAG,APP
from upload_release_assets import github_token

token=github_token()
def session():
    s=requests.Session();s.proxies.update(urllib.request.getproxies())
    s.headers.update({'Authorization':'Bearer '+token,'Accept':'application/vnd.github+json','User-Agent':'CV1-volume-upload'})
    return s
s=session();api=f'https://api.github.com/repos/{REPO}'
r=s.get(api+'/releases?per_page=100',timeout=45);r.raise_for_status()
release=next(x for x in r.json() if x['tag_name']==TAG)
if not release['draft']:raise ValueError('This uploader only operates on a draft')
r=s.get(api+f"/releases/{release['id']}/assets?per_page=100",timeout=45);r.raise_for_status()
existing={x['name']:x for x in r.json()}
names=[]
for prefix in ('CV1_Client_Data','CV1_Server_Data'):
    data=json.loads((OUT/(prefix+'.manifest.json')).read_text(encoding='utf8'))
    names.extend(p['name'] for p in data['parts'])
    names.append(prefix+'.manifest.json')

def upload(name):
    if name in existing:
        if existing[name]['state']=='starter':return 'IN PROGRESS ELSEWHERE '+name
        if existing[name]['size']!=(OUT/name).stat().st_size:raise ValueError('Conflicting remote asset '+name)
        return 'ALREADY UPLOADED '+name
    path=OUT/name;size=path.stat().st_size;started=time.monotonic()
    print('START',name,size,flush=True)
    class Progress:
        def __init__(self,f):self.f=f;self.total=0;self.notice=256*1024*1024
        def read(self,size=-1):
            data=self.f.read(size);self.total+=len(data)
            if self.total>=self.notice:
                print('SENT',name,round(self.total/1e6),'MB',round(time.monotonic()-started),'seconds',flush=True)
                self.notice+=256*1024*1024
            return data
        def tell(self):return self.f.tell()
        def seek(self,*args):return self.f.seek(*args)
        def __len__(self):return size
    connection=session()
    with path.open('rb') as f:
        r=connection.post(release['upload_url'].split('{')[0],params={'name':name},
            headers={'Content-Type':'application/octet-stream','Content-Length':str(size)},data=Progress(f),timeout=(45,7200))
    if r.status_code!=201:raise RuntimeError(f'{name} HTTP {r.status_code}: '+r.text[:250])
    with path.open('rb') as f:expected='sha256:'+hashlib.file_digest(f,'sha256').hexdigest()
    if r.json().get('digest') not in (None,expected):raise ValueError('Remote hash mismatch '+name)
    return f'UPLOADED {name} {round(time.monotonic()-started,1)} seconds'

with ThreadPoolExecutor(max_workers=3) as pool:
    pending=[pool.submit(upload,name) for name in names]
    for result in as_completed(pending):print(result.result(),flush=True)
print('PARALLEL VOLUMES COMPLETE; final full-release validation is still required.',flush=True)
