"""Retry one failed asset of the current draft, guarding against deleting good files."""
import argparse,hashlib,time
import requests,urllib.request
from publish_offline_release import OUT,REPO,TAG
from upload_release_assets import github_token

p=argparse.ArgumentParser();p.add_argument('name');p.add_argument('--failed-starter-id',type=int,required=True);args=p.parse_args()
path=OUT/args.name
if path.parent!=OUT or not path.is_file():raise ValueError('Unknown local release asset')
s=requests.Session();s.proxies.update(urllib.request.getproxies());s.headers.update({'Authorization':'Bearer '+github_token(),'Accept':'application/vnd.github+json','User-Agent':'CV1-release-retry'})
api=f'https://api.github.com/repos/{REPO}'
r=s.get(api+'/releases?per_page=100',timeout=60);r.raise_for_status();release=next(x for x in r.json() if x['tag_name']==TAG)
if not release['draft']:raise ValueError('Only this draft can be repaired')
r=s.get(api+f"/releases/{release['id']}/assets?per_page=100",timeout=60);r.raise_for_status()
failed=next(x for x in r.json() if x['id']==args.failed_starter_id)
if failed['name']!=args.name or failed['state']!='starter' or failed.get('digest'):raise ValueError('Refusing to remove a completed or unrelated asset')
r=s.delete(api+f'/releases/assets/{failed["id"]}',timeout=60);r.raise_for_status()
print('Removed only failed draft transfer',args.name,flush=True)
size=path.stat().st_size;started=time.monotonic()
class Progress:
    def __init__(self,f):self.f=f;self.total=0;self.next=128*1024*1024
    def read(self,size=-1):
        data=self.f.read(size);self.total+=len(data)
        if self.total>=self.next:
            print('SENT',args.name,round(self.total/1e6),'MB',round(time.monotonic()-started),'seconds',flush=True);self.next+=128*1024*1024
        return data
    def __len__(self):return size
    def tell(self):return self.f.tell()
    def seek(self,*a):return self.f.seek(*a)
with path.open('rb') as f:
    r=s.post(release['upload_url'].split('{')[0],params={'name':args.name},headers={'Content-Type':'application/octet-stream','Content-Length':str(size)},data=Progress(f),timeout=(60,7200))
if r.status_code!=201:raise RuntimeError(f'HTTP {r.status_code}: '+r.text[:300])
with path.open('rb') as f:digest='sha256:'+hashlib.file_digest(f,'sha256').hexdigest()
if r.json().get('digest') not in (None,digest):raise ValueError('Remote checksum mismatch')
print('UPLOADED',args.name,'VERIFIED',digest,flush=True)
