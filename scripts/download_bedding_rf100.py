"""Pinned public CC BY 4.0 source dataset; no model API or credentials required."""
import hashlib,json,sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import requests
ROOT=Path(__file__).resolve().parents[1]
REPO='LibreYOLO/sedimentary-features-9eosf'
REVISION='c6da8adf32587a4f5b2a1b6e1adee8ecbdab268b'

def main():
    target=(ROOT/'open_datasets/bedding_20261106/rf100').resolve();target.mkdir(parents=True,exist_ok=True)
    r=requests.get(f'https://huggingface.co/api/datasets/{REPO}/tree/{REVISION}?recursive=true',timeout=40);r.raise_for_status()
    files=[v for v in r.json() if v['type']=='file' and not v['path'].startswith('.git')]
    # Resolve and create directories sequentially: Windows parent resolution while
    # another worker creates that same directory can return inconsistent casing.
    for row in files:
        path=(target/row['path']).resolve()
        if not path.is_relative_to(target):raise ValueError('Dataset path escape')
        path.parent.mkdir(parents=True,exist_ok=True)
    def fetch(row):
        path=(target/row['path']).resolve()
        if not path.is_relative_to(target):raise ValueError('Dataset path escape: '+repr((row['path'],str(path),str(target))))
        path.parent.mkdir(parents=True,exist_ok=True)
        if not path.exists() or path.stat().st_size!=row['size']:
            for attempt in range(3):
                try:
                    r=requests.get(f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{row['path']}",timeout=(20,60));r.raise_for_status()
                    if len(r.content)!=row['size']:raise ValueError('Size mismatch')
                    path.write_bytes(r.content);break
                except requests.RequestException:
                    if attempt==2:raise
        return {'path':row['path'],'size':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    rows=[]
    with ThreadPoolExecutor(max_workers=6) as executor:
        for i,row in enumerate(executor.map(fetch,files)):
            rows.append(row)
            if i%100==0:print(json.dumps({'downloaded':i+1,'total':len(files)}),flush=True)
    (target/'source_manifest.json').write_text(json.dumps({'repository':REPO,'revision':REVISION,
        'source_author':'Ammar Abdlmutalib','license':'CC BY 4.0',
        'original_source':'https://universe.roboflow.com/roboflow-100/sedimentary-features-9eosf/dataset/4',
        'known_preprocessing':'EXIF auto-orientation; resized to 640 x 640 with stretch; no augmentation',
        'files':rows},ensure_ascii=False,indent=2),encoding='utf8')
    print('Downloaded '+str(len(rows))+' source files',flush=True)

if __name__=='__main__':main()
