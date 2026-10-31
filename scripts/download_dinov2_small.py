"""Download only public, revision-pinned DINOv2 data files; no remote code."""
import hashlib
import json
from pathlib import Path
import requests

ROOT=Path(__file__).resolve().parents[1]
REVISION='ed25f3a31f01632728cabb09d1542f84ab7b0056'


def main():
    model='facebook/dinov2-small'
    root=ROOT.parent/'vlm_core_v1/model_cache/local_dinov2_small'/REVISION
    if (root/'download_receipt.json').is_file():
        receipt=json.loads((root/'download_receipt.json').read_text(encoding='utf-8'))
        if receipt['revision']!=REVISION or any(hashlib.sha256((root/name).read_bytes()).hexdigest()!=value for name,value in receipt['sha256'].items()):raise ValueError('Cached DINO differs; preserve it for investigation')
        print('Cached DINO revision and all file hashes verified.',flush=True)
        return
    response=requests.get('https://huggingface.co/api/models/'+model+'/revision/'+REVISION,timeout=45)
    response.raise_for_status();meta=response.json();revision=meta['sha']
    if revision!=REVISION or meta.get('cardData',{}).get('license')!='apache-2.0':raise ValueError('Unexpected revision/license')
    root.mkdir(parents=True,exist_ok=False);hashes={};sizes={}
    for name in ('config.json','preprocessor_config.json','model.safetensors','README.md'):
        url=f'https://huggingface.co/{model}/resolve/{revision}/{name}'
        response=requests.get(url,stream=True,timeout=(20,90));response.raise_for_status()
        h=hashlib.sha256();size=0
        with (root/name).open('xb') as stream:
            for block in response.iter_content(1024*1024):
                size+=len(block)
                if size>150*1024**2:raise ValueError('Unexpected file size')
                stream.write(block);h.update(block)
        hashes[name]=h.hexdigest();sizes[name]=size
        print(json.dumps(dict(file=name,bytes=size,sha256=hashes[name])),flush=True)
    receipt=dict(model_id=model,revision=revision,license='Apache-2.0',sha256=hashes,bytes=sizes,remote_code=False,
                 source_url='https://huggingface.co/facebook/dinov2-small',official_project='https://github.com/facebookresearch/dinov2')
    (root/'download_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    print('Local DINO model: '+str(root),flush=True)


if __name__=='__main__':main()
