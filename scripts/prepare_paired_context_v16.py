"""Fixed vertical context from assigned source photos, never masks or class text."""
import argparse
from collections import defaultdict
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import zipfile
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
def digest(path):return sha256(Path(path).read_bytes()).hexdigest()
def pixels(im):return sha256(f'{im.width}x{im.height}:RGB:'.encode()+im.tobytes()).hexdigest()
def rows(path):return [json.loads(x) for x in path.read_text(encoding='utf-8').splitlines() if x]


def context_box(box,size,y_limits=None,axis='y'):
    x0,y0,x1,y1=box;width,height=size
    if not (0<=x0<x1<=width and 0<=y0<y1<=height):raise ValueError('Focal box outside source')
    if axis=='x':
        center=(x0+x1)/2;span=3*(x1-x0)
        return [max(0,round(center-span/2)),y0,min(width,round(center+span/2)),y1]
    if axis!='y':raise ValueError('Only x/y axes supported')
    low,high=y_limits or (0,height);center=(y0+y1)/2;span=3*(y1-y0)
    return [x0,max(low,round(center-span/2)),x1,min(high,round(center+span/2))]


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output.resolve()
 if out.exists() or not out.is_relative_to(ROOT.parent/'vlm_core_v1/data'):raise ValueError('Choose new local data folder')
 source=ROOT.parent/'vlm_core_v1/outputs/core_qwen_new_well_facies_20261031_v10';rr={s:rows(source/f) for s,f in [('train','train.jsonl'),('val','val.jsonl'),('external','external_holdout.jsonl')]}
 meta={}
 for name in ('train','val'):
  for r in rows(ROOT.parent/f'vlm_core_v1/data/unibo_facies_expanded_20261030_v2/{name}.jsonl'):meta[r['metadata']['record_id']]=r['metadata']
 out.mkdir(parents=True);(out/'images').mkdir();(out/'summary.json').write_text(json.dumps(dict(do_not_train=True,reason='Incomplete preparation')),encoding='utf-8')
 grouped=defaultdict(list);prepared={s:[] for s in rr};audit=[]
 def save(r,split,context,method,info):
  path=out/'images'/f"{split}_{r['record_id']}.png";context.save(path)
  item=dict(r,focal_image=r['image'],focal_image_sha256=r.get('image_sha256',r.get('metadata',{}).get('image_sha256')),image=str(path),image_sha256=digest(path),context_method=method,context_provenance=info,
            context_pixel_sha256=pixels(context),no_annotation_mask_input=True,context_can_contain_other_facies=True)
  prepared[split].append(item)
 for split,records in rr.items():
  for r in records:
   key=r.get('record_id',r.get('metadata',{}).get('record_id'))
   if 'record_id' not in r:r=dict(r,record_id=key)
   if key in meta:grouped[meta[key]['source_sha256']].append((split,r,meta[key]))
   elif key.startswith('newwell_v10_'):
    raw=ROOT/'open_datasets/new_core_20261031_v10/raw'/r['source_file']
    if digest(raw)!=r['source_file_sha256']:raise ValueError('Source photo changed')
    with Image.open(raw) as im:
     im=im.convert('RGB');limits=(105,1283) if r['source_id']=='bohai' else (0,830) if r['source_id']=='gsj' else None
     box=context_box(r['box'],im.size,limits);save(r,split,im.crop(box),'fixed_3x_vertical',dict(source_file=r['source_file'],source_sha256=digest(raw),box=box))
   else:
    with Image.open(r['image']) as im:save(r,split,im.convert('RGB'),'focal_fallback_unreviewed_external_context',dict(reason='No verified larger photo geometry; no guessed labels/depths'))
 for i,(h,records) in enumerate(sorted(grouped.items())):
  splits={x[0] for x in records}
  if len(splits)!=1:raise ValueError('Source photo crosses train/control')
  first=records[0][2];entry=first['source_archive_member']
  if '/Images/' not in entry['member'].replace('\\','/'):raise ValueError('Only original photograph allowed')
  with zipfile.ZipFile(entry['source_path']) as z:
   info=z.getinfo(entry['member'])
   if info.file_size!=entry['bytes'] or info.file_size>100*1024**2:raise ValueError('Source member changed')
   with Image.open(BytesIO(z.read(info))) as im:original=im.convert('RGB')
  if pixels(original)!=h:raise ValueError('Source photograph pixel hash differs')
  factor=first['resize_factor'];resized=original.resize(tuple(max(2,round(v*factor)) for v in original.size),Image.Resampling.LANCZOS)
  for split,r,m in records:
   if m['source_archive_member']!=entry or m['resize_factor']!=factor:raise ValueError('Source geometry differs')
   box=context_box(m['resized_source_crop_box'],resized.size,axis='x')
   save(r,split,resized.crop(box),'fixed_3x_along_horizontal_core',dict(source_sha256=h,source_group=r.get('source_group',m['source_group']),archive_member=entry['member'],resize_factor=factor,box=box,axis='x'))
  audit.append(dict(source_sha256=h,split=next(iter(splits)),original_member=entry['member'],known_assigned_photo_only=True))
  print(json.dumps(dict(event='context_source',done=i+1,total=len(grouped))),flush=True)
 for split,records in rr.items():
  indexed={r['record_id']:r for r in prepared[split]};prepared[split]=[indexed[r.get('record_id',r.get('metadata',{}).get('record_id'))] for r in records]
  (out/(split+'.jsonl')).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in prepared[split]),encoding='utf-8')
 summary=dict(do_not_train=False,counts={s:len(v) for s,v in prepared.items()},context_counts={s:sum(r['context_method'].startswith('fixed_3x') for r in v) for s,v in prepared.items()},
  context_is_label_independent=True,masks_read=0,source_photos=audit,source_manifests_sha256={s:digest(source/f) for s,f in [('train','train.jsonl'),('val','val.jsonl'),('external','external_holdout.jsonl')]},
  closed_project_test_pixels_opened=0,new_independent_photos=0,distribution_approved=False,limitation='Same assigned source photos, fixed geometric neighbours may include other facies and tray gaps. Source original Test members were previously regrouped into train/val; not a new sealed test.')
 (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(json.dumps({k:v for k,v in summary.items() if k!='source_photos'},indent=2),flush=True)


if __name__=='__main__':main()
