"""Rebuild every context from its assigned source photograph, without annotation masks."""
from collections import defaultdict
from io import BytesIO
import json
from pathlib import Path
import zipfile
from PIL import Image
from scripts.prepare_paired_context_v16 import rows,pixels,digest

ROOT=Path(__file__).resolve().parents[1]


def main():
 data=ROOT.parent/'vlm_core_v1/data/paired_context_20261031_v16_along_core';summary=json.loads((data/'summary.json').read_text(encoding='utf-8'))
 if summary['do_not_train']:raise ValueError('Rejected context')
 focal={}
 for s in ('train','val'):
  for r in rows(ROOT.parent/f'vlm_core_v1/data/unibo_facies_expanded_20261030_v2/{s}.jsonl'):focal[r['metadata']['record_id']]=r['metadata']
 grouped=defaultdict(list);count=0;fallback=0;original_records={}
 for split in ('train','val','external'):
  rr=rows(data/(split+'.jsonl'));original_records[split]=rr
  for r in rr:
   if digest(r['image'])!=r['image_sha256'] or digest(r['focal_image'])!=r['focal_image_sha256']:raise ValueError('Prepared/focal changed')
   if r['record_id'] in focal:grouped[focal[r['record_id']]['source_sha256']].append((split,r,focal[r['record_id']]))
   elif r['context_method']=='focal_fallback_unreviewed_external_context':
    with Image.open(r['focal_image']) as im:expected=im.convert('RGB')
    with Image.open(r['image']) as im:actual=im.convert('RGB')
    if pixels(expected)!=pixels(actual):raise ValueError('Fallback changed focal image')
    fallback+=1;count+=1
   else:
    raw=ROOT/'open_datasets/new_core_20261031_v10/raw'/r['source_file']
    if digest(raw)!=r['source_file_sha256']:raise ValueError('External source changed')
    with Image.open(raw) as im:
     im=im.convert('RGB');x0,y0,x1,y1=r['box'];h=y1-y0;low,high=(105,1283) if r['source_id']=='bohai' else (0,830) if r['source_id']=='gsj' else (0,im.height)
     box=[x0,max(low,y0-h),x1,min(high,y1+h)];expected=im.crop(box)
    if box!=r['context_provenance']['box'] or pixels(expected)!=r['context_pixel_sha256']:raise ValueError('Vertical context geometry/pixels differ')
    count+=1
 for h,rr in grouped.items():
  if len({s for s,r,m in rr})!=1:raise ValueError('Source photo crosses train/control')
  entry=rr[0][2]['source_archive_member']
  if '/Images/' not in entry['member'].replace('\\','/'):raise ValueError('Annotation requested instead of photograph')
  with zipfile.ZipFile(entry['source_path']) as z:
   info=z.getinfo(entry['member'])
   if info.file_size!=entry['bytes'] or info.file_size>100*1024**2:raise ValueError('Member changed')
   with Image.open(BytesIO(z.read(info))) as im:original=im.convert('RGB')
  if pixels(original)!=h:raise ValueError('Source pixel hash differs')
  factor=rr[0][2]['resize_factor'];im=original.resize(tuple(max(2,round(v*factor)) for v in original.size),Image.Resampling.LANCZOS)
  for split,r,m in rr:
   x0,y0,x1,y1=m['resized_source_crop_box'];w=x1-x0;box=[max(0,x0-w),y0,min(im.width,x1+w),y1]
   if box!=r['context_provenance']['box'] or pixels(im.crop(box))!=r['context_pixel_sha256']:raise ValueError('Horizontal along-core context differs')
   count+=1
 for r in [r for rr in original_records.values() for r in rr]:
  with Image.open(r['image']) as im:
   if pixels(im.convert('RGB'))!=r['context_pixel_sha256']:raise ValueError('Saved context pixels differ')
 if count!=888 or fallback!=100:raise ValueError('Record coverage differs')
 audit=dict(status='passed',records_reconstructed=count,focal_fallbacks=fallback,larger_contexts_reconstructed=count-fallback,known_assigned_source_photos_reopened=len(grouped),annotation_masks_read=0,
  independent_geometry_reconstruction=True,no_class_text_drawn=True,source_photos_disjoint_between_train_and_control=True,closed_project_test_pixels_opened=0,
  new_independent_photos=0,distribution_approved=False,script_sha256=digest(__file__),limits='Context may contain neighbouring facies or tray gaps; target label remains the focal crop. Original publisher Test was previously regrouped, not newly treated as sealed.')
 (data/'audit_result.json').write_text(json.dumps(audit,indent=2)+'\n',encoding='utf-8');print(json.dumps(audit,indent=2),flush=True)


if __name__=='__main__':main()
