"""Expand TRAIN coverage only; preserve the previous50 validation images."""
from collections import Counter,defaultdict
from copy import deepcopy
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import random
import shutil
import sys
from uuid import uuid4
import zipfile
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import prepare_unibo_vlm_facies as prep
from scripts import train_unibo_visual_facies as source

SHAPES=((96,96),(128,64),(192,96),(128,128),(256,128))


def windows(mask):
    """Exact palette, >=98% target/foreground, <=0.5% ignored pixels."""
    def integral(values): return np.pad(values.astype(np.int64).cumsum(0).cumsum(1),((1,0),(1,0)))
    ids=[int(x) for x in np.unique(mask) if 1<=x<=6]
    counts={i:integral(mask==i) for i in ids}
    fg=integral((mask>=1)&(mask<=6)); ignored=integral(mask==255)
    for w,h in SHAPES:
        for y in prep.offsets(mask.shape[0],h):
            for x in prep.offsets(mask.shape[1],w):
                def area(table): return int(table[y+h,x+w]-table[y,x+w]-table[y+h,x]+table[y,x])/(w*h)
                quality={'foreground_fraction':area(fg),'ignore_fraction':area(ignored)}
                if quality['foreground_fraction']<.98 or quality['ignore_fraction']>.005: continue
                for i in ids:
                    quality['target_area_fraction']=area(counts[i])
                    if prep.eligible_quality(quality):
                        yield i,(x,y,x+w,y+h),dict(quality)
                        break


def overlap_min(a,b):
    intersection=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
    return intersection/min((a[2]-a[0])*(a[3]-a[1]),(b[2]-b[0])*(b[3]-b[1]))


def select_additions(candidates,old,limit=128):
    rng=random.Random(20261030)
    used=defaultdict(list); photos=Counter(); groups=Counter(); totals=Counter()
    for r in old:
        m=r['metadata']; c=m['mask_id']; p=m['source_sha256']; g=m['source_group']
        used[p].append(m['resized_source_crop_box']); photos[c,p]+=1; groups[c,g]+=1; totals[c]+=1
    selected=[]
    def accept(c,item):
        row=item['row']; p=row['source_sha256']; g=row['well_group']; box=item['resized_source_box']
        if totals[c]>=limit or photos[c,p]>=16 or groups[c,g]>=32: return False
        if any(prep.box_iou(box,b)>.1 or overlap_min(box,b)>.2 for b in used[p]): return False
        used[p].append(box); photos[c,p]+=1; groups[c,g]+=1; totals[c]+=1; selected.append((c,item)); return True
    for c,*_ in prep.CLASSES:
        by_photo=defaultdict(list)
        pool=list(candidates['train',c]); rng.shuffle(pool)
        for item in pool: by_photo[item['row']['source_sha256']].append(item)
        # Cover source photos not represented in this class before filling.
        for photo in sorted(by_photo,key=lambda p:(photos[c,p]>0,p)):
            if photos[c,photo]: continue
            while by_photo[photo]:
                if accept(c,by_photo[photo].pop()): break
        while by_photo and totals[c]<limit:
            progress=False
            for photo in sorted(list(by_photo),key=lambda p:(groups[c,by_photo[p][0]['row']['well_group']] if by_photo[p] else 999,photos[c,p],p)):
                while by_photo[photo]:
                    if accept(c,by_photo[photo].pop()): progress=True; break
                if not by_photo[photo]: del by_photo[photo]
                if totals[c]>=limit: break
            if not progress: break
    return selected


def prepare(old_data,prepared,output):
    old_data,prepared,output=(Path(x).resolve() for x in (old_data,prepared,output))
    if output.exists(): raise ValueError('Choose a new expanded dataset directory')
    old=source.read_rows(old_data/'train.jsonl'); val=source.read_rows(old_data/'val.jsonl')
    source.validate_facies_split(old,val)
    all_meta=source.read_rows(prepared/'manifest.jsonl'); assignments=prep.validate_source_groups(all_meta)
    train_meta=[r for r in all_meta if r['split']=='train']
    candidates,mask_audit=prep.collect_candidates(prepared,train_meta,windows)
    selected=select_additions(candidates,old)
    stage=output.with_name(output.name+'.stage_'+uuid4().hex[:8]); stage.mkdir(parents=True)
    records=deepcopy(old); current_sha=None; original=None; tile_key=None; tile=None; verified=[]
    for c,item in sorted(selected,key=lambda pair:(pair[1]['row']['source_sha256'],pair[1]['row']['tile_xywh'],pair[1]['box'])):
        row=item['row']; sha=row['source_sha256']
        if sha!=current_sha:
            entry=row['source']
            if '/Images/' not in entry['member'].replace('\\','/'): raise ValueError('Source must be a photograph')
            with zipfile.ZipFile(entry['source_path']) as archive:
                info=archive.getinfo(entry['member'])
                if info.file_size!=entry['bytes'] or info.file_size>100*1024**2: raise ValueError('Source member size differs')
                with Image.open(BytesIO(archive.read(info))) as opened: original=opened.convert('RGB')
            if prep.pixel_digest(original)!=sha: raise ValueError('Original source pixel checksum differs')
            factor=row['resize_factor']; original=original.resize(tuple(max(2,round(v*factor)) for v in original.size),Image.Resampling.LANCZOS)
            current_sha=sha; tile_key=None; verified.append(sha)
        key=(sha,tuple(row['tile_xywh']))
        if key!=tile_key:
            x,y,w,h=row['tile_xywh']; buf=BytesIO()
            original.crop((x,y,x+w,y+h)).save(buf,format='JPEG',quality=94,subsampling=0)
            buf.seek(0)
            with Image.open(buf) as opened: tile=opened.convert('RGB')
            tile_key=key
        crop=tile.crop(item['box']); _,code,label,rgb=prep.CLASSES[c-1]
        identity=sha256(json.dumps([sha,item['resized_source_box'],code]).encode()).hexdigest()[:24]
        relative=f'images/train/{code}_{identity}.png'; path=stage/relative; path.parent.mkdir(parents=True,exist_ok=True); crop.save(path)
        meta={'record_id':'unibo_facies_'+identity,'record_type':'source_labeled_facies_crop','source_id':'unibo7308',
            'source_url':prep.SOURCE_URL,'source_license':prep.LICENSE,'usage_scope':'noncommercial_research_only_no_automatic_deployment',
            'label_provenance':'source_direct_mask_exact_palette','source_class_code':code,'source_class_display_ru':label,
            'mask_id':c,'mask_rgb':list(rgb),'training_approved':True,'approval_scope':'source_mask_research_crop_not_expert_certification',
            'geology_approved':False,'split':'train','source_group':row['well_group'],'source_group_definition':'S-prefix proxy, not verified borehole ID',
            'source_sha256':sha,'source_original_split':row['source_split'],'source_archive_member':row['source'],
            'annotation_archive_member':row['annotation'],'original_annotation_sha256':row['annotation_sha256'],
            'relative_image':relative,'image_sha256':source.digest(path),'pixel_sha256':prep.pixel_digest(crop),'image_dimensions_px':list(crop.size),
            'tile_xywh':row['tile_xywh'],'tile_crop_box':list(item['box']),'resized_source_crop_box':list(item['resized_source_box']),
            'resize_factor':row['resize_factor'],'quality':item['quality'],'depths_available':False,
            'rendering':'source original resized, tile JPEG q94 subsampling0, then PNG focal crop'}
        records.append({'image':str(output/relative),'conversations':[{'from':'human','value':'<image>\n'+prep.PROMPT},
            {'from':'gpt','value':json.dumps({'facies_name':label},ensure_ascii=False)}],'metadata':meta})
    prep.assert_crop_separation(records+val)
    source.write_rows(stage/'train.jsonl',records); shutil.copy2(old_data/'val.jsonl',stage/'val.jsonl')
    summary={'status':'complete','source_license':prep.LICENSE,'research_only':True,'prepared_test_pixels_opened':0,
        'old_train_crops':len(old),'new_train_crops':len(records),'added_crops':len(records)-len(old),
        'train_class_counts':dict(Counter(r['metadata']['source_class_code'] for r in records)),
        'old_source_photos':len({r['metadata']['source_sha256'] for r in old}),
        'new_source_photos':len({r['metadata']['source_sha256'] for r in records}),
        'old_source_groups':sorted({r['metadata']['source_group'] for r in old}),
        'new_source_groups':sorted({r['metadata']['source_group'] for r in records}),
        'class_group_support':{code:sorted({r['metadata']['source_group'] for r in records if r['metadata']['source_class_code']==code}) for _,code,*_ in prep.CLASSES},
        'validation_manifest_sha256':source.digest(old_data/'val.jsonl'),'validation_images':len(val),
        'train_only_shapes':list(SHAPES),'target_and_foreground_min':.98,'ignore_max':.005,'overlap_min_area_max':.2,
        'per_photo_class_cap':16,'per_group_class_cap':32,'target_class_limit':128,
        'original_source_photos_verified':verified,'mask_audit':mask_audit,
        'script_sha256':source.digest(__file__),'source_metadata_sha256':source.digest(prepared/'manifest.jsonl'),
        'notes':['No validation/test examples moved into train','Publisher Test regrouped earlier, not a sealed test',
                 'Additional crops remain correlated within source photos/groups','S-groups are not verified borehole IDs']}
    source.write_json(stage/'summary.json',summary)
    shutil.copy2(__file__,stage/'preparation_script_snapshot.py'); shutil.copy2(old_data/'LICENSE_SOURCES.md',stage/'LICENSE_SOURCES.md')
    stage.rename(output)
    source.validate_facies_split(source.read_rows(output/'train.jsonl'),val)
    return summary


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--old-data',type=Path,default=ROOT.parent/'vlm_core_v1/data/unibo_vlm_facies_20261030')
    p.add_argument('--prepared',type=Path,default=ROOT/'open_datasets/unibo/prepared/unibo_v1')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(); r=prepare(a.old_data,a.prepared,a.output)
    print(json.dumps({k:r[k] for k in ('new_train_crops','added_crops','old_source_photos','new_source_photos','train_class_counts')}),flush=True)
