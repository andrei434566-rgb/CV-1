"""Read-only inventory of source labels for the non-generative CV-1 engine."""
from pathlib import Path
import json, collections

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT.parent/'vlm_core_v1/data'
def rows(p):
    return [json.loads(s) for s in p.read_text(encoding='utf-8-sig').splitlines() if s.strip()]
def main():
    names=['mendeley_lithology_4000/train.jsonl','mendeley_lithology_next/train.jsonl',
           'mendeley_lithology_40000/additional_train.jsonl',
           'mendeley_lithology_pretrained_holdout/val.jsonl','mendeley_lithology_pretrained_holdout/test.jsonl',
           'e_drive_interval_crops_v2/qwen3_vl_interval_crops.jsonl',
           'core_facies_20261031_v6_clean/train.jsonl']
    for name in names:
        records=rows(DATA/name)
        print(json.dumps({'manifest':name,'rows':len(records),'example':records[0] if 'mendeley' not in name else records[0].get('metadata'),
                          'wells':dict(collections.Counter(r.get('metadata',{}).get('well_id','?') for r in records)),
                          'labels':dict(collections.Counter(r.get('metadata',{}).get('source_label_en','?') for r in records))},ensure_ascii=False),flush=True)
    for p in sorted((ROOT/'projects').rglob('project.json')):
        d=json.loads(p.read_text(encoding='utf-8-sig'))
        photos=d.get('photos',d.get('images',[]))
        dets=[x for r in photos for x in r.get('detections',r.get('annotations',[]))] if isinstance(photos,list) else []
        print(json.dumps({'project':str(p.relative_to(ROOT)),'keys':list(d),'photos':len(photos),
                          'photo_example':{k:v for k,v in photos[0].items() if k not in ['detections','annotations']} if photos else None,
                          'detections':len(dets),'labels':dict(collections.Counter(str(x.get('label',x.get('class_name',x.get('category_id')))) for x in dets)),
                          'detection_example':dets[0] if dets else None},ensure_ascii=False),flush=True)
if __name__=='__main__':main()
