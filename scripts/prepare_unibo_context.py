"""Paired views of the SAME source photo, with the original focal label.

No facies masks or class-specific colours are passed to the model. Only
train/val source photos are opened; source S-group test stays unopened.
"""
from collections import Counter, defaultdict
from copy import deepcopy
from io import BytesIO
import json
from pathlib import Path
import shutil
import sys
from uuid import uuid4
import zipfile

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import train_unibo_visual_facies as source
from scripts.prepare_unibo_vlm_facies import pixel_digest

CONTEXT_PROMPT = (
    "На одном изображении показаны два вида ОДНОГО участка керна. "
    "Слева — контекст исходной фотографии; голубой прямоугольник отмечает целевой участок. "
    "Справа — увеличенный вид только этого участка. Определи фацию ЦЕЛЕВОГО участка справа, "
    "учитывая видимые признаки и контекст слева. " + source.PROMPT
)
CONTEXT_SIDE = 384
FOCAL_SIDE = 256
CANVAS_SIZE = (640, 384)
TARGET_BOX = (384, 64, 640, 320)
NEUTRAL = (128, 128, 128)


def context_panels(original, focal, box):
    """Fixed neutral layout/marker for every class; no source-mask pixels."""
    x0, y0, x1, y1 = box
    if not (0 <= x0 < x1 <= original.width and 0 <= y0 < y1 <= original.height):
        raise ValueError("Focal coordinates outside original photo")
    if (x1-x0, y1-y0) != focal.size:
        raise ValueError("Focal dimensions disagree with source coordinates")
    left = max(0, min(round((x0+x1)/2-256), max(0, original.width-512)))
    top = max(0, min(round((y0+y1)/2-256), max(0, original.height-512)))
    region = original.crop((left, top, min(left+512, original.width), min(top+512, original.height)))
    ox, oy = (512-region.width)//2, (512-region.height)//2
    padded = Image.new("RGB", (512, 512), NEUTRAL)
    padded.paste(region, (ox, oy))
    marked = padded.resize((CONTEXT_SIDE, CONTEXT_SIDE), Image.Resampling.LANCZOS)
    marker = tuple(round(v*CONTEXT_SIDE/512) for v in (x0-left+ox, y0-top+oy, x1-left+ox, y1-top+oy))
    ImageDraw.Draw(marked).rectangle(marker, outline=(0, 190, 230), width=2)
    canvas = Image.new("RGB", CANVAS_SIZE, NEUTRAL)
    canvas.paste(marked, (0, 0))
    canvas.paste(focal.resize((FOCAL_SIDE, FOCAL_SIDE), Image.Resampling.LANCZOS), (384, 64))
    return canvas, {"source_context_box": [left, top, min(left+512, original.width), min(top+512, original.height)],
                    "context_padding_xy": [ox, oy], "context_marker_box": list(marker), "target_panel_box": list(TARGET_BOX)}


def blank_context(image):
    result = image.copy()
    result.paste(NEUTRAL, (0, 0, CONTEXT_SIDE, CONTEXT_SIDE))
    return result


def prepare(data, output):
    data, output = Path(data).resolve(), Path(output).resolve()
    if output.exists():
        raise ValueError("Choose a new context data directory")
    raw_train, raw_val = source.read_rows(data/'train.jsonl'), source.read_rows(data/'val.jsonl')
    audit = source.validate_facies_split(raw_train, raw_val)
    stage = output.with_name(output.name+'.stage_'+uuid4().hex[:8])
    stage.mkdir(parents=True)
    grouped = defaultdict(list)
    for row in raw_train+raw_val:
        grouped[row['metadata']['source_sha256']].append(row)
    records, source_audit = [], []
    for index, (sha, rows) in enumerate(sorted(grouped.items()), 1):
        first = rows[0]['metadata']
        entry = first['source_archive_member']
        if '/Images/' not in entry['member'].replace('\\', '/'):
            raise ValueError("Only source photographs can form the context")
        with zipfile.ZipFile(entry['source_path']) as archive:
            info = archive.getinfo(entry['member'])
            if info.file_size != entry['bytes'] or info.file_size > 100*1024**2:
                raise ValueError("Source archive member size differs")
            with Image.open(BytesIO(archive.read(info))) as opened:
                original = opened.convert('RGB')
        if pixel_digest(original) != sha:
            raise ValueError("Original photo pixel checksum differs")
        factor = first['resize_factor']
        resized = original.resize(tuple(max(2, round(v*factor)) for v in original.size), Image.Resampling.LANCZOS)
        source_audit.append({'source_sha256':sha, 'source_group':first['source_group'], 'split':first['split'],
                             'original_source':entry, 'source_pixel_hash_verified':True})
        for row in rows:
            meta = row['metadata']
            if meta['source_archive_member'] != entry or meta['resize_factor'] != factor or meta['split'] not in {'train','val'}:
                raise ValueError("Source identity or split changed")
            with Image.open(row['image']) as opened:
                focal = opened.convert('RGB')
            panel, geometry = context_panels(resized, focal, meta['resized_source_crop_box'])
            stem = source.digest(row['image'])[:24]
            relative = f"images/{meta['split']}/{meta['source_class_code']}_{stem}.png"
            target = stage/relative
            target.parent.mkdir(parents=True, exist_ok=True)
            panel.save(target)
            new = source.transform_facies(row)
            new['image'] = str(output/relative)
            new['conversations'][0]['value'] = '<image>\n'+CONTEXT_PROMPT
            new['metadata'].update({'record_id':'context_'+meta['record_id'], 'focal_record_id':meta['record_id'],
                'focal_image':row['image'], 'focal_image_sha256':meta['image_sha256'],
                'focal_pixel_sha256':meta['pixel_sha256'], 'input_variant':'same_photo_context_and_focal_panel',
                'quality_scope':'focal_crop_only_not_context', 'image_sha256':source.digest(target),
                'pixel_sha256':pixel_digest(panel), 'image_dimensions_px':list(CANVAS_SIZE), 'panel_geometry':geometry,
                'relative_image':relative, 'class_text_drawn_on_input':False, 'source_mask_passed_to_model':False})
            records.append(new)
        print(f'context source {index}/{len(grouped)}',flush=True)
    if len({r['metadata']['pixel_sha256'] for r in records}) != len(records):
        raise ValueError('Duplicate context pixels')
    for field in ('source_group','source_sha256','pixel_sha256','focal_record_id'):
        a={r['metadata'][field] for r in records if r['metadata']['split']=='train'}
        b={r['metadata'][field] for r in records if r['metadata']['split']=='val'}
        if a & b: raise ValueError('Context train/val leakage: '+field)
    for split in ('train','val'):
        chosen=[r for r in records if r['metadata']['split']==split]
        source.write_rows(stage/(split+'.jsonl'),chosen)
    summary={'status':'complete','source_url':'https://amsacta.unibo.it/id/eprint/7308/',
        'source_license':'CC BY-NC-SA 4.0','research_only':True,'prepared_test_pixels_opened':0,
        'new_independent_photos':0,'counts':{s:dict(Counter(r['metadata']['source_class_code'] for r in records if r['metadata']['split']==s)) for s in ('train','val')},
        'canvas_size':list(CANVAS_SIZE),'source_split_audit':audit,'source_photos':source_audit,
        'focal_manifests':{str(p):source.digest(p) for p in (data/'train.jsonl',data/'val.jsonl')},
        'preparation_script_sha256':source.digest(__file__),
        'notes':['Target labels apply to right focal panel; left context can contain other facies and tray background',
                 'Source mask colours and class captions are never model input',
                 'S groups are proxies, not verified borehole IDs; val is development',
                 'Publisher original Test is not sealed after prior regrouping',
                 'Interpolation does not add real image detail or independent examples']}
    source.write_json(stage/'summary.json',summary)
    shutil.copy2(__file__,stage/'preparation_script_snapshot.py')
    shutil.copy2(data/'LICENSE_SOURCES.md',stage/'LICENSE_SOURCES.md')
    # Preview labels belong to this contact sheet only, never training panels.
    preview=Image.new('RGB',(1280,6*414),'white')
    draw=ImageDraw.Draw(preview)
    for i,code in enumerate(source.CODES):
        for j,split in enumerate(('train','val')):
            row=next(r for r in records if r['metadata']['split']==split and r['conversations'][1]['value']==code)
            with Image.open(stage/row['metadata']['relative_image']) as im: preview.paste(im,(j*640,i*414))
            draw.text((j*640+5,i*414+388),f'{code} {split} {row["metadata"]["source_group"]}',fill='black')
    preview.save(stage/'contact_sheet.png')
    stage.rename(output)
    return summary


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,default=ROOT.parent/'vlm_core_v1/data/unibo_vlm_facies_20261030')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(prepare(args.data,args.output)['counts']),flush=True)
