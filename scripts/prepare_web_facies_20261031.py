"""Prepare manually inspected source-caption crops; retain partial labels.

No masks inferred from colour, no imputed drainage labels, no web scraping
at training time. Download the public source files separately first.
"""
import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import shutil
from uuid import uuid4
from PIL import Image,ImageDraw

ROOT=Path(__file__).resolve().parents[1]
SOURCE_SHA={
    'sydney_p16_image0.jpg':'791263b5e4fa4ff0f624dce8359efe5e5309ef14b10f6be8151a27c20d773260',
    'sydney_p16_image1.jpg':'ed1f86558290d7bc5d3324e84ab7f554ddd5da42dfd156bdb4755be8020a9c97',
    'fujian_fig3.webp':'ac4ca0cbd27423160013cedfa1b78bc634692b3b5cd7e25df27f2ec65d4aeedc',
}
SOURCES={
    'sydney2021':{'doi':'10.3390/geosciences11070273','url':'https://www.mdpi.com/2076-3263/11/7/273',
        'author':'Melehan et al. (2021)','license':'CC BY 4.0','figure':'Figure 8, page 16, panels C and E',
        'label_basis':'Authors explicitly interpret FA7 as prodelta. Panel C: sand/mud heterolithics; panel E: syneresis cracks.',
        'source_group':'WEB_Sydney_FA7_unknown_wells','group_note':'Entire paper grouped together; photo-specific well IDs not supplied in caption.',
        'scope_note':'Lithified Permian prodelta, unlike Holocene UniBo mud. Environmental association, not one lithology.'},
    'fujian2026':{'doi':'10.3389/fmars.2026.1909600','url':'https://www.frontiersin.org/journals/marine-science/articles/10.3389/fmars.2026.1909600/full',
        'author':'Chen et al. (2026)','license':'CC BY 4.0','figure':'Figure 3, core LJ01, columns 7 and 8',
        'label_basis':'Column 7 explicitly labelled Floodplain, column 8 Fluvial channel; section4.1.2 permits ponding/lacustrine influence.',
        'source_group':'WEB_Fujian_LJ01','group_note':'All crops from one borehole held together.',
        'scope_note':'Floodplain has no drainage subclass label; admissible source codes A or B, never either as a hard target.'},
}


def digest(path): return sha256(Path(path).read_bytes()).hexdigest()


def pixel_digest(im):
    h=sha256(f'{im.width}x{im.height}:RGB:'.encode()); h.update(im.tobytes()); return h.hexdigest()


def specifications():
    rows=[]
    def add(filename,source,panel,allowed,box):
        rows.append({'file':filename,'source':source,'panel':panel,'allowed_codes':allowed,'box':list(box)})
    # Coordinates were checked against the complete page and extracted originals.
    # C-panel right-side F5/F6/F7 labels begin beyond x1230; exclude them.
    for x in (990,1120):
        for y in (60,196,332,468): add('sydney_p16_image0.jpg','sydney2021','8C',['C'],(x,y,x+128,y+128))
    # E-panel bottom arrows/caption and top panel letter are outside these boxes.
    for x in (980,1110):
        for y in (165,295): add('sydney_p16_image1.jpg','sydney2021','8E',['C'],(x,y,x+128,y+128))
    # Figure3: left curves, green borders, rulers and vertical in-photo labels excluded.
    for y in (20,156,292,875,1011,1147): add('fujian_fig3.webp','fujian2026','3-column7',['A','B'],(2155,y,2230,y+128))
    for y in (20,156,292,428,564,1120): add('fujian_fig3.webp','fujian2026','3-column8',['E'],(2330,y,2415,y+128))
    return rows


def prepare(raw,output):
    raw,output=Path(raw).resolve(),Path(output).resolve()
    if output.exists(): raise ValueError('Choose a new output directory')
    images={}
    for filename,expected in SOURCE_SHA.items():
        if digest(raw/filename)!=expected: raise ValueError('Source bytes differ: '+filename)
        with Image.open(raw/filename) as im: images[filename]=im.convert('RGB')
    stage=output.with_name(output.name+'.stage_'+uuid4().hex[:8]); stage.mkdir(parents=True)
    rows=[]; seen=set(); thumbs=[]
    for spec in specifications():
        im=images[spec['file']]; x0,y0,x1,y1=spec['box']
        if not (0<=x0<x1<=im.width and 0<=y0<y1<=im.height): raise ValueError('Crop outside original')
        crop=im.crop(spec['box']); pixels=pixel_digest(crop)
        if pixels in seen: raise ValueError('Duplicate crop')
        seen.add(pixels)
        rid='webfacies_'+sha256(json.dumps(spec,sort_keys=True).encode()).hexdigest()[:24]
        relative='images/'+rid+'.png'; dest=stage/relative; dest.parent.mkdir(exist_ok=True); crop.save(dest)
        record={'image':str(output/relative),'record_id':rid,'allowed_codes':spec['allowed_codes'],
            'exact_code':spec['allowed_codes'][0] if len(spec['allowed_codes'])==1 else None,
            'source':SOURCES[spec['source']],'source_id':spec['source'],'source_group':SOURCES[spec['source']]['source_group'],
            'source_file':spec['file'],'source_file_sha256':SOURCE_SHA[spec['file']],
            'source_panel':spec['panel'],'source_box':spec['box'],'image_sha256':digest(dest),'pixel_sha256':pixels,
            'image_size_px':list(crop.size),'label_provenance':'direct_publication_figure_and_text_manually_localized',
            'review_scope':'research_caption_and_crop_review_not_expert_project_facies_approval',
            'training_approved':True,'geologist_approved':False,'split':'train','depth_available':False,
            'project_92_facies_mapping':None,'preprocessing':'Unmodified RGB crop saved as PNG; no inpainting, recolouring or synthetic detail.'}
        rows.append(record)
        thumb=Image.new('RGB',(170,175),'white'); tile=crop.copy(); tile.thumbnail((145,140))
        thumb.paste(tile,((170-tile.width)//2,4)); ImageDraw.Draw(thumb).text((4,148),spec['panel']+' '+'/'.join(spec['allowed_codes']),fill='black')
        thumbs.append(thumb)
    (stage/'train.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
    summary={'crops':len(rows),'exact_class_counts':dict(Counter(r['exact_code'] for r in rows if r['exact_code'])),
        'partial_floodplain_crops':sum(r['exact_code'] is None for r in rows),'source_groups':sorted({r['source_group'] for r in rows}),
        'source_photo_panels':4,'sources':SOURCES,'source_file_sha256':SOURCE_SHA,'script_sha256':digest(__file__),
        'prepared_test_pixels_opened':0,'validation_examples_moved':0,'geologist_approved':False,
        'limitations':['24 crops from only4 panels and2 source groups, not24 independent core photos',
            'Sydney provenance differs in age, consolidation and imaging from UniBo',
            'Figure crops have limited native resolution; no physical grain size inferred',
            'Fujian column2 offshore/prodelta excluded as an ambiguous setting; no forced prodelta label',
            'No A/B drainage subclass invented for generic floodplain examples']}
    (stage/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    sheet=Image.new('RGB',(6*170,4*175),'#dddddd')
    for i,thumb in enumerate(thumbs): sheet.paste(thumb,((i%6)*170,(i//6)*175))
    sheet.save(stage/'contact_sheet.png')
    shutil.copy2(__file__,stage/'preparation_script_snapshot.py')
    stage.rename(output)
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw',type=Path,default=ROOT/'open_datasets/facies_web_20261031/raw')
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args(); s=prepare(args.raw,args.output)
    print(json.dumps({k:s[k] for k in ('crops','exact_class_counts','partial_floodplain_crops','source_groups')}),flush=True)
