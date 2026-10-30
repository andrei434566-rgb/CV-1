"""Download public original papers and prepare reviewed, source-grounded crops.

Requires pypdf/Pillow, not the training environment. Outcrop is explicitly
separate from core; coordinates exclude annotations and mixed facies.
"""
import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import urllib.request
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    'chengjiang': dict(author='Saleh et al. (2022)', doi='10.1038/s41467-022-29246-z',
        url='https://www.nature.com/articles/s41467-022-29246-z',
        pdf='https://www.nature.com/articles/s41467-022-29246-z.pdf',
        sha256='d424de10bde1f729d56b23fa94053e49806a3f0093bd0a1117242e962d4b515e',
        license='CC BY 4.0', license_basis='PDF page9, applies to figures unless otherwise credited; no separate credit on Fig2',
        source_group='WEB_Chengjiang_Yuanshan', domain='core', figure='2i,2j; page3',
        description_ru='Массивный аргиллит с местной слабой ламинацией, отложения пробковых потоков и малоплотностных турбидитных течений продельты. Биотурбированные гемипелагические прослои относятся к шельфу и исключены из выбранных участков.',
        label_basis='Fig2i,j caption plus Fig3/environmental interpretation; only Mm beds selected, not dark organic/bioturbated interbeds',
        interval='Figure-specific depth unavailable; Cambrian Yuanshan Formation',
        limitations='Ancient lithified core, not a Holocene mud; several patches from two panels of one paper'),
    'nakdong': dict(author='Jeong et al. (2022; published2021-12-24)', doi='10.3390/app12010177',
        url='https://www.mdpi.com/2076-3417/12/1/177',
        pdf='https://mdpi-res.com/d_attachment/applsci/applsci-12-00177/article_deploy/applsci-12-00177.pdf',
        sha256='a3b3f9c7cfda18cc726539322e438988b42e286655523087fec6536d79aa1cef',
        license='CC BY 4.0', license_basis='PDF page1 copyright and CC BY4 link',
        source_group='WEB_Nakdong_ND3', domain='core', figure='2e; page5',
        description_ru='Преимущественно массивные тонкозернистые илистые отложения продельты, с морскими микрофоссилиями и обломками раковин; по разрезу слабое укрупнение вверх. Отдельная фотография показывает часть керна около 17 м.',
        label_basis='Fig2e UnitD 17.00-17.78m; abstract and section5.1.4 assign UnitD 23.50-15.10m to prodelta',
        interval='Core ND-3, photographed section17.00-17.78m within prodelta UnitD23.50-15.10m; no precise depth assigned to individual patches',
        limitations='Native photo width about110px; no physical grain size inferred from pixels'),
    'boss_point': dict(author='Rygel et al. (2015)', doi='10.4138/atlgeol.2015.001',
        url='https://journals.lib.unb.ca/index.php/ag/article/view/21439',
        pdf='https://journals.lib.unb.ca/index.php/ag/article/download/21439/26308/34252',
        sha256='9551811e7a1f3be3b3d0001f35dd76193faa6e72aa179cb223fcb2026549d02f',
        license='CC BY 4.0 stated on current article page, checked2026-10-31',
        license_basis='Current article License box: journal adopted CC BY4 on2025-01-01; old PDF retains2015 copyright footer. This date distinction is retained, not silently rewritten.',
        source_group='WEB_BossPoint_type_section', domain='outcrop_analogue', figure='11b; page17',
        description_ru='Серые глинистые породы с блоковой отдельностью в плохо дренируемой пойменной ассоциации. Авторы связывают глинистые породы с осаждением взвеси на насыщенной водой пойме и незрелым гидроморфным почвообразованием.',
        label_basis='Fig11b explicitly grey blocky mudrocks at174m, poorly drained facies association; interpretation page16',
        interval='174m measured stratigraphic section, NOT borehole depth',
        limitations='One weathered outcrop photograph, not core; scale/surface/age domain shift. Low auxiliary training weight, no claim of new independent cores'),
}
PHOTO_FILES = {
    'chengjiang_p3_9.jpg': ('chengjiang', 2, 9, '25160fd9e4dc344cd2878a46014fcb98bdb9bc1f2799a3b0785e899503a527c3'),
    'chengjiang_p3_0.jpg': ('chengjiang', 2, 0, '146e3ecbbfe9bd6542f6ff58833a0287f9d447c42ff25c45f14b52c48bd78f42'),
    'nakdong_fig2_0.png': ('nakdong', 4, 0, '887d97781686d62bfe94be3215d1c01a589fac6af2ff98eda594b22c25246519'),
    'boss_point_p17_image1.jpg': ('boss_point', 16, 1, '75e914d474c1daf5766e173303cab394d2cbe6c283954c63f45fd4da1db66f02'),
}


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def pixel_digest(im):
    h = sha256(f'{im.width}x{im.height}:RGB:'.encode()); h.update(im.tobytes())
    return h.hexdigest()


def specifications():
    specs = []
    def add(file, panel, code, box):
        specs.append(dict(file=file, panel=panel, allowed_codes=[code], box=list(box)))
    # Extracted originals have PDF vector labels removed by extraction, not retouching.
    # Panel i: select grey Mm interiors, avoid the dark organic-rich bands.
    for x in (40, 330):
        for y in (60, 620): add('chengjiang_p3_9.jpg', '2i', 'C', (x, y, x+192, y+128))
    # Panel j: avoid the top/bottom and bioturbated bed tops shown on complete page.
    for y in (95, 230, 490): add('chengjiang_p3_0.jpg', '2j', 'C', (40, y, 290, y+100))
    # UnitD only; letter e, crack annotation and shell-fragment arrow are outside.
    for y in (230, 360, 750, 880): add('nakdong_fig2_0.png', '2e', 'C', (601, y, 687, y+120))
    # Single photo b, not photo c's crevasse sandstone. Hammer x310..414/y210..403 excluded.
    for box in ((25,60,245,240), (25,260,245,440), (25,465,245,645), (275,450,495,630)):
        add('boss_point_p17_image1.jpg', '11b', 'B', box)
    return specs


def fetch(raw):
    from pypdf import PdfReader
    raw.mkdir(parents=True, exist_ok=True)
    for sid, meta in SOURCES.items():
        path = raw/(sid+'_pdf.pdf')
        if not path.exists():
            req = urllib.request.Request(meta['pdf'], headers={'User-Agent':'CV-1 open-research source retrieval'})
            with urllib.request.urlopen(req, timeout=120) as response: data = response.read()
            if not data.startswith(b'%PDF'): raise ValueError('Not a public PDF: '+sid)
            if sha256(data).hexdigest()!=meta['sha256']: raise ValueError('Publication changed; review before extraction: '+sid)
            path.write_bytes(data)
        if digest(path)!=meta['sha256']: raise ValueError('PDF checksum differs: '+sid)
    for name,(sid,page,index,expected) in PHOTO_FILES.items():
        path=raw/name
        if not path.exists():
            embedded=PdfReader(raw/(sid+'_pdf.pdf')).pages[page].images[index]
            # Preserve original embedded encoded image, including its actual extension.
            if sha256(embedded.data).hexdigest()!=expected: raise ValueError('Extraction differs; review with pinned pypdf6.10.0: '+name)
            path.write_bytes(embedded.data)
        if digest(path)!=expected: raise ValueError('Photo checksum differs: '+name)


def prepare(raw, output):
    if output.exists(): raise ValueError('Choose a new output directory')
    photos={}
    for name,(_,_,_,expected) in PHOTO_FILES.items():
        if digest(raw/name)!=expected: raise ValueError('Photo changed: '+name)
        with Image.open(raw/name) as im: photos[name]=im.convert('RGB')
    output.mkdir(parents=True); (output/'images').mkdir()
    rows=[]; sheet=Image.new('RGB',(5*220,3*190),'#eeeeee')
    for i,spec in enumerate(specifications()):
        name=spec['file']; sid=PHOTO_FILES[name][0]; meta=SOURCES[sid]
        im=photos[name]; x0,y0,x1,y1=spec['box']
        if not (0<=x0<x1<=im.width and 0<=y0<y1<=im.height): raise ValueError('Outside photo')
        crop=im.crop(spec['box']); rid='targetfacies_'+sha256(json.dumps(spec,sort_keys=True).encode()).hexdigest()[:24]
        path=output/'images'/(rid+'.png'); crop.save(path)
        rows.append(dict(record_id=rid,image=str(path.resolve()),image_sha256=digest(path),pixel_sha256=pixel_digest(crop),
            image_size_px=list(crop.size),source_id=sid,source=meta,source_group=meta['source_group'],domain=meta['domain'],
            source_file=name,source_file_sha256=PHOTO_FILES[name][3],source_panel=spec['panel'],source_box=spec['box'],
            allowed_codes=spec['allowed_codes'],exact_code=spec['allowed_codes'][0],split='train',external=True,
            training_approved=True,geologist_approved=False,project_92_facies_mapping=None,
            preprocessing='Unmodified RGB source crop; no recolouring, upscaling, inpainting or invented detail',
            review_scope='Source caption and whole-page localization reviewed; not expert project facies approval'))
        tile=crop.copy(); tile.thumbnail((205,150)); pos=((i%5)*220,(i//5)*190)
        sheet.paste(tile,(pos[0]+5,pos[1]+5))
        ImageDraw.Draw(sheet).text((pos[0]+5,pos[1]+158),f'{i+1} {sid} {spec["panel"]} {spec["allowed_codes"][0]}',fill='black')
    (output/'train.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
    summary=dict(crops=len(rows),exact_class_counts=dict(Counter(r['exact_code'] for r in rows)),
        photographic_panels=4,core_panels=3,outcrop_panels=1,sources=SOURCES,source_photos=PHOTO_FILES,
        prepared_test_pixels_opened=0,validation_examples_moved=0,geologist_approved=False,
        do_not_train=False,script_sha256=digest(__file__),
        limitations=['15 correlated patches from4 panels and3 paper/source groups, not15 independent core photos',
            'B has only one new outcrop analogue; no newly downloaded exact B core',
            'Old development50 remains reused, not an independent final accuracy estimate'])
    (output/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    sheet.save(output/'contact_sheet.png')
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw',type=Path,default=ROOT/'open_datasets/target_facies_20261031_v4/raw')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--fetch',action='store_true')
    a=p.parse_args()
    if a.fetch: fetch(a.raw.resolve())
    print(json.dumps(prepare(a.raw.resolve(),a.output.resolve())['exact_class_counts']))
