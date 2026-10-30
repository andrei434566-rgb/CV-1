"""Prepare inspected CORE photos only; retain unresolved distribution rights.

Salpi is a public author copy/public publisher figure, but no explicit CC
grant was verified. This local-research set must not be exported as CC data.
"""
import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import urllib.request
from PIL import Image,ImageDraw

ROOT=Path(__file__).resolve().parents[1]
FILES={
 'salpi_manuscript.pdf':('https://www.earth-prints.org/server/api/core/bitstreams/9ebde40e-07a2-486b-b023-d97d737cdff3/content','2a349d786845723cc013d02cb0ed74cfe6380db64a0aaa7ada4eb396fe2ff3fc'),
 'salpi_figure3_original.jpg':('https://ars.els-cdn.com/content/image/1-s2.0-S1040618222003391-gr3_lrg.jpg','ab51d094a3476943c16a493baec2eb57bce4145b6e7bf4eb94f8ba610fdb7bcf'),
 'gaoyou.pdf':('https://mdpi-res.com/d_attachment/applsci/applsci-15-04153/article_deploy/applsci-15-04153.pdf','8274eb5c48b6022217e30351c5321fa6b410607e03be487e7d845742cc9578f5'),
 'south_china_sea.pdf':('https://www.nature.com/articles/s41598-023-40721-5.pdf','d73d8ffb67da10057bf2b7f76eda7f11ee6b069c80bd4e00bf1756a77f226013'),
}
PHOTOS={
 'salpi_figure3_original.jpg':'ab51d094a3476943c16a493baec2eb57bce4145b6e7bf4eb94f8ba610fdb7bcf',
 'gaoyou_figure3_photos.png':'7e0a7adfe8661c123893fa3f9ab41f18e2ce38ffbc9b6f03e1468838ce33ad72',
 'south_china_sea_p4_image0.png':'575a61cb613745b5507237c96e79bdd95852cd16c48ac79ad7e19bde2e404937',
}
SOURCES={
 'salpi':dict(author='Susini et al. (2023)',doi='10.1016/j.quaint.2022.10.012',
  url='https://www.earth-prints.org/handle/2122/16042',group='WEB_Salpi_SAM9',domain='core_longitudinal',
  figure='Figure3a,b; caption in accepted manuscript page52; original publisher gr3_lrg.jpg',
  interval='SAM9; photo-specific depth unavailable, no inferred metre labels',
  description_ru='По подписи авторов: плохо дренируемая пойма AF1, керн SAM9. Серо-бурые и бурые алевритовые отложения с тонкозернистым песком, преимущественно массивные, местами с тонкими песчаными прослоями. Корешки редкие. В панели b выбран только участок AF1 выше границы с песчаным прорывным конусом AF2.',
  label_basis='Fig3a explicitly poorly drained floodplain; Fig3b AF1/AF2 boundary manually localized. Fig3c partially drained floodplain NOT used as B.',
  rights='Repository metadata open.access; no explicit CC grant verified. Crossref lists Elsevier TDM terms, not CC.',
  distribution_approved=False,geologist_approved=False,
  limitations='Two correlated photos of one core; authors/region near some Italian source data. No claim of independent geology without provenance audit.'),
 'gaoyou':dict(author='Zhu et al. (2025)',doi='10.3390/app15084153',url='https://www.mdpi.com/2076-3417/15/8/4153',
  group='WEB_Gaoyou_L24',domain='core_longitudinal',figure='Figure3c, page6',interval='Well L24,2374.2m (caption); exact patch depths unavailable',
  description_ru='Скважина L24, глубина 2374,2 м; авторская интерпретация — продельта озёрной дельтовой системы. Тёмные глинистые породы с песчанистыми включениями и деформациями осадка; показаны признаки оползания и свёрнутая слоистость.',
  label_basis='Figure3c explicitly Prodelta; section5 establishes lacustrine delta context, not marine prodelta.',
  rights='CC BY4.0; PDF page1',distribution_approved=True,geologist_approved=False,
  limitations='Lithified lacustrine prodelta, source-domain shift from Holocene marine muds; one photograph.'),
 'hainan':dict(author='Liang et al. (2023)',doi='10.1038/s41598-023-40721-5',url='https://www.nature.com/articles/s41598-023-40721-5',
  group='WEB_Hainan_KZK01',domain='core_end_face',figure='Figure2,page4, upper sample photograph within U1',interval='CoreKZK01, U1 about16-19m; individual sample exact depth not assigned',
  description_ru='Продельтовый интервал U1: серо-жёлтая глина с серыми тонкопесчано-алевритовыми включениями, местами параллельная слоистость. Выбран участок поверхности образца керна без шкалы и фона; это поперечный торец, не продольный разрез.',
  label_basis='Fig2 U1 facies column and text; selected clay sample, not lowermost calcareous sandstone/gravel example.',
  rights='CC BY4.0; publisher Rights and permissions and PDF',distribution_approved=True,geologist_approved=False,
  limitations='One small core-sample end face, exact sample depth absent; orientation differs from longitudinal core.')}


def digest(path):return sha256(Path(path).read_bytes()).hexdigest()
def pixel_digest(im):
 h=sha256(f'{im.width}x{im.height}:RGB:'.encode());h.update(im.tobytes());return h.hexdigest()


def specs():
 rows=[]
 def add(sid,file,panel,code,box):rows.append(dict(source_id=sid,file=file,panel=panel,code=code,box=list(box)))
 for y in (30,200,370,540,870,1040):add('salpi','salpi_figure3_original.jpg','3a','B',(175,y,300,y+128))
 for y in (30,400):add('salpi','salpi_figure3_original.jpg','3b-AF1','B',(570,y,675,y+140))
 for x in (3650,3890):
  for y in (280,480):add('gaoyou','gaoyou_figure3_photos.png','3c','C',(x,y,x+210,y+180))
 add('hainan','south_china_sea_p4_image0.png','2-U1-endface','C',(1470,1605,1580,1705))
 return rows


def originals(raw,fetch=False):
 from pypdf import PdfReader
 raw.mkdir(parents=True,exist_ok=True)
 for name,(url,expected) in FILES.items():
  p=raw/name
  if not p.exists() and fetch:
   with urllib.request.urlopen(url,timeout=120) as response:data=response.read()
   if sha256(data).hexdigest()!=expected:raise ValueError('Source changed, needs review: '+name)
   p.write_bytes(data)
  if digest(p)!=expected:raise ValueError('Source hash differs: '+name)
 # Restore nine contiguous native-resolution image strips from Figure3. No scaling.
 p=raw/'gaoyou_figure3_photos.png'
 if not p.exists():
  page=PdfReader(raw/'gaoyou.pdf').pages[5];full=Image.new('RGB',(4339,900))
  for i in range(9):
   im=page.images[i].image.convert('RGB')
   if im.size!=(4339,100):raise ValueError('PDF image tiling changed')
   full.paste(im,(0,i*100))
  full.save(p)
 p=raw/'south_china_sea_p4_image0.png'
 if not p.exists():p.write_bytes(PdfReader(raw/'south_china_sea.pdf').pages[3].images[0].data)
 for name,expected in PHOTOS.items():
  if digest(raw/name)!=expected:raise ValueError('Original photo changed: '+name)


def prepare(raw,out):
 if out.exists():raise ValueError('Choose new data directory')
 out.mkdir(parents=True);(out/'images').mkdir();photos={}
 for name in PHOTOS:
  with Image.open(raw/name) as im:photos[name]=im.convert('RGB')
 rows=[];sheet=Image.new('RGB',(5*170,3*185),'#eeeeee')
 for i,s in enumerate(specs()):
  image=photos[s['file']];box=s['box'];x0,y0,x1,y1=box
  if not (0<=x0<x1<=image.width and 0<=y0<y1<=image.height):raise ValueError('Invalid box')
  crop=image.crop(box);rid='corefacies_'+sha256(json.dumps(s,sort_keys=True).encode()).hexdigest()[:24]
  path=out/'images'/(rid+'.png');crop.save(path);meta=SOURCES[s['source_id']]
  rows.append(dict(record_id=rid,image=str(path.resolve()),image_sha256=digest(path),pixel_sha256=pixel_digest(crop),image_size_px=list(crop.size),
   source=meta,source_id=s['source_id'],source_group=meta['group'],domain=meta['domain'],source_file=s['file'],source_file_sha256=PHOTOS[s['file']],source_panel=s['panel'],source_box=box,
   allowed_codes=[s['code']],exact_code=s['code'],split='train',external=True,training_scope='local_research_only',
   distribution_approved=meta['distribution_approved'],geologist_approved=False,project_92_facies_mapping=None,
   preprocessing='Original RGB crop without augmentation, retouching, recolouring or added detail'))
  tile=crop.copy();tile.thumbnail((155,145));px=(i%5)*170;py=(i//5)*185;sheet.paste(tile,(px+5,py+5))
  ImageDraw.Draw(sheet).text((px+5,py+155),str(i+1)+' '+s['source_id']+' '+s['panel']+' '+s['code'],fill='black')
 (out/'train.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8');sheet.save(out/'contact_sheet.png')
 summary=dict(crops=len(rows),class_counts=dict(Counter(r['exact_code'] for r in rows)),photographic_panels=4,source_groups=3,
  sources=SOURCES,raw_files=FILES,original_photos=PHOTOS,training_scope='local_research_only',distribution_approved=False,
  outcrop_crops=0,prepared_test_pixels_opened=0,geologist_approved=False,preparation_script_sha256=digest(__file__),
  limitations=['Eight B patches are two photos of only SAM9; five C patches are two different core photos.',
   'Hainan end-face and Gaoyou lacustrine setting retain separate metadata; no blind physical equivalence.',
   'No explicit CC permission verified for Salpi: raw files and trained mixed weights not published.'])
 (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 return summary


if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--raw',type=Path,default=ROOT/'open_datasets/target_core_facies_20261031_v6/raw');p.add_argument('--output',type=Path,required=True);p.add_argument('--fetch',action='store_true')
 a=p.parse_args();originals(a.raw.resolve(),a.fetch);print(json.dumps(prepare(a.raw.resolve(),a.output.resolve())['class_counts']))
