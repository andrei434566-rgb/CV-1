"""Core-only caption-grounded contrasts. No network or invented depths.

Po CC BY photographs and Rome NC-ND research-review material are separate.
This script publishes no images/weights and grants no new distribution rights.
"""
import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
from PIL import Image,ImageDraw

ROOT=Path(__file__).resolve().parents[1]
PDFS={
 'po_basin.pdf':'e1cd5f694ad4f3cf32859e111925fbd0970dcad46426d9b9c0f0de58c9563974',
 'rome2018.pdf':'e83a26f9046f40253ed25c6e7c0e084923d6c862fb9ab1768dc2a280038ffc77',
}
PHOTOS={
 'po_p8_image0.jpg':'d7894ebac86e7998f83ba3350375bc4945ab0f9076c5d7c1e8d38ab3ce427677',
 'rome_plateII_ef.png':'245a5d0cc44f050dac795aa5179d86b479f2847c06a8b72301cd846fab8c0a0e',
 'rome_p17_image4.jpg':'9d943e81dd0fb98d4dafb3e8baf0abc754d3f8d95a5d1194080c0651cb8852cc',
}
SOURCES={
 'po':dict(author='Amorosi et al. (2021)',doi='10.1111/sed.12784',
  url='https://doi.org/10.3929/ethz-b-000444047',
  download_url='https://www.research-collection.ethz.ch/server/api/core/bitstreams/60343ac9-e6d6-4327-b517-a8ccbac1675a/content',
  license='CC BY4.0, ETH generated cover and published article page402',distribution_approved=True,
  source_group='WEB_Po2021_B1_B2',group_note='Both cores kept together because one paper/figure; possible reuse/overlap with anonymized UniBo regions not ruled out',
  figure='Fig3, article page408, PDF page8 including generated ETH cover',
  domains=['core_longitudinal'],geologist_approved=False,
  download_note='Downloaded PDF has an automatically generated ETH cover and therefore differs in byte count/MD5 from repository bitstream metadata. This exact returned PDF and extracted image were reviewed and pinned separately.'),
 'rome':dict(author='Mancini et al. (2018)',doi='10.26382/AMQ.2018.11',
  url='https://amq.aiqua.it/index.php/amq/article/view/137',
  download_url='https://amq.aiqua.it/index.php/amq/article/download/137/110',
  license='CC BY-NC-ND4.0, explicit article page',distribution_approved=False,
  source_group='WEB_Rome2018_Palatine',group_note='All selected boreholes grouped together to avoid paper/photo leakage',
  figure='PlateII e/f (PDF15, printed185), PlateIII f (PDF17, printed187)',
  domains=['core_longitudinal'],geologist_approved=False,
  restriction='Retain as separate local research/review materials; cropped images and mixed weights not redistributed under a false CC BY/NC-SA grant.'),
}
TARGETS={
 'Po3A':dict(source='po',code='E',well='B1',interval=[29.0,29.4],panel='3A',
  description_ru='По авторам: средне- и крупнозернистый русловой песок, умеренно сортированный. Выбран внутренний участок фотографии, без края и фона.'),
 'Po3D':dict(source='po',code='A',well='B1',interval=[14.4,14.9],panel='3D',
  description_ru='По авторам: пёстроокрашенные алевритовые и глинистые отложения хорошо дренируемой поймы. Светлая основная окраска с желтовато-оранжевыми пятнами сохранена без коррекции цвета.'),
 'Po3E':dict(source='po',code='B',well='B2',interval=[7.2,7.7],panel='3E',
  description_ru='По авторам: алевритовые и глинистые отложения плохо дренируемой поймы с органическим веществом. Сохранены светло-серые тонкозернистые участки и мелкие тёмные включения; точная гранулометрия по пикселям не назначается.'),
 'Po3F':dict(source='po',code='F',well='B2',interval=[10.3,10.8],panel='3F',
  description_ru='По авторам: обогащённая органическим веществом болотная глина. Это отдельный контраст к плохо дренируемой пойме; наличие растительных остатков не превращает выбранную глину в отдельный прослой торфа.'),
 'Po3G':dict(source='po',code='D',well='B2',interval=[17.3,17.8],panel='3G',
  description_ru='По авторам: болотный торф. Участки с заметным органическим материалом сохраняются как класс торфа, отдельно от болотной глины и плохо дренируемой поймы.'),
 'RomeIIe':dict(source='rome',code='B',well='22 P',interval=[20.0,25.0],panel='PlateIIe',
  description_ru='По авторам: светло-серая, преимущественно массивная алевритовая глина с редкими карбонатными конкрециями, плохо дренируемая пойма FTR D. Выбран чистый керн; красная стрелка и лоток исключены.'),
 'RomeIIf':dict(source='rome',code='B',well='9 MS',interval=[25.0,30.0],panel='PlateIIf-FTRD',
  description_ru='По авторам: голубовато-серая массивная, богатая органическим веществом глина плохо дренируемой поймы FTR D. Угольный прослой и вышележащий хорошо дренируемый интервал исключены из выбранных участков.'),
 'RomeIIIfB':dict(source='rome',code='B',well='18 P',interval=[15.0,20.0],panel='PlateIIIf-SFTbaD',
  description_ru='По авторам: серый, богатый органическим веществом пойменный ил плохо дренируемой поймы SFTba D. Взяты участки серого столбика без надписи и этикетки.'),
 'RomeIIIfA':dict(source='rome',code='A',well='18 P',interval=[15.0,20.0],panel='PlateIIIf-SFTbaF',
  description_ru='По авторам: вышележащий светло-бурый педогенный песчанистый алеврит хорошо дренируемой поймы SFTba F. Выбраны участки выше красной границы, отдельно от серого плохо дренируемого материала.'),
}


def digest(p):return sha256(Path(p).read_bytes()).hexdigest()
def pixel_digest(im):
 h=sha256(f'{im.width}x{im.height}:RGB:'.encode());h.update(im.tobytes());return h.hexdigest()


def specifications():
 s=[]
 def add(target,file,box):s.append(dict(target=target,file=file,box=list(box)))
 for name,x in [('Po3A',40),('Po3D',900),('Po3F',1470),('Po3G',1770)]:
  for y in (60,250,440,630,820,1010):add(name,'po_p8_image0.jpg',(x,y,x+150,y+150))
 for x in (1180,1280):
  for y in (60,250,440,630,820,1010):add('Po3E','po_p8_image0.jpg',(x,y,x+95,y+150))
 # Narrow native core lanes in Rome: no interpolation into new detail.
 for x in (168,268,365):
  for y in (110,280):add('RomeIIe','rome_plateII_ef.png',(x,y,x+50,y+140))
 for x in (880,980):
  for y in (80,300):add('RomeIIf','rome_plateII_ef.png',(x,y,x+45,y+140))
 for y in (100,650):add('RomeIIIfB','rome_p17_image4.jpg',(990,y,1045,y+140))
 for y in (110,300,490,680):add('RomeIIIfA','rome_p17_image4.jpg',(1110,y,1150,y+140))
 return s


def extract(raw):
 from pypdf import PdfReader
 for name,h in PDFS.items():
  if digest(raw/name)!=h:raise ValueError('Reviewed publication differs: '+name)
 po=PdfReader(raw/'po_basin.pdf').pages[7].images[0]
 if sha256(po.data).hexdigest()!=PHOTOS['po_p8_image0.jpg']:raise ValueError('Po original differs')
 (raw/'po_p8_image0.jpg').write_bytes(po.data)
 rome=PdfReader(raw/'rome2018.pdf');a=rome.pages[14].images[3].image.convert('RGB');b=rome.pages[14].images[4].image.convert('RGB')
 if a.size!=(1534,504) or b.size!=(1534,503):raise ValueError('PlateII layout differs')
 full=Image.new('RGB',(1534,1007));full.paste(a,(0,0));full.paste(b,(0,504));full.save(raw/'rome_plateII_ef.png')
 (raw/'rome_p17_image4.jpg').write_bytes(rome.pages[16].images[4].data)
 actual={n:digest(raw/n) for n in PHOTOS}
 if actual!=PHOTOS:raise ValueError('Original extraction differs from reviewed images')
 return actual


def prepare(raw,out):
 if out.exists():raise ValueError('Choose a new output directory')
 hashes=extract(raw);out.mkdir(parents=True);(out/'images').mkdir();photos={}
 for n in PHOTOS:
  with Image.open(raw/n) as im:photos[n]=im.convert('RGB')
 records=[];thumbs=[]
 for i,s in enumerate(specifications()):
  target=TARGETS[s['target']];meta=SOURCES[target['source']];im=photos[s['file']];x,y,xx,yy=s['box']
  if not(0<=x<xx<=im.width and 0<=y<yy<=im.height):raise ValueError('Invalid crop')
  crop=im.crop(s['box']);rid='contrast_'+sha256(json.dumps(s,sort_keys=True).encode()).hexdigest()[:24]
  path=out/'images'/(rid+'.png');crop.save(path)
  records.append(dict(record_id=rid,image=str(path.resolve()),image_sha256=digest(path),pixel_sha256=pixel_digest(crop),image_size_px=list(crop.size),
   allowed_codes=[target['code']],exact_code=target['code'],source=meta,source_id=target['source'],source_group=meta['source_group'],well=target['well'],
   source_panel=target['panel'],source_file=s['file'],source_file_sha256=hashes[s['file']],source_box=s['box'],
   photographed_interval_m=target['interval'],crop_exact_depth_m=None,description_ru=target['description_ru'],
   domain='core_longitudinal',external=True,split='train',distribution_approved=meta['distribution_approved'],
   training_scope='source_research_within_verified_distribution_terms' if meta['distribution_approved'] else 'local_research_review_only',
   geologist_approved=False,project_92_facies_mapping=None,preprocessing='Unmodified RGB crop, native source resolution; no recolouring, retouching or invented detail'))
  tile=crop.copy();tile.thumbnail((145,135));thumb=Image.new('RGB',(180,175),'white');thumb.paste(tile,(5,5))
  ImageDraw.Draw(thumb).text((5,145),str(i+1)+' '+s['target']+' '+target['code'],fill='black');thumbs.append(thumb)
 def write(name,rows):(out/name).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
 write('all_local.jsonl',records);write('public_terms_train.jsonl',[r for r in records if r['distribution_approved']]);write('local_review_only.jsonl',[r for r in records if not r['distribution_approved']])
 cols=6;sheet=Image.new('RGB',(cols*180,((len(thumbs)+cols-1)//cols)*175),'#eeeeee')
 for i,t in enumerate(thumbs):sheet.paste(t,((i%cols)*180,(i//cols)*175))
 sheet.save(out/'contact_sheet.png')
 summary=dict(crops=len(records),class_counts=dict(Counter(r['exact_code'] for r in records)),
  distribution_crops=sum(r['distribution_approved'] for r in records),local_review_crops=sum(not r['distribution_approved'] for r in records),
  source_groups=sorted({r['source_group'] for r in records}),named_boreholes=sorted({r['well'] for r in records}),
  source_photo_panels=8,pdfs=PDFS,photo_sha256=hashes,sources=SOURCES,targets=TARGETS,
  source_preparation_sha256=digest(__file__),prepared_test_pixels_opened=0,geologist_approved=False,
  limitations=['52 correlated crops from8 panels and2 papers, not52 independent core photos',
   'Five named boreholes do not establish independence from old anonymized S-groups',
   'Publication intervals retained; exact crop depths not invented',
   'No new prodelta photo in these two papers accepted; lagoon and marine clay are not relabelled prodelta',
   'Rome NC-ND terms retained separately; images/mixed weights not publicly redistributed'])
 (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');return summary


if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--raw',type=Path,default=ROOT/'open_datasets/core_materials_20261031_v7/raw');p.add_argument('--output',type=Path,required=True)
 a=p.parse_args();result=prepare(a.raw.resolve(),a.output.resolve());print(json.dumps({k:result[k] for k in ('crops','class_counts','distribution_crops','local_review_crops','named_boreholes')}))
