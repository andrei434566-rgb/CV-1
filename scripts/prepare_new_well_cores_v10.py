"""Caption-grounded new wells. Fixed crops, whole-well holdout, no inference."""
import argparse
from collections import Counter
from hashlib import sha256
import io
import json
from pathlib import Path
from PIL import Image, ImageCms, ImageDraw

PDFS = {
 'andsbjerg2003.pdf':'95e049fb937a441891426cce2f6cf9de97bdc213ddb67881a1879d10f75a896e',
 'bohai2025.pdf':'44a85bfb63941f9adfa7d93cf145e26484a1f0e1bb6ddc874300be3ad8f7dd24',
 'gsj2016.pdf':'29d4cb5e4f44b1604f27479f248134be20c81760f39081c381ec8a93c9f1d6fe'}
SOURCES = {
 'geus':dict(author='Andsbjerg (2003)',doi='10.34194/geusb.v1.4676',url='https://www.geus.dk/media/13879/nr1_p301-347.pdf',
  rights='Public GEUS PDF; explicit licence on publisher page not verified in this acquisition. Local research only; no photo/weight distribution.',
  note='West Lulu-3 training, West Lulu-2 held entirely out; shared publication/photographic style, not independent articles.'),
 'bohai':dict(author='Lei et al. (2025)',doi='10.3390/jmse13050881',url='https://www.mdpi.com/2077-1312/13/5/881',
  download_url='https://mdpi-res.com/d_attachment/jmse/jmse-13-00881/article_deploy/jmse-13-00881.pdf',rights='CC BY4.0, PDF page1',
  note='Only Fig3f: caption, depth0.64-1.04m and DU1 text agree on prodelta. Fig3a/c/d excluded for caption/text or depth/unit conflicts. BHZK/BXZK spelling variation retained as source alias.'),
 'gsj':dict(author='GSJ/AIST (2016), Kioroshi Formation chapter',url='https://www.gsj.jp/data/prompt-rep/GSJ_INTERIMREP_071_2016.pdf',
  rights='Public GSJ report; explicit redistribution grant not verified. Local held-out evaluation only.',
  note='All GS-TM-1 records held out. Only numbered1 fluvial sand and5 explicitly prodelta. Other environments not forced into six classes.')}


def digest(path):return sha256(Path(path).read_bytes()).hexdigest()
def pixel_digest(im):return sha256(f'{im.width}x{im.height}:RGB:'.encode()+im.tobytes()).hexdigest()
def write(path,data):path.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def specifications():
 rows=[]
 def add(key,source,well,code,split,figure,interval,description,boxes):
  for box in boxes:rows.append(dict(source_file=key,source_id=source,well=well,allowed_codes=[code],split=split,figure=figure,
   published_panel_interval_m=interval,crop_exact_interval_m=None,description_ru=description,box=list(box)))
 base='По авторам: активное заполнение речного русла, песчаники с косой/рябной слоистостью либо массивные, с углистыми и глинистыми обломками.'
 for panel,index,depth,ys in [('B',1,3751.0,[(30,140),(175,285),(365,465),(525,660)]),('C',2,3751.8,[(30,140),(500,640)]),
                             ('D',3,3753.3,[(40,180),(300,420),(500,600)]),('E',4,3754.1,[(40,180),(320,420),(460,600)])]:
  left,right={'B':(45,105),'C':(45,95),'D':(35,85),'E':(25,75)}[panel]
  add(f'geus_p12_{index}.png','geus','West Lulu-3','E','train','Fig5'+panel,[depth,None],base,[(left,y,right,end) for y,end in ys])
 for panel,index,depth,boxes in [('I',2,3860.0,[(50,600,115,740),(50,760,115,900)]),('J',3,3861.6,[(40,230,110,365),(40,600,110,690)]),
                               ('K',4,3862.2,[(40,355,100,475),(40,570,100,700)])]:
  add(f'geus_p13_{index}.png','geus','West Lulu-2','E','external_holdout','Fig5'+panel,[depth,None],base,boxes)
 add('bohai_fig3.png','bohai','BHZK2017-2','C','train','Fig3f',[.64,1.04],
  'По авторам: продельтовые алевритисто-глинистые отложения DU1; биотурбация, тонкое переслаивание и морские/солоноватоводные раковинные остатки.',
  [(2410,y,2580,y+140) for y in (150,330,510,690,870,1050)])
 add('japan_fig4.png','gsj','GS-TM-1','E','external_holdout','Fig4-1',[62.9,63.4],
  'По авторам: речная фация UnitI, песок с гравием и косой слоистостью.',[(10,y,120,y+130) for y in (35,185,335,485,635)])
 add('japan_fig4.png','gsj','GS-TM-1','C','external_holdout','Fig4-5',[28.1,28.6],
  'По авторам: продельтовая фация UnitIV, биотурбированный песчанистый ил.',[(635,y,740,y+130) for y in (35,185,335,485,635)])
 return rows


def extract(raw):
 from pypdf import PdfReader
 for name,h in PDFS.items():
  if digest(raw/name)!=h:raise ValueError('Reviewed PDF changed: '+name)
 r=PdfReader(raw/'andsbjerg2003.pdf')
 cs=r.pages[11]['/Resources']['/ColorSpace']['/DefaultCMYK'].get_object()
 profile=cs[1].get_object().get_data()
 if sha256(profile).hexdigest()!='990ff8ae33980a5d51846e17afef684946f659e2f834eb1b8eefa63ecb6ce316':raise ValueError('Reviewed embedded CMYK profile changed')
 for page,indices in ((11,(1,2,3,4)),(12,(2,3,4))):
  for index in indices:
   im=r.pages[page].images[index].image
   if im.mode!='CMYK':raise ValueError('GEUS source colour mode changed')
   ImageCms.profileToProfile(im,io.BytesIO(profile),ImageCms.createProfile('sRGB'),outputMode='RGB').save(raw/f'geus_p{page+1}_{index}.png')
 r=PdfReader(raw/'bohai2025.pdf');images=[x.image.convert('RGB') for x in r.pages[5].images]
 if [x.size for x in images]!=[(2725,969),(2725,322)]:raise ValueError('Bohai original strips changed')
 full=Image.new('RGB',(2725,1291));full.paste(images[0],(0,0));full.paste(images[1],(0,969));full.save(raw/'bohai_fig3.png')
 r=PdfReader(raw/'gsj2016.pdf');im=r.pages[47].images[1].image
 if im.mode!='CMYK' or im.size!=(1286,865):raise ValueError('Japan source changed')
 # The report uses DeviceCMYK, no ICC/default override; this conversion was
 # checked visually against a full Poppler-rendered PDF page, including labels.
 im.convert('RGB').save(raw/'japan_fig4.png')


def prepare(raw,out):
 if out.exists():raise ValueError('Choose a new data folder')
 extract(raw);out.mkdir(parents=True);(out/'images').mkdir();rows=[]
 write(out/'summary.json',dict(do_not_train=True,reason='Preparation not yet completed'))
 specs=specifications()
 for i,spec in enumerate(specs):
  with Image.open(raw/spec['source_file']) as im:crop=im.convert('RGB').crop(spec['box'])
  if min(crop.size)<50:raise ValueError('Crop too small')
  # Reject white page margins/blank core gaps, without recolouring rock.
  blank=sum(all(v>245 for v in p) for p in crop.getdata())/(crop.width*crop.height)
  if blank>.001:raise ValueError('White margin/gap in crop '+str(i))
  rid=f"newwell_v10_{i:03d}";path=out/'images'/f'{rid}.png';crop.save(path)
  rows.append(dict(spec,record_id=rid,image=str(path.resolve()),image_sha256=digest(path),pixel_sha256=pixel_digest(crop),
   source_file_sha256=digest(raw/spec['source_file']),source=SOURCES[spec['source_id']],external=True,domain='core_longitudinal',
   source_group='WEB_GEUS2003_WestLulu3' if spec['source_id']=='geus' and spec['split']=='train' else 'WEB_'+spec['source_id']+'_'+spec['well'],
   geologist_approved=False,training_approval_scope='author-caption-grounded six-class research only; not92 project facies',
   color_processing='Embedded PDF CMYK ICC -> sRGB' if spec['source_id']=='geus' else 'PDF image RGB interpretation; no manual colour correction',white_fraction=blank))
 train=[r for r in rows if r['split']=='train'];hold=[r for r in rows if r['split']=='external_holdout']
 if {r['well'] for r in train}&{r['well'] for r in hold}:raise ValueError('Whole-well leakage')
 if len({r['pixel_sha256'] for r in rows})!=len(rows):raise ValueError('Repeated pixels')
 for name,subset in (('all',rows),('train',train),('external_holdout',hold)):
  (out/(name+'.jsonl')).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in subset),encoding='utf-8')
 contact=Image.new('RGB',(7*190,((len(rows)+6)//7)*190),'white');draw=ImageDraw.Draw(contact)
 for i,r in enumerate(rows):
  with Image.open(r['image']) as im:im.thumbnail((170,145));contact.paste(im,((i%7)*190+10,(i//7)*190+26))
  draw.text(((i%7)*190+3,(i//7)*190+3),f"{i:02} {r['allowed_codes'][0]} {r['split']}",fill='black')
 contact.save(out/'contact.png')
 summary=dict(train_count=len(train),holdout_count=len(hold),sources=SOURCES,pdf_sha256=PDFS,source_photo_sha256={r['source_file']:r['source_file_sha256'] for r in rows},
  train_classes=dict(Counter(r['allowed_codes'][0] for r in train)),holdout_classes=dict(Counter(r['allowed_codes'][0] for r in hold)),
  split_frozen_before_model_predictions=True,geologist_approved=False,distribution_approved=False,do_not_train=False,
  exclusions=['Bohai Fig3a/c/d: caption/body/depth conflicts; no drainage subtype guessed','GEUS Fig5A/G/H/L and estuarine figures: mixed/ambiguous or outside chosen class','Japan Fig4-2/3/4/6/7/8: not forcibly relabelled as peat/floodplain/channel/prodelta'])
 write(out/'summary.json',summary);print(json.dumps(summary,ensure_ascii=True,indent=2))


if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--raw',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();prepare(a.raw.resolve(),a.output.resolve())
