"""Caption-linked photographic ROIs; retain source, group and visibility limits.

Core photos only. Original overlays, sketches, thin sections and ambiguous
compound captions are kept in the audit rather than silently used as labels.
"""
from pathlib import Path
import hashlib,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
R=ROOT/'open_datasets/bedding_20261106/expansion'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

SOURCES={
 'chengjiang':('Saleh et al., 2022, Nature Communications 13:1569','https://doi.org/10.1038/s41467-022-29246-z','CC BY 4.0'),
 'benxi':('Liu et al., 2026, Fractal and Fractional 10:374','https://doi.org/10.3390/fractalfract10060374','CC BY 4.0'),
 'pinghu':('Chen et al., 2022, Energies 15:7214','https://doi.org/10.3390/en15197214','CC BY 4.0'),
 'sydney':('Sedimentology and Stratigraphy of an Upper Permian Sedimentary Succession, 2021, Geosciences 11:273','https://doi.org/10.3390/geosciences11070273','CC BY 4.0'),
 'longtan':('Deciphering Ancient Tidal Depositional Systems, 2026, JMSE 14:1797','https://doi.org/10.3390/jmse14191797','CC BY 4.0'),
 'gulong':('Discovery and significance of tempestites and storm deposits, 2022, Frontiers Earth Science','https://doi.org/10.3389/feart.2022.999135','CC BY'),
 'sangonghe':('Li et al., 2019, Journal of Petroleum Exploration and Production Technology 9:1883-1896','https://doi.org/10.1007/s13202-018-0599-0','CC BY 4.0'),
 'xujiahe':('Geomechanical Response Characteristics of Different Sedimentary Hydrodynamic Cycles, 2024, Sustainability 16:4304','https://doi.org/10.3390/su16104304','CC BY 4.0'),
 'zubair':('Al-Rubaye et al., 2019, Iraqi Journal of Science 60:290-307','https://ijs.uobaghdad.edu.iq/index.php/eijs/article/view/606','Journal CC BY policy; retained for review'),
 'ordos':('The Differential Enrichment Law of Tight Sandstone Gas, 2024, Energies 17:5978','https://doi.org/10.3390/en17235978','CC BY 4.0'),
 'opalinus':('Lauper et al., 2021, Frontiers Earth Science 9:645596','https://doi.org/10.3389/feart.2021.645596','CC BY'),
}

def main():
 from PIL import Image,ImageDraw
 R.mkdir(parents=True,exist_ok=True);(R/'crops').mkdir(exist_ok=True)
 rows=[]
 def add(identifier,file,box,label,source,group,caption,split,figure,well=None,notes=''):
  p=ROOT/file
  with Image.open(p) as im:
   if not (0<=box[0]<box[2]<=im.width and 0<=box[1]<box[3]<=im.height):raise ValueError(identifier+': bad crop')
   crop=im.convert('RGB').crop(box);q=R/'crops'/f'{identifier}.png';crop.save(q)
  rows.append({'id':identifier,'image':str(q.relative_to(ROOT)),'sha256':sha(q),'source_image':file,
   'source_sha256':sha(p),'source_box':box,'positive_type':label,'source':source,'source_group':group,
   'well':well,'author_description':caption,'figure':figure,'split':split,
   'domain':'photographic_core','label_provenance':'caption_and_spatially_matched_author_label',
   'research_training_approved':split=='train','deployment_training_approved':False,'notes':notes,
   'unknown_other_attributes':True})
 def local(identifier,file,box,label,source,group,caption,split,figure,well=None,notes=''):
  add(identifier,str((R/file).relative_to(ROOT)),box,label,source,group,caption,split,figure,well,notes)
 # Multiple ROIs from one core stay together and are not independent samples.
 local('cheng_hcs_a','chengjiang_fig2.png',(10,400,300,950),'Бугорчатая косая','chengjiang','chengjiang:single_core',
       'Hummocky cross-stratified sandstone, Fig. 2a.','train','2a','Yuanshan Jinning core')
 local('cheng_hcs_s1a','chengjiang_s1.png',(220,350,460,600),'Бугорчатая косая','chengjiang','chengjiang:single_core',
       'HCS sandstone in the upper part of Supplementary Figure 1a.','train','S1a','Yuanshan Jinning core')
 local('cheng_hcs_s1b','chengjiang_s1.png',(570,350,960,470),'Бугорчатая косая','chengjiang','chengjiang:single_core',
       'Hummocky cross-stratified sandstone at the top of Supplementary Figure 1b.','train','S1b','Yuanshan Jinning core')
 for suffix,box,fig in [('e',(420,580,580,770),'2e'),('g',(45,1245,475,1355),'2g'),('i',(580,1030,1080,1210),'2i')]:
  local('cheng_graded_'+suffix,'chengjiang_fig2.png',box,'Градационная','chengjiang','chengjiang:single_core',
        'Normal graded bed identified by white triangles in the original author figure.','train',fig,'Yuanshan Jinning core',
        'Crop excludes triangles and text; bedding may coexist with other structures. No inference of other absent properties.')
 local('pinghu_herringbone','pinghu_fig10.png',(1260,940,1420,1065),'Разнонаправленная косая','pinghu','pinghu:W3',
       '4058.72 m, developed herringbone crossbedding.','train','10c','W-3',
       'Herringbone is a specific positive example of opposite-dipping sets, not a synonym for all cross bedding.')
 local('sydney_flaser','sydney_fig5.png',(976,1010,1092,1145),'Флазерная','sydney','sydney:unresolved_core',
       'Flaser bedding at the author arrow in Figure 5C.','train','5C',None,
       'Selected clean part next to author arrow; figure text and arrow excluded. Exact well not specified for this panel.')
 local('sydney_lenticular','sydney_fig5.png',(774,830,1050,940),'Линзовидная','sydney','sydney:unresolved_core',
       'Lenticular bedding in the upper author-labelled part of Figure 5C.','train','5C')
 local('benxi_flaser','benxi_fig4.png',(688,975,888,1230),'Флазерная','benxi','benxi:J17',
       'Medium sandstone with flaser lamination and weak bioturbation, 2272.03-2272.22 m.','test','4f','J17')
 local('benxi_lenticular','benxi_fig4.png',(930,977,1131,1200),'Линзовидная','benxi','benxi:J1',
       'Fine sandstone with lenticular lamination and strong bioturbation, 2086.02-2086.21 m.','test','4g','J1')
 local('gulong_wave','gulong_fig4.webp',(18,1435,210,1660),'Волнистая','gulong','gulong:unresolved_core',
       'Wave bedding in Figure 4G; small unannotated photographic portion.','train','4G',None,
       'Context-limited crop; physical orientation and full wavelength not recovered. Gulong remains one source group.')
 add('opalinus_lenticular','open_datasets/bedding_20261106/additional/opalinus_fig2.webp',
     (480,656,880,980),'Линзовидная','opalinus','opalinus:BDM_BGC_unresolved',
     'SF2 macrofacies, characterized by lenticular bedding.','train','2 SF2',None,'BDM-B2 or BGC-2; keep both unresolved wells in one group.')
 add('opalinus_lenticular_wavy','open_datasets/bedding_20261106/additional/opalinus_fig2.webp',
     (480,1128,875,1440),'Линзовидно-волнистая','opalinus','opalinus:BDM_BGC_unresolved',
     'SF3 macrofacies, characterized by lenticular to wavy bedding, often bioturbated.','train','2 SF3')
 # The author's two named graded examples, held out as a whole source group.
 local('sangonghe_graded_c','sangonghe_fig3.jpg',(1140,90,1495,570),'Градационная','sangonghe','sangonghe:all_unresolved_wells',
       'Light grey medium conglomerate with graded bedding.','test','3c',None,'The source caption and scale are authoritative; figure wells unresolved.')
 local('sangonghe_graded_h','sangonghe_fig3.jpg',(620,1400,995,2300),'Градационная','sangonghe','sangonghe:all_unresolved_wells',
       'Light grey medium conglomerate with graded bedding.','test','3h',None,'Same source group as 3c; do not count as two independent wells.')
 excluded=[
  {'source':'benxi','figure':'4b,4d','candidate_types':['Разнонаправленная косая','Бугорчатая косая'],
   'reason':'White traced laminae overlay the diagnostic morphology; photographic truth retained only for reference.'},
  {'source':'sydney','figure':'5D','candidate_types':['Разнонаправленная косая'],
   'reason':'Traced bedding and directional arrows overlay the diagnostic geometry.'},
  {'source':'longtan','figure':'13b,13d','reason':'Outcrop domain, excluded from core-photo training.'},
  {'source':'longtan','figure':'13a,13c','candidate_types':['Линзовидная'],
   'reason':'Inscribed arrows/text and fractures obscure a large part of narrow core; retained for review.'},
  {'source':'zubair','figure':'5','candidate_types':['Флазерная'],
   'reason':'Poor visibility/contrast of lamination, pen and card; do not train solely because caption says flaser.'},
  {'source':'xujiahe','figure':'3B','candidate_types':['Градационная'],
   'reason':'Most complete grain-size trend obscured by drilled plug holes and labels in the published photo.'},
  {'source':'xujiahe','figure':'3C,4A','candidate_types':['Троговая крупная косая'],
   'reason':'Author labels trough cross-bedding but does not establish large-set size; cannot use as exact requested type.'},
  {'source':'sangonghe','figure':'3d','reason':'Caption calls it parallel bedding, text also refers to it under graded bedding; inconsistent source label.'},
  {'source':'sangonghe','figure':'3e,3f','candidate_types':['Таблитчатая крупная косая','Крупная косая'],
   'reason':'Caption has plate/cross bedding without exact large-set confirmation for these specific photos.'},
 ]
 # Provenance of downloaded publications, including rejected material.
 downloads=[]
 for p in sorted(R.glob('*')):
  if p.suffix.lower() in ('.pdf','.html','.webp') or p.name in ('chengjiang_fig2.png','sangonghe_fig3.jpg'):
   downloads.append({'file':p.name,'sha256':sha(p),'bytes':p.stat().st_size})
 manifest={'schema':'cv1-bedding-expansion-v1','sources':{k:{'citation':v[0],'url':v[1],'license':v[2]} for k,v in SOURCES.items()},
   'examples':rows,'excluded':excluded,'downloaded_files':downloads,'selection':'manual visual matching to author-captioned photographic regions',
   'future_training_policy':'Positive label only; unmentioned properties unknown. Same source/core group cannot cross splits.',
   'limitations':'Very small, publication-biased positive sample. Not a representative 16-class dataset. All promotion gates remain closed.'}
 (R/'dataset.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf8')
 # Visual QA sheet from the exact exported training/test ROIs.
 sheet=Image.new('RGB',(1500,((len(rows)+4)//5)*310),'white');draw=ImageDraw.Draw(sheet)
 for i,row in enumerate(rows):
  im=Image.open(ROOT/row['image']);im.thumbnail((290,275));x=(i%5)*300;y=(i//5)*310
  sheet.paste(im,(x,y+30));draw.text((x,y),row['id']+' / '+row['split'],fill='black')
 sheet.save(R/'exported_contact_sheet.jpg')
 from collections import Counter
 print(json.dumps({'n':len(rows),'splits':dict(Counter(r['split'] for r in rows)),
                  'types':dict(Counter(r['positive_type'] for r in rows))},ensure_ascii=False))

if __name__=='__main__':main()
