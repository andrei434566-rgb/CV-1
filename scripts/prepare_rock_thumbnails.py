"""Real source-labelled illustrative training images, never generated analogues."""
import io,json,hashlib,sys,zipfile
from pathlib import Path
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from app.domain.rock_types import ROCK_CLASSES

def main():
    out=ROOT/'app/resources/lithology_examples'
    attribution=json.loads((out/'ATTRIBUTION.json').read_text(encoding='utf8'))
    parts=json.loads((ROOT/'models/vision/rock_taxonomy_20261105/dataset.json').read_text(encoding='utf8'))
    examples=attribution['examples']
    for i,code in enumerate(ROCK_CLASSES):
        if code=='sandstone':continue  # Preserve the already attributed Mendeley example.
        row=next(r for r in parts['train'] if r['label']==i)
        if 'archive' in row:
            with zipfile.ZipFile(row['archive']) as z:data=z.read(row['member'])
            credit='Jiayu Li et al. · DCID · CC BY-NC 4.0'
            entry={'source_member':row['member'],'source_label':row['member'].split('/')[3],
                   'source':'https://github.com/JiayuLi1120/drill-core-image-dataset','license':'CC BY-NC 4.0',
                   'license_url':'https://creativecommons.org/licenses/by-nc/4.0/','credit':credit}
        else:
            data=Path(row['image']).read_bytes();entry=dict(examples['garbage'])
            assert hashlib.sha256(data).hexdigest()==entry['source_image_sha256'], 'Background source changed; update its member attribution'
            entry.update(credit='Fatimah Alzubaidi · Lithology dataset · CC BY 4.0')
        with Image.open(io.BytesIO(data)) as im:im.convert('RGB').resize((160,160),Image.Resampling.LANCZOS).save(out/(code+'.jpg'),quality=90)
        entry.update(file=code+'.jpg',source_image_sha256=hashlib.sha256(data).hexdigest(),source_split='train',
                     label_provenance='source_direct_label',modification='Resized to 160 x 160 for illustrative thumbnail; not a nearest analogue')
        examples[code]=entry
    coal=json.loads((ROOT/'models/vision/rock_taxonomy_coal_20261105/coal_dataset.json').read_text(encoding='utf8'))['train'][0]
    with zipfile.ZipFile(coal['archive']) as z:data=z.read(coal['member'])
    with Image.open(io.BytesIO(data)) as im:im.convert('RGB').crop(coal['box']).resize((160,160),Image.Resampling.LANCZOS).save(out/'coal.jpg',quality=90)
    examples['coal']={'file':'coal.jpg','source_member':coal['member'],'box':coal['box'],
         'source_image_sha256':hashlib.sha256(data).hexdigest(),'source_label':'0: coal',
         'source':'https://zenodo.org/records/10702879','author':'Wenyu Wang','license':'CC BY 4.0',
         'license_url':'https://creativecommons.org/licenses/by/4.0/', 'credit':'Wenyu Wang · Coal · CC BY 4.0',
         'modification':'Source-labelled bounding-box crop resized to 160 x 160',
         'source_split':'train (parent-photo grouped split)','domain':'Coal surface texture; not core-box photograph'}
    attribution['examples']=examples
    (out/'ATTRIBUTION.json').write_text(json.dumps(attribution,ensure_ascii=False,indent=2),encoding='utf8')
    codes=list(ROCK_CLASSES)+['coal'];sheet=Image.new('RGB',(600,600),'white');draw=ImageDraw.Draw(sheet)
    for i,code in enumerate(codes):
        x=(i%3)*200;y=(i//3)*200
        with Image.open(out/(code+'.jpg')) as im:sheet.paste(im,(x,y))
        draw.text((x,y+163),code,fill='black')
    sheet.save(ROOT/'research_reports/rock_taxonomy_20261105/class_thumbnails.jpg')
    print('Prepared '+str(len(codes))+' attributed source thumbnails')

if __name__=='__main__':main()
