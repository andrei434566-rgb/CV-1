"""Exact laboratory/source records, with depth-to-photo links, never whole-photo labels.

These are unconsolidated sediments, not automatically sandstone/argillite.
Pixel ROI alignment and the absent coarse-sand class prevent four-class deployment.
"""
import csv,hashlib,io,json,re,sys,zipfile
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from app.infrastructure.grain_measurement import classify_sand_mm

def main():
    folder=ROOT/'open_datasets/rock_web_20261106'
    with zipfile.ZipFile(folder/'usgs_corephotos.zip') as z:photos=[n for n in z.namelist() if n.endswith('.jpg')]
    with zipfile.ZipFile(folder/'usgs_grainsize.zip') as z:
        member='2023-309-FA_GrainSize_SumStats.csv';raw=z.read(member)
    original=list(csv.DictReader(io.StringIO(raw.decode('utf-8-sig'),newline='')))
    rows=[];excluded=[]
    for i,row in enumerate(original):
        if not re.fullmatch(r'2023-309-FA_TYND-VC\d+',row['Core ID'] or ''):
            excluded.append({'source_row':i+2,'content':row['Core ID'],'reason':'Footer/abbreviation note, not a sample'});continue
        core=row['Core ID'].removeprefix('2023-309-FA_');depth=row['Depth\n (cm)']
        top,base=map(float,depth.split('-'));matches=[]
        for name in photos:
            m=re.fullmatch(re.escape(core)+r'_(\d+)-(\d+)_FullStitch.jpg',name)
            if m and float(m[1])<=top and base<=float(m[2]):matches.append(name)
        value=row['D50 \n(µm)'].strip();d50=float(value)/1000 if value else None
        rows.append({'id':f'usgs_standrew_{i:03}','core':core,'interval_top_cm':top,'interval_base_cm':base,
            'photo_members':matches,'source_csv_member':member,'source_csv_sha256':hashlib.sha256(raw).hexdigest(),
            'source_row':i+2,'source_fields':row,'lab_d50_mm':d50,
            'wentworth_median_band':classify_sand_mm(d50) if d50 else '',
            'label_definition':'Laboratory bulk sediment median D50, not mean grain size or 2D visible intercepts',
            'source':'https://doi.org/10.5066/P1AFJDNE','license':'US Government public domain',
            'author':'U.S. Geological Survey, St. Petersburg Coastal and Marine Science Center',
            'image_roi':None,'training_approved':False,
            'exclusion_reason':'Depth interval is linked, but pixel ROI is unverified; unconsolidated sediment domain and missing coarse class'})
    counts=Counter(r['wentworth_median_band'] or ('missing_D50' if r['lab_d50_mm'] is None else 'outside_sand_bands') for r in rows)
    report={'source':'https://doi.org/10.5066/P1AFJDNE','source_metadata':
        'https://cmgds.marine.usgs.gov/catalog/spcmsc/2023-309-FA_Sediment_metadata.faq.html',
        'license':'US Government public domain; attribution retained','photos':len(photos),'lab_records':len(rows),
        'rows_with_one_depth_matched_photo':sum(len(r['photo_members'])==1 for r in rows),
        'median_band_counts':dict(counts),'four_class_grain_model_ready':False,
        'non_sample_rows_excluded':excluded,
        'not_trained_reason':'No coarse-sand D50 examples, no verified pixel ROIs, unconsolidated sediment domain',
        'rows':rows}
    (folder/'usgs_grain_candidates.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
    out=ROOT/'research_reports/rock_grain_20261106';out.mkdir(exist_ok=True,parents=True)
    (out/'usgs_grain_audit.json').write_text(json.dumps({k:v for k,v in report.items() if k!='rows'},ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps({k:v for k,v in report.items() if k!='rows'},ensure_ascii=False))

if __name__=='__main__':main()
