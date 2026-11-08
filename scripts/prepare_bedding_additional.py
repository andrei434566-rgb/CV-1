"""Additional author-captioned core photographs; insufficient sets stay pending.

No image search pseudo-labels, screenshots, sketches or thin sections become
photographic core examples. Whole publication/figure remains the split group.
"""
from pathlib import Path
import hashlib,json,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    from PIL import Image
    import requests
    root=ROOT/'open_datasets/bedding_20261106/additional';root.mkdir(parents=True,exist_ok=True)
    articles={
        'opalinus':{'citation':'Lauper et al. (2021), Frontiers in Earth Science 9:645596',
            'url':'https://www.frontiersin.org/journals/earth-science/articles/10.3389/feart.2021.645596/full',
            'license':'CC BY','rights_evidence':'Copyright section of article','file':'opalinus.html'},
        'plos':{'citation':'Abdlmutalib et al. (2025), PLOS ONE 20(7):e0327738',
            'url':'https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0327738',
            'license':'CC BY','rights_evidence':'Copyright section of article','file':'plos.html'},
        'iodp381':{'citation':'McNeill et al. (2019), IODP 381, Expedition 381 facies associations',
            'url':'https://publications.iodp.org/proceedings/381/103/381_103.html',
            'license':'CC BY 4.0','rights_evidence':'Chapter footer','file':'iodp381.html'},
    }
    downloads=[]
    def fetch(name,url,source):
        p=root/name
        if not p.exists():
            response=requests.get(url,timeout=(15,50));response.raise_for_status();p.write_bytes(response.content)
        downloads.append({'file':name,'url':url,'sha256':sha(p),'source':source,'bytes':p.stat().st_size})
        return p
    for key,row in articles.items():fetch(row['file'],row['url'],key)
    fetch('opalinus_fig2.webp','https://www.frontiersin.org/files/Articles/645596/xml-images/feart-09-645596-g002.webp','opalinus')
    fetch('plos_fig4.png','https://journals.plos.org/plosone/article/figure/image?size=large&id=10.1371/journal.pone.0327738.g004','plos')
    for n in (3,4,14,18):
        fetch(f'iodp381_F{n:02}.png',f'https://publications.iodp.org/proceedings/381/103/figures/381_103_F{n}.png','iodp381')
    specifications=[
        ('opalinus_sf2_lenticular.png','opalinus_fig2.webp',(480,656,880,980),'opalinus','Figure 2, macrofacies SF2',
         'SF2: argillaceous-siliceous fine to medium mudstone, characterized by lenticular bedding.',
         'Линзовидная','BDM-B2 or BGC-2 (individual image ID not provided)',
         'One macrofacies photograph; rounded bioturbational lenses also occur in this subfacies.'),
        ('opalinus_sf3_lenticular_wavy.png','opalinus_fig2.webp',(480,1128,875,1440),'opalinus','Figure 2, macrofacies SF3',
         'SF3: medium mudstone, characterized by lenticular to wavy bedding; often disturbed by bioturbation.',
         'Линзовидно-волнистая','BDM-B2 or BGC-2 (individual image ID not provided)',
         'One macrofacies photograph; mixed sedimentary and bioturbational features.'),
        ('plos_wavy_O_left.png','plos_fig4.png',(625,2306,885,2492),'plos','Figure 4O, left',
         'Wavy bedding: alternating mud and rippled sand layers.','Волнистая',None,
         'Same author/data family as RF100. Possible overlap with RF100; never an independent well test.'),
        ('plos_wavy_O_right.png','plos_fig4.png',(893,2306,1153,2492),'plos','Figure 4O, right',
         'Wavy bedding: alternating mud and rippled sand layers.','Волнистая',None,
         'Same author/data family as RF100. Possible overlap with RF100; never an independent well test.'),
        ('iodp_crosslamination_F18.png','iodp381_F18.png',(90,1385,255,1575),'iodp381','Figure F18, sandy interval 100-110 cm',
         'Sandy turbidite with cross-laminations, 381-M0078A-10H-1.',
         'Косая слоистость, подтип не уточнён','M0078A',
         'Author cross-lamination label does not establish large/small/trough/tabular subtype.'),
    ]
    rows=[]
    for name,file,box,key,figure,caption,label,well,limitation in specifications:
        with Image.open(root/file) as source:
            if not 0<=box[0]<box[2]<=source.width or not 0<=box[1]<box[3]<=source.height:raise ValueError('Bad source crop')
            source.convert('RGB').crop(box).save(root/name)
        rows.append({'image':str((root/name).relative_to(ROOT)),'sha256':sha(root/name),
            'source_image':file,'source_image_sha256':sha(root/file),'source_box':box,'source':key,'figure':figure,
            'author_description':caption,'candidate_label':label,'well':well,
            'split_group':key+':'+('figure2' if key=='opalinus' else 'figure4' if key=='plos' else 'M0078A'),
            'domain':'photographic_drill_core','description_origin':'author_text_and_labelled_panel',
            'limitation':limitation,'training_approved':False,
            'pending_reason':'Insufficient independent labelled examples/wells for a new validated class; no random crop split.'})
    inventory={'schema':'cv1-bedding-extra-material-v1','sources':articles,'downloads':downloads,'examples':rows,
       'theory_is_not_visual_training':True,
       'korea_core.pdf':{'doi':'10.14770/jgsk.2021.57.6.765','figure':'4, page 773',
          'captioned_structures':['lenticular bedding','wavy bedding','flaser bedding',
                                 'hummocky AND swaley cross stratification','trough cross-beds'],
          'training_approved':False,'reason':'No reuse license confirmed; compound captions need exact ROI separation.'},
       'iodp_context_figures':{'F3':'M0078A-36P-2, laminated and homogeneous mud intervals',
          'F4':'M0078A-32P-1, laminated mud and mud/sand beds',
          'F14':'M0080A-67R-2, fining-upward conglomerates; full package needs interval review',
          'F18':'M0078A-10H-1, sandy turbidite and homogeneous mud'},
       'excluded_domains':['texture diagrams','thin sections','classification colour overlays','legends']}
    (root/'additional_inventory.json').write_text(json.dumps(inventory,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps({'downloaded_files':len(downloads),'caption_matched_core_crops':len(rows),'new_examples_fitted':0,
                      'inventory':str(root/'additional_inventory.json')},ensure_ascii=False))

if __name__=='__main__':main()
