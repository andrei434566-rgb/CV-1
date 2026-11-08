"""Non-deployed coal/non-coal research head with new, source-described core photos.

All photos below come from one well. A seam group is held out, not claimed to
be an unseen-well test. Primary classifier and active feature pointers are untouched.
"""
import hashlib,json,sys,io,random
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from torch import nn
from PIL import Image,ImageDraw
from torchvision import transforms
from app.infrastructure.local_vision import VisionEngine
from app.infrastructure.vision_models import MEAN,STD
from app.infrastructure.supervised_learning import classification_metrics,classifier_input_layer

SOURCE='https://www.frontiersin.org/journals/earth-science/articles/10.3389/feart.2025.1690793/full'
# Whole panel correspondence visually checked against caption; only rock ROIs.
# Fig 3d, 3f, 4b, 4l are mixed lithologies and are deliberately excluded.
PANELS=[
 ('3a','coal',4365.5,4365.59,'seam1','train',[90,100,180,140]),
 ('3b','carbonaceous_shale',4366.08,4366.33,'roof1','train',[365,80,410,140]),
 ('3c','mudstone',4367.20,4367.38,'lower_mud','train',[720,105,910,170]),
 ('3e','silty_mudstone',4372.57,4372.83,'lower_silty','train',[380,265,500,295]),
 ('4a','mudstone',4284.69,4284.87,'middle_mud1','train',[250,75,310,120]),
 ('4c','mudstone',4290.37,4290.55,'middle_mud2','train',[800,40,960,85]),
 ('4d','mudstone',4295.10,4295.30,'middle_mud3','train',[155,410,240,445]),
 ('4e','coal',4301.28,4301.36,'seam7','train',[500,270,580,355]),
 ('4f','coal',4303.48,4303.59,'seam7','train',[810,340,890,380]),
 ('4g','shale',4304.59,4304.72,'above_seam7','train',[240,690,325,720]),
 ('4h','coal',4324.48,4324.65,'seam4_and_roof','test',[470,620,625,695]),
 ('4i','coal',4324.48,4324.65,'seam4_and_roof','test',[835,570,910,625]),
 ('4j','carbonaceous_shale',4324.91,4325.12,'seam4_and_roof','test',[285,910,350,960]),
 ('4k','shale',4332.74,4332.94,'above_seam4','test',[535,835,625,880]),
]


def write(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf8')
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def metrics(logits,y):return classification_metrics(y.tolist(),logits.argmax(1).tolist(),['not_coal','coal'])


def prepare():
    source=ROOT/'open_datasets/rock_web_20261106';target=source/'nt1h_core_rois';target.mkdir(exist_ok=True)
    rows=[];sheet=Image.new('RGB',(700,600),'white');draw=ImageDraw.Draw(sheet)
    for i,(panel,lit,top,base,group,split,box) in enumerate(PANELS):
        figure=source/f'nt1h_fig{panel[0]}.jpg'
        with Image.open(figure) as im:crop=im.convert('RGB').crop(box)
        path=target/(panel+'.png');crop.save(path)
        rows.append({'id':'nt1h_'+panel,'well':'NT1H','depth_from_m':top,'depth_to_m':base,
                     'source_lithology':lit,'binary_label':int(lit=='coal'),'split':split,
                     'group':'NT1H_'+group,'image':str(path),'image_sha256':digest(path),'source_figure_sha256':digest(figure),
                     'figure':int(panel[0]),'panel':panel[1],'crop_box':box,'source':SOURCE,
                     'author':'Yao, Tian, Liu et al. (2025)','license':'CC BY 4.0',
                     'label_provenance':'exact author caption; manually checked panel and rock ROI',
                     'modification':'Rock ROI cropped; labels, arrows, ruler and red background excluded where possible',
                     'russian_ten_class_target_confirmed':False})
        thumb=crop.copy();thumb.thumbnail((130,100));x=(i%5)*140;y=(i//5)*200;sheet.paste(thumb,(x,y))
        draw.text((x,y+105),panel+' '+lit,fill='black');draw.text((x,y+130),split+' '+str(top),fill='black')
    assert not {r['group'] for r in rows if r['split']=='train'}&{r['group'] for r in rows if r['split']=='test'}
    write(source/'nt1h_core_manifest.json',{'source':SOURCE,'doi':'10.3389/feart.2025.1690793','license':'CC BY 4.0',
        'source_native_labels_only':True,'well_independent_test':False,'rows':rows,
        'excluded_panels':['3d mixed coal/shale','3f muddy siltstone-sandstone','4b coal and immediate roof','4l mixed siltstone-sandstone']})
    sheet.save(source/'nt1h_rois_review.jpg');return rows


def encode(engine,rows):
    transform=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),transforms.Normalize(MEAN,STD)])
    captured=[];hook=classifier_input_layer(engine.model,'convnext_tiny').register_forward_pre_hook(lambda m,v:captured.append(v[0].detach().float().cpu()))
    try:
        for r in rows:
            with Image.open(r['image']) as im:x=transform(im.convert('RGB'))[None].to('cuda')
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):engine.model(x)
    finally:hook.remove()
    return torch.cat(captured)


def main():
    rows=prepare()
    if '--prepare-only' in sys.argv:return
    out=ROOT/'models/features/coal_core_research_20261106'
    if out.exists():raise ValueError('Research version already exists')
    out.mkdir(parents=True)
    base=ROOT/'models/vision/vision_20261104_143516';before=digest(base/'model.pt')
    engine=VisionEngine(device='cuda');engine._load(base);torch.set_num_threads(4);torch.manual_seed(20261106)
    cache=ROOT/'models/vision/rock_taxonomy_convnext_20261106';features={}
    for split in ('train','val'):
        d=torch.load(cache/(split+'_features.pt'),weights_only=True)
        assert d['base_sha256']==before
        features[split]=(d['features'],(d['targets']==8).long())
    core=encode(engine,rows);train_idx=[i for i,r in enumerate(rows) if r['split']=='train'];test_idx=[i for i,r in enumerate(rows) if r['split']=='test']
    targets=torch.tensor([r['binary_label'] for r in rows])
    x=torch.cat([features['train'][0],core[train_idx]]).to('cuda');y=torch.cat([features['train'][1],targets[train_idx]]).to('cuda')
    vx,vy=[v.to('cuda') for v in features['val']]
    weights=torch.ones(len(y),device='cuda');weights[-len(train_idx):]=75.
    head=nn.Linear(x.shape[1],2).to('cuda');opt=torch.optim.AdamW(head.parameters(),lr=.002,weight_decay=.01)
    counts=torch.bincount(y,minlength=2).float();ce=nn.CrossEntropyLoss(weight=(counts.sum()/counts).sqrt(),reduction='none',label_smoothing=.02)
    best=-1.;history=[]
    for epoch in range(45):
        head.train()
        for ids in torch.randperm(len(y),device='cuda').split(256):
            opt.zero_grad(set_to_none=True);loss=(ce(head(x[ids]),y[ids])*weights[ids]).mean();loss.backward();opt.step()
        head.eval()
        with torch.inference_mode():val=metrics(head(vx),vy)
        history.append({'epoch':epoch+1,'validation':val})
        if val['macro_f1']>best:
            best=val['macro_f1'];best_validation=val;torch.save(head.state_dict(),out/'head.pt')
    head.load_state_dict(torch.load(out/'head.pt',weights_only=True));head.eval()
    with torch.inference_mode():
        logits=head(core[test_idx].to('cuda'));result=metrics(logits,targets[test_idx]);scores=torch.softmax(logits,1).cpu().tolist()
    assert digest(base/'model.pt')==before
    report={'schema':'cv1-coal-core-research-v1','generative':False,'base_model_sha256':before,
        'classes':['not_coal','coal'],'new_real_core_train':len(train_idx),'real_core_test':len(test_idx),
        'replay_train':len(features['train'][1]),'validation':best_validation,'test':result,
        'test_samples':[dict(rows[i],predicted_coal_score=scores[j][1]) for j,i in enumerate(test_idx)],
        'deployment_allowed':False,'well_independent_test':False,
        'limitations':'Only one new core well, four test panels; no generalisation claim. Coal/non-coal only; carbonaceous shale is not relabelled to a Russian class.',
        'primary_model_unchanged':True,'sources':[{'source':SOURCE,'license':'CC BY 4.0'},
            {'source':'https://github.com/JiayuLi1120/drill-core-image-dataset','license':'CC BY-NC 4.0'},
            {'source':'https://zenodo.org/records/10702879','license':'CC BY 4.0'}],'noncommercial_only':True}
    write(out/'config.json',report);write(out/'history.json',history)
    dest=ROOT/'research_reports/rock_grain_20261106';dest.mkdir(exist_ok=True,parents=True);write(dest/'coal_core_training.json',report)
    print(json.dumps({'done':True,'new_core_train':len(train_idx),'test_count':len(test_idx),'test':result,'primary_model_unchanged':True}),flush=True)

if __name__=='__main__':main()
