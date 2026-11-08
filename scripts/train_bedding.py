"""Source-labelled RF100 core structures. Separate non-generative CNN research.

Only files with explicit well IDs enter this experiment. Entire wells, not
neighbouring crops, are held out. Unknown-well images never enter fitting.
"""
from __future__ import annotations
import argparse, hashlib, json, random, re, sys, time
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from PIL import Image, ImageOps
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from app.infrastructure.vision_models import MEAN, STD, create_network, pretrained_network

CLASSES=('cross_bedding','low_angle','massive','parallel_lamination','mud_drape')
NAMES=('Косая слоистость, подтип не уточнён','Пологонаклонная','Массивная структура · вне списка слоистости',
       'Параллельная слоистость','Глинистая драпировка · вне списка слоистости')
SEED=20261106
VAL_WELLS={'10-17-27-27W4'}
TEST_WELLS={'09-15-25-26W4'}

def write(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def prepare(out):
    source=ROOT/'open_datasets/bedding_20261106/rf100'
    manifest=json.loads((source/'source_manifest.json').read_text(encoding='utf8'))
    expected={v['path']:v['sha256'] for v in manifest['files']}
    rows={s:[] for s in ('train','val','test')};unknown=[];excluded=[];images={};duplicates=[]
    for p in sorted(source.glob('*/images/*')):
        if sha(p)!=expected[p.relative_to(source).as_posix()]:raise ValueError('Changed source image')
        labelpath=p.parent.parent/'labels'/f'{p.stem}.txt'
        if sha(labelpath)!=expected[labelpath.relative_to(source).as_posix()]:raise ValueError('Changed annotation')
        m=re.match(r'^(\d\d-\d\d-\d\d-\d\dW\d)',p.stem,re.I)
        if not m:unknown.append(str(p.relative_to(ROOT)));continue
        well=m[1];split='test' if well in TEST_WELLS else 'val' if well in VAL_WELLS else 'train'
        with Image.open(p) as original:im=original.convert('RGB');w,h=im.size
        digest=hashlib.sha256(im.tobytes()).hexdigest()
        if digest in images:
            if images[digest]['well']!=well:raise ValueError('Same image assigned to different wells')
            duplicates.append({'image':str(p),'same_as':images[digest]});continue
        images[digest]={'image':str(p.relative_to(ROOT)),'well':well,'split':split}
        for i,line in enumerate(labelpath.read_text().splitlines()):
            if not line.strip():continue
            k,x,y,bw,bh=map(float,line.split());k=int(k)
            if not 0<=k<len(CLASSES):raise ValueError('Unknown class')
            box=[max(0,round((x-bw/2)*w)),max(0,round((y-bh/2)*h)),
                 min(w,round((x+bw/2)*w)),min(h,round((y+bh/2)*h))]
            if min(box[2]-box[0],box[3]-box[1])<8 or (box[2]-box[0])*(box[3]-box[1])<256:
                excluded.append({'image':str(p.relative_to(ROOT)),'line':i,'reason':'ROI too small','label':k});continue
            crop=im.crop(box);crop_sha=hashlib.sha256(crop.tobytes()+str(crop.size).encode()).hexdigest()
            rows[split].append({'image':str(p.relative_to(ROOT)),'image_sha256':sha(p),'pixel_sha256':digest,
                                'crop_sha256':crop_sha,'box':box,'label':k,'well':well,'annotation_line':i,
                                'source_annotation':str(labelpath.relative_to(ROOT)),
                                'label_origin':'author_yolo_bbox','source_domain':'photographic_drill_core'})
    # Exact crop duplicates across splits are rejected, not silently assigned.
    seen={}
    for split,values in rows.items():
        unique=[]
        for row in values:
            key=row['crop_sha256']
            if key in seen:
                if seen[key]['split']!=split:raise ValueError('Crop leakage between wells')
                if seen[key]['label']!=row['label']:raise ValueError('Conflicting identical ROI labels')
                duplicates.append({'crop':row,'same_as':seen[key]});continue
            seen[key]=dict(row,split=split);unique.append(row)
        rows[split]=unique
    audit={'schema':'cv1-bedding-source-audit-v1','source':manifest,'seed':SEED,
           'split_policy':'explicit known well ID; frozen val 10-17-27-27W4, test 09-15-25-26W4 before fitting',
           'unknown_well_images_excluded':unknown,'excluded_small_rois':excluded,'duplicates_excluded':duplicates,
           'source_limitation':'Author stretch-resized images to 640x640. Physical angles and sizes cannot be recovered.',
           'domain_limitation':'Single author dataset; only 1 validation well and 1 test well. Not a 16-class validation.',
           'native_classes':CLASSES,'generative':False}
    for split,values in rows.items():
        counts=Counter(CLASSES[r['label']] for r in values)
        if len(counts)!=len(CLASSES):raise ValueError('Missing class in '+split)
        audit[split]={'n':len(values),'classes':dict(counts),'wells':sorted({r['well'] for r in values}),
                      'images':len({r['image'] for r in values})}
    write(out/'dataset.json',rows);write(out/'data_audit.json',audit)
    return rows

class Images(Dataset):
    def __init__(self,rows,training=False):
        self.rows=rows;self.images=[]
        for row in rows:
            with Image.open(ROOT/row['image']) as im:
                self.images.append(ImageOps.pad(im.convert('RGB').crop(row['box']),(224,224),
                    method=Image.Resampling.BILINEAR,color=(127,127,127)))
        operations=([transforms.RandomHorizontalFlip(),transforms.ColorJitter(.08,.08,.04)] if training else [])
        self.transform=transforms.Compose(operations+[transforms.ToTensor(),transforms.Normalize(MEAN,STD)])
    def __len__(self):return len(self.rows)
    def __getitem__(self,i):return self.transform(self.images[i]),self.rows[i]['label']

def metrics(y,p):
    from app.infrastructure.supervised_learning import classification_metrics
    return classification_metrics(y,p,list(CLASSES))

def evaluate(model,loader,device):
    model.eval();ys=[];ps=[];logits=[]
    with torch.inference_mode():
        for x,y in loader:
            with torch.autocast(device,dtype=torch.float16,enabled=device=='cuda'):z=model(x.to(device))
            logits.append(z.float().cpu());ys+=y.tolist();ps+=z.argmax(1).cpu().tolist()
    return metrics(ys,ps),torch.cat(logits),torch.tensor(ys)

def main():
    p=argparse.ArgumentParser();p.add_argument('--epochs',type=int,default=15);p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--architecture',choices=('efficientnet_b0','convnext_tiny'),default='efficientnet_b0')
    p.add_argument('--output',type=Path,default=ROOT/'models/bedding/rf100_b0_20261106');args=p.parse_args()
    out=args.output.resolve()
    if not out.is_relative_to(ROOT.resolve()):raise ValueError('Output must stay in editable C workspace')
    if (out/'model.pt').exists():raise ValueError('Choose a new version; never overwrite trained weights')
    rows=prepare(out);print(json.dumps({s:dict(Counter(CLASSES[r['label']] for r in v)) for s,v in rows.items()}),flush=True)
    if args.prepare_only:return
    random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.set_num_threads(4)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    loaders={s:DataLoader(Images(v,s=='train'),batch_size=32,shuffle=s=='train',num_workers=0,
                          generator=torch.Generator().manual_seed(SEED)) for s,v in rows.items()}
    model=pretrained_network(args.architecture,ROOT/'models/vision/pretrained')
    layer=1 if args.architecture=='efficientnet_b0' else 2
    model.classifier[layer]=nn.Linear(model.classifier[layer].in_features,len(CLASSES));model=model.to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=.01)
    counts=torch.bincount(torch.tensor([r['label'] for r in rows['train']]),minlength=len(CLASSES)).float()
    weights=(counts.sum()/counts).sqrt();weights/=weights.mean()
    loss_fn=nn.CrossEntropyLoss(weight=weights.to(device))
    scaler=torch.amp.GradScaler(device,enabled=device=='cuda');history=[];best=-1;started=time.perf_counter()
    for epoch in range(args.epochs):
        model.train();total=0.;n=0
        for x,y in loaders['train']:
            x,y=x.to(device),y.to(device);optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device,dtype=torch.float16,enabled=device=='cuda'):z=model(x);loss=loss_fn(z,y)
            if not torch.isfinite(loss):raise RuntimeError('Non-finite training loss')
            scaler.scale(loss).backward();scaler.step(optimizer);scaler.update();total+=loss.item()*len(y);n+=len(y)
        val,_,_=evaluate(model,loaders['val'],device)
        record={'epoch':epoch+1,'loss':total/n,'validation':val,'elapsed_seconds':time.perf_counter()-started};history.append(record)
        if val['macro_f1']>best:
            best=val['macro_f1'];torch.save({k:v.detach().cpu() for k,v in model.state_dict().items()},out/'model.pt');best_epoch=epoch+1
        write(out/'history.json',history)
        print(json.dumps(record,ensure_ascii=False),flush=True)
    model.load_state_dict(torch.load(out/'model.pt',map_location=device,weights_only=True))
    # Test is opened only after the training schedule and best validation epoch are fixed.
    val,_,_=evaluate(model,loaders['val'],device);test,z,y=evaluate(model,loaders['test'],device)
    write(out/'test_predictions.json',[dict(row,probabilities=prob.tolist()) for row,prob in zip(rows['test'],z.softmax(1))])
    card={'schema':'cv1-bedding-cnn-v1','architecture':args.architecture,'generative':False,
          'native_classes':CLASSES,'names_ru':NAMES,'model_sha256':sha(out/'model.pt'),'input_size':224,
          'preprocessing':'whole_roi_rgb_letterbox_224_gray127_imagenet_v1','license':'CC BY 4.0',
          'source':'RF100 sedimentary-features-9eosf / Ammar Abdlmutalib',
          'best_epoch':best_epoch,'epochs':args.epochs,'validation':val,'test':test,
          'deployment':'research_only','requested_16_classes_trained':False,
          'notes':'Native 5-way ROI classifier; no detector, no absolute dip/scale, no mapping to large/trough/tabular cross-bedding.',
          'elapsed_seconds':time.perf_counter()-started,'device':device}
    write(out/'config.json',card);print(json.dumps(card,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
