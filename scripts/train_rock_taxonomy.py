"""Noncommercial CNN research training on source-labelled DCID and Mendeley crops.

No generated labels, LLMs, artificial interbeds or renaming carbonaceous siltstone
to carbonaceous argillite. DCID has no borehole IDs: its test is NOT well blind.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import random
import sys
import time
import zipfile

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms
from PIL import Image
from app.infrastructure.vision_models import MEAN,STD,create_network,pretrained_network

CLASSES=('sandstone','siltstone','mudrock','crystalline','carbonaceous_siltstone','carbonate_other','breccia_other','background')
NAMES=('Песчаник','Алевролит','Аргиллит / глинистая порода','Магматические / метаморфические / железистые породы',
       'Углистый алевролит · вне целевого списка','Карбонатная порода · вне целевого списка','Брекчия / обломки · вне целевого списка','Фон / не керн')
SEED=20261105


def write(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8');tmp.replace(path)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()


def source_group(number):
    if number in (1,2,3):return 'sandstone'
    if number in (4,5):return 'siltstone'
    if number==6:return 'carbonaceous_siltstone'
    if number in (7,20,21):return 'mudrock'
    if number in (17,18,19):return 'carbonate_other'
    if number in (29,30,34,35):return 'breccia_other'
    if number in (*range(8,17),*range(22,29),31,32,33):return 'crystalline'
    raise ValueError(number)


def prepare(out):
    archive=ROOT/'open_datasets/dcid/archives/DCID.zip'
    candidates={split:[] for split in ('train','val','test')}
    by_native={};seen={};duplicates=[];corrupt=[];conflicting=set()
    with zipfile.ZipFile(archive) as z:
        for member in z.namelist():
            parts=member.split('/')
            if len(parts)!=5 or parts[:2]!=['DCID','DCID-512-35'] or parts[2] not in ('train','test') or not member.lower().endswith(('.jpg','.jpeg','.png')):continue
            number=int(parts[3].split('.')[0]);label=source_group(number)
            by_native.setdefault((parts[2],parts[3]),[]).append((member,label))
        for (original,native),items in sorted(by_native.items()):
            items=sorted(items);random.Random(str(SEED)+original+native).shuffle(items)
            # Hold out a part of the author's train before fitting; official test
            # remains separate. No synthetic/noisy or duplicate DCID-7 variants.
            for i,(member,label) in enumerate(items):
                split='test' if original=='test' else 'val' if i<160 else 'train'
                if split=='train' and i>=560:continue # 400 original train crops/native class
                try:
                    blob=z.read(member)
                    with Image.open(io.BytesIO(blob)) as image:image.verify()
                except (zipfile.BadZipFile,OSError,ValueError) as exc:
                    corrupt.append({'member':member,'reason':str(exc)});continue
                digest=hashlib.sha256(blob).hexdigest()
                if digest in seen:
                    duplicates.append({'member':member,'same_as':seen[digest]});continue
                seen[digest]={'member':member,'label':CLASSES.index(label)}
                candidates[split].append({'archive':str(archive),'member':member,'label':CLASSES.index(label),
                                          'source_label':native,'source':'dcid35','source_sha256':digest,'well':None})
    data=ROOT.parent/'vlm_core_v1/data'
    aliases={'sandstone':'sandstone','shale':'mudrock','limestone':'carbonate_other','garbage':'background'}
    for split,file in [('train','mendeley_lithology_4000/train.jsonl'),('val','mendeley_lithology_pretrained_holdout/val.jsonl'),('test','mendeley_lithology_pretrained_holdout/test.jsonl')]:
        for line in (data/file).read_text(encoding='utf-8-sig').splitlines():
            row=json.loads(line);m=row['metadata'];assert m['label_provenance']=='source_direct_folder'
            if split=='train':assert m['training_approved'] is True
            label=aliases[m['source_label_en']];path=Path(row['image']).resolve()
            if not path.is_relative_to(ROOT.parent.resolve()):raise ValueError('Input outside working copy on C')
            digest=sha(path)
            if digest!=m['image_sha256']:raise ValueError('Changed Mendeley image')
            if digest in seen:
                duplicates.append({'member':str(path),'same_as':seen[digest]});continue
            seen[digest]={'member':str(path),'label':CLASSES.index(label)}
            candidates[split].append({'image':str(path),'label':CLASSES.index(label),'source_label':m['source_label_en'],
                                      'source':'mendeley','source_sha256':digest,'well':m['well_id']})
    audit={'schema':'cv1-source-rocks-audit-v1','source_licences':{'dcid35':'CC BY-NC 4.0','mendeley':'CC BY 4.0'},
           'scope':'noncommercial research explicitly confirmed by user','all_labels':'source_direct_classes',
           'excluded':['DCID-7 duplicate subset','all noise/RWDA variants','synthetic geology','unreviewed R-281 labels','assumed facies'],
           'duplicates_excluded':duplicates,'corrupt_files_excluded':corrupt,'dcid_well_ids_available':False,
           'validation_limitation':'DCID split is crop-level; original well IDs absent. Generalisation to other wells is not demonstrated.',
           'mendeley_limitation':'Existing research benchmark reused; not a newly sealed test.',
           'source_mapping':{native:source_group(int(native.split('.')[0])) for _,native in by_native},
           'classes':list(CLASSES),'missing_requested_classes':['Гравелит','Уголь','Углистый аргилит','Битуминозный и окремненный аргилит'],
           'interbedding':'Requires along-column context and verified lithology proportions; no synthetic training labels generated.'}
    for split,rows in candidates.items():
        audit[split]={'count':len(rows),'by_class':dict(Counter(CLASSES[r['label']] for r in rows)),
                      'by_source':dict(Counter(r['source'] for r in rows)),
                      'mendeley_wells':sorted({r['well'] for r in rows if r['well']})}
    for a,b in [('train','val'),('train','test'),('val','test')]:
        assert not {r['source_sha256'] for r in candidates[a]}&{r['source_sha256'] for r in candidates[b]}
        assert not set(audit[a]['mendeley_wells'])&set(audit[b]['mendeley_wells'])
    write(out/'dataset.json',candidates);write(out/'data_audit.json',audit)
    return candidates,audit


class Images(Dataset):
    def __init__(self,rows,training=False):
        self.rows=rows;self.archives={}
        ops=[transforms.Resize((224,224))]
        if training:ops += [transforms.RandomHorizontalFlip(),transforms.ColorJitter(brightness=.08,contrast=.08,saturation=.04)]
        self.transform=transforms.Compose(ops+[transforms.ToTensor(),transforms.Normalize(MEAN,STD)])
    def __len__(self):return len(self.rows)
    def __getitem__(self,index):
        row=self.rows[index]
        if 'archive' in row:
            if row['archive'] not in self.archives:self.archives[row['archive']]=zipfile.ZipFile(row['archive'])
            file=io.BytesIO(self.archives[row['archive']].read(row['member']))
        else:file=row['image']
        with Image.open(file) as image:x=self.transform(image.convert('RGB'))
        return x,row['label']


def init_worker(_):torch.set_num_threads(1)


def loader(rows,batch,workers,training=False):
    return DataLoader(Images(rows,training),batch_size=batch,shuffle=training,num_workers=workers,
                      persistent_workers=workers>0,pin_memory=True,worker_init_fn=init_worker,
                      generator=torch.Generator().manual_seed(SEED))


def metrics(target,predicted):
    from app.infrastructure.supervised_learning import classification_metrics
    return classification_metrics(target,predicted,list(CLASSES))


def evaluate(model,ld,temperature=1.):
    model.eval();targets=[];predicted=[];logits_list=[]
    with torch.inference_mode():
        for x,y in ld:
            with torch.autocast('cuda',dtype=torch.float16):logits=model(x.to('cuda',non_blocking=True,memory_format=torch.channels_last))
            logits_list.append(logits.float().cpu());targets.extend(y.tolist());predicted.extend(logits.argmax(1).cpu().tolist())
    return metrics(targets,predicted),torch.cat(logits_list),torch.tensor(targets)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--epochs',type=int,default=6);parser.add_argument('--batch',type=int,default=24)
    parser.add_argument('--workers',type=int,default=2);parser.add_argument('--noncommercial',action='store_true')
    args=parser.parse_args()
    if not args.noncommercial:raise ValueError('DCID requires confirmed noncommercial research scope')
    if not torch.cuda.is_available():raise RuntimeError('CUDA is required for this research training script')
    out=args.output.resolve()
    if not out.is_relative_to(ROOT.resolve()):raise ValueError('Output must stay within editable C project; E backup is protected')
    if out.exists() and any(out.iterdir()):raise ValueError('Output run already contains data; choose a new version')
    out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4);torch.manual_seed(SEED);np.random.seed(SEED);random.seed(SEED)
    torch.backends.cudnn.benchmark=True
    started=time.monotonic();print('Preparing source-labelled crops...',flush=True)
    parts,audit=prepare(out)
    print(json.dumps({s:audit[s]['count'] for s in parts}),flush=True)
    model=pretrained_network('efficientnet_b0',ROOT/'models/vision/pretrained')
    model.classifier[1]=nn.Linear(model.classifier[1].in_features,len(CLASSES))
    model=model.to('cuda',memory_format=torch.channels_last)
    train=loader(parts['train'],args.batch,args.workers,True);val=loader(parts['val'],args.batch,args.workers)
    counts=torch.bincount(torch.tensor([r['label'] for r in parts['train']]),minlength=len(CLASSES)).float()
    criterion=nn.CrossEntropyLoss(weight=(counts.sum()/counts).sqrt().to('cuda'),label_smoothing=.03)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.00015,weight_decay=.01)
    scaler=torch.amp.GradScaler('cuda');best=-1.;history=[]
    for epoch in range(args.epochs):
        model.train();loss_sum=0.;seen=0;epoch_started=time.monotonic()
        for step,(x,y) in enumerate(train):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.float16):
                logits=model(x.to('cuda',non_blocking=True,memory_format=torch.channels_last));loss=criterion(logits,y.to('cuda',non_blocking=True))
            if not torch.isfinite(loss):raise RuntimeError('Non-finite training loss')
            scaler.scale(loss).backward();scaler.unscale_(optimizer);nn.utils.clip_grad_norm_(model.parameters(),1.)
            scaler.step(optimizer);scaler.update();loss_sum+=float(loss.detach())*len(y);seen+=len(y)
            if step%200==0:print(json.dumps({'epoch':epoch+1,'step':step,'seen':seen,'loss':round(loss_sum/seen,5)}),flush=True)
        result,_,_=evaluate(model,val)
        row={'epoch':epoch+1,'loss':loss_sum/seen,'validation':result,'seconds':time.monotonic()-epoch_started}
        history.append(row);write(out/'history.json',history)
        if result['macro_f1']>best:
            best=result['macro_f1'];torch.save({k:v.cpu() for k,v in model.state_dict().items()},out/'model.pt');write(out/'best_validation.json',row)
        print(json.dumps({'epoch':epoch+1,'val_macro_f1':result['macro_f1'],'seconds':row['seconds']}),flush=True)
    model.load_state_dict(torch.load(out/'model.pt',map_location='cpu',weights_only=True))
    result,logits,targets=evaluate(model,val)
    # Pick temperature on validation only. It changes scores, not class argmax.
    candidates=[.8,1.,1.2,1.5,2.,2.5,3.]
    temperature=min(candidates,key=lambda t:float(nn.functional.cross_entropy(logits/t,targets)))
    test=loader(parts['test'],args.batch,args.workers);test_metrics,test_logits,test_y=evaluate(model,test)
    torch.save({'logits':test_logits,'targets':test_y},out/'test_logits.pt')
    write(out/'test_metrics.json',test_metrics)
    card={'schema':'cv1-discriminative-vision-v2','generative':False,'architecture':'efficientnet_b0',
          'classes':list(CLASSES),'class_names_ru':list(NAMES),'model_sha256':sha(out/'model.pt'),
          'temperature':temperature,'review_threshold':.85,'margin_threshold':.20,'background_class':'background',
          'licence':'Research only: includes CC BY-NC 4.0 data','noncommercial_only':True,
          'sources':[{'name':'DCID-35','url':'https://huggingface.co/datasets/168sir/drill-core-image-dataset','license':'CC BY-NC 4.0'},
                     {'name':'Mendeley Lithology','url':'https://data.mendeley.com/datasets/b6rj29mrss/1','license':'CC BY 4.0'}],
          'validation':result,'test':test_metrics,'well_independent_test':False,'limitations':audit['validation_limitation'],
          'requested_classes_missing_direct_examples':audit['missing_requested_classes'],'interbedding_trained':False,
          'crystalline_is_not_proof_of_basement':True,'elapsed_seconds':round(time.monotonic()-started,1)}
    write(out/'config.json',card)
    print(json.dumps({'done':True,'test_macro_f1':test_metrics['macro_f1'],'model':str(out),'seconds':card['elapsed_seconds']}),flush=True)


if __name__=='__main__':main()
