"""Train two CNNs on identical approved lithology data and select on validation.

Test and Russian R-41 transfer data are evaluated only after both training runs.
No facies targets, synthetic attributes, LLM/VLM weights or generated labels.
"""
from __future__ import annotations
import argparse, collections, hashlib, json, math, os, random, shutil, sys, time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
DATA=ROOT.parent/'vlm_core_v1/data'
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from app.infrastructure.vision_models import LABELS, LABELS_RU, MEAN, STD, pretrained_network, create_network

SEED=20261104
def read(p):return [json.loads(s) for s in p.read_text(encoding='utf-8-sig').splitlines() if s.strip()]
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(2*1024*1024),b''):h.update(b)
    return h.hexdigest()
def write(p,value):
    p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
    tmp.replace(p)

def prepare(out):
    sources={'train':['mendeley_lithology_4000/train.jsonl','mendeley_lithology_next/train.jsonl',
                      'mendeley_lithology_40000/additional_train.jsonl'],
             'val':['mendeley_lithology_pretrained_holdout/val.jsonl'],
             'test':['mendeley_lithology_pretrained_holdout/test.jsonl']}
    parts={};audit={'sources':{},'policy':'source_direct_folder; no synthetic or model-generated labels',
                   'facies_trained':False,'facies_reason':'No genuine confirmed facies labels on this computer',
                   'test_limit':'Existing research benchmark reused; well-disjoint from these training runs, not a new sealed test'}
    for split,names in sources.items():
        part=[]
        for name in names:
            audit['sources'][name]=sha(DATA/name)
            for r in read(DATA/name):
                m=r['metadata'];p=Path(r['image']).resolve()
                assert p.is_relative_to(ROOT.parent.resolve()),'Only working-copy images on C are permitted'
                assert m['label_provenance']=='source_direct_folder' and m['source_label_en'] in LABELS
                if split=='train':assert m['training_approved'] is True
                assert p.is_file(),str(p)
                part.append({'image':str(p),'label':LABELS.index(m['source_label_en']),
                             'well':m['well_id'],'sha256':m['image_sha256']})
        assert len(part)==(40000 if split=='train' else 400), (split,len(part))
        assert len({r['sha256'] for r in part})==len(part),'Duplicate image content'
        parts[split]=part
        audit[split]={'count':len(part),'by_class':dict(collections.Counter(LABELS[r['label']] for r in part)),
                      'wells':sorted({r['well'] for r in part})}
    for a,b in [('train','val'),('train','test'),('val','test')]:
        assert not ({r['well'] for r in parts[a]} & {r['well'] for r in parts[b]}),'Well leakage'
        assert not ({r['sha256'] for r in parts[a]} & {r['sha256'] for r in parts[b]}),'Image leakage'
    audit_file=out/'data_audit.json'
    if not audit_file.is_file():
        started=time.monotonic()
        for split,part in parts.items():
            for i,r in enumerate(part):
                assert sha(r['image'])==r['sha256'],'Source image changed: '+r['image']
                if i%4000==0:print(json.dumps({'stage':'data_sha256','split':split,'done':i,'total':len(part)}),flush=True)
        audit['all_image_files_sha256_verified']=True
        audit['verification_seconds']=round(time.monotonic()-started,1)
        write(audit_file,audit)
    else:
        old=json.loads(audit_file.read_text(encoding='utf-8'))
        assert old['sources']==audit['sources'] and old['all_image_files_sha256_verified']
    return parts,audit

class Images(Dataset):
    def __init__(self,rows,training=False):
        self.rows=rows
        ops=[transforms.Resize((224,224))]
        if training:ops += [transforms.RandomHorizontalFlip(),transforms.ColorJitter(brightness=.08,contrast=.08,saturation=.04)]
        self.transform=transforms.Compose(ops+[transforms.ToTensor(),transforms.Normalize(MEAN,STD)])
    def __len__(self):return len(self.rows)
    def __getitem__(self,i):
        with Image.open(self.rows[i]['image']) as im: x=self.transform(im.convert('RGB'))
        return x,self.rows[i]['label']

def worker_seed(_):
    torch.set_num_threads(1)
    random.seed(torch.initial_seed()%2**32);np.random.seed(torch.initial_seed()%2**32)
def loader(rows,batch,workers,training=False):
    return DataLoader(Images(rows,training),batch_size=batch,shuffle=training,num_workers=workers,
                      pin_memory=True,persistent_workers=workers>0,worker_init_fn=worker_seed,
                      generator=torch.Generator().manual_seed(SEED))
def metrics(target,pred):
    cm=np.zeros((4,4),dtype=int)
    for a,b in zip(target,pred):cm[int(a),int(b)]+=1
    per={};f=[]
    for c,name in enumerate(LABELS):
        tp=int(cm[c,c]);n=int(cm[c].sum());den=int(cm[c].sum()+cm[:,c].sum())
        score=2*tp/den if den else 0.;f.append(score)
        per[name]={'support':n,'correct':tp,'recall':tp/n if n else None,'f1':score}
    return {'count':int(cm.sum()),'accuracy':float(np.trace(cm)/max(1,cm.sum())),
            'macro_f1':float(np.mean(f)),'per_class':per,'confusion_matrix':cm.tolist(),'labels':list(LABELS)}
@torch.inference_mode()
def evaluate(model,ld):
    model.eval();ys=[];logs=[]
    for x,y in ld:
        with torch.autocast('cuda',dtype=torch.float16): z=model(x.to('cuda',non_blocking=True).to(memory_format=torch.channels_last))
        ys.extend(y.tolist());logs.extend(z.float().cpu().tolist())
    z=np.asarray(logs);return metrics(ys,z.argmax(1)),ys,logs

def train(name,parts,args):
    out=args.output/name;out.mkdir(parents=True,exist_ok=True)
    if (out/'complete.json').is_file():return json.loads((out/'complete.json').read_text(encoding='utf-8'))
    random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
    model=pretrained_network(name,ROOT/'models/vision/pretrained').to('cuda',memory_format=torch.channels_last)
    opt=torch.optim.AdamW([{'params':model.features.parameters(),'lr':3e-5},
                           {'params':model.classifier.parameters(),'lr':3e-4}],weight_decay=1e-4)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=args.epochs)
    scaler=torch.amp.GradScaler('cuda');loss_fn=nn.CrossEntropyLoss(label_smoothing=.02)
    train_ld=loader(parts['train'],args.batch,args.workers,True);val_ld=loader(parts['val'],args.batch,args.workers)
    history=[];best=-1.;best_epoch=0;start_epoch=1
    if (out/'resume.pt').is_file():
        state=torch.load(out/'resume.pt',map_location='cpu',weights_only=True)
        model.load_state_dict(state['model']);opt.load_state_dict(state['optimizer']);sched.load_state_dict(state['scheduler']);scaler.load_state_dict(state['scaler'])
        start_epoch=state['epoch']+1;history=state['history'];best=state['best'];best_epoch=state['best_epoch']
    total_started=time.monotonic()
    for epoch in range(start_epoch,args.epochs+1):
        torch.manual_seed(SEED+epoch);model.train();started=time.monotonic();last=started;seen=0;loss_sum=0.
        opt.zero_grad(set_to_none=True)
        for step,(x,y) in enumerate(train_ld):
            x=x.to('cuda',non_blocking=True).to(memory_format=torch.channels_last);y=y.to('cuda',non_blocking=True)
            with torch.autocast('cuda',dtype=torch.float16):
                raw_loss=loss_fn(model(x),y);loss=raw_loss/args.accumulation
            scaler.scale(loss).backward()
            if (step+1)%args.accumulation==0 or step+1==len(train_ld):
                scaler.unscale_(opt);torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                scaler.step(opt);scaler.update();opt.zero_grad(set_to_none=True)
            seen+=len(y);loss_sum+=float(raw_loss.detach())*len(y)
            if time.monotonic()-last>15:
                progress={'stage':'training','architecture':name,'epoch':epoch,'done':seen,'total':len(parts['train']),
                          'loss':loss_sum/seen,'seconds':round(time.monotonic()-started,1)}
                print(json.dumps(progress),flush=True);write(args.output/'progress.json',progress);last=time.monotonic()
        measured,ys,logits=evaluate(model,val_ld);sched.step()
        record={'epoch':epoch,'train_loss':loss_sum/seen,'validation':measured,'seconds':round(time.monotonic()-started,1)}
        history.append(record)
        if measured['macro_f1']>best:
            best=measured['macro_f1'];best_epoch=epoch
            torch.save({k:v.detach().cpu() for k,v in model.state_dict().items()},out/'model.pt')
            write(out/'validation_predictions.json',{'targets':ys,'logits':logits})
        write(out/'history.json',{'architecture':name,'epochs':history,'best_epoch':best_epoch,'best_validation_macro_f1':best,
                                 'test_used_for_selection':False,'seed':SEED,'settings':vars(args)})
        torch.save({'model':model.state_dict(),'optimizer':opt.state_dict(),'scheduler':sched.state_dict(),
                    'scaler':scaler.state_dict(),'epoch':epoch,'history':history,'best':best,'best_epoch':best_epoch},out/'resume.pt')
        print(json.dumps({'stage':'epoch_complete','architecture':name,**record,'best_epoch':best_epoch}),flush=True)
        if epoch>=3 and epoch-best_epoch>=args.patience:break
    complete={'architecture':name,'best_epoch':best_epoch,'validation_macro_f1':best,
              'model_sha256':sha(out/'model.pt'),'seconds_this_run':round(time.monotonic()-total_started,1)}
    write(out/'complete.json',complete)
    del model,opt,train_ld,val_ld;torch.cuda.empty_cache()
    return complete

@torch.inference_mode()
def benchmark(model):
    result={};model.eval()
    for batch in (1,8):
        x=torch.rand(batch,3,224,224,device='cuda').to(memory_format=torch.channels_last)
        with torch.autocast('cuda',dtype=torch.float16):
            for _ in range(8):model(x)
            torch.cuda.synchronize();times=[]
            for _ in range(30):
                start=time.perf_counter();model(x);torch.cuda.synchronize();times.append((time.perf_counter()-start)*1000)
        result[str(batch)]={'p50_ms':float(np.percentile(times,50)),'p95_ms':float(np.percentile(times,95))}
    return result

def finish(parts,args,audit):
    completions=[json.loads((args.output/n/'complete.json').read_text(encoding='utf-8')) for n in ('efficientnet_b0','convnext_tiny')]
    winner=max(completions,key=lambda r:r['validation_macro_f1'])['architecture']
    reports={}
    for name in ('efficientnet_b0','convnext_tiny'):
        model=create_network(name).to('cuda',memory_format=torch.channels_last)
        model.load_state_dict(torch.load(args.output/name/'model.pt',map_location='cpu',weights_only=True))
        measured,y,z=evaluate(model,loader(parts['test'],args.batch,args.workers))
        reports[name]={'selection':next(c for c in completions if c['architecture']==name),
                       'test':measured,'gpu_inference':benchmark(model)}
        write(args.output/name/'test_predictions.json',{'targets':y,'logits':z})
        del model;torch.cuda.empty_cache()
    vp=json.loads((args.output/winner/'validation_predictions.json').read_text(encoding='utf-8'))
    logits=torch.tensor(vp['logits']);targets=torch.tensor(vp['targets'])
    # Temperature is calibrated only on validation; it is not a geological confidence guarantee.
    choices=np.geomspace(.5,4.,50);temperature=min(choices,key=lambda t:float(nn.functional.cross_entropy(logits/float(t),targets)))
    card={'schema':'cv1-discriminative-vision-v1','engine':'vision','generative':False,'architecture':winner,
          'classes':list(LABELS),'classes_ru':list(LABELS_RU),'image_side':224,'normalization':{'mean':MEAN,'std':STD},
          'temperature':float(temperature),'review_threshold':.8,'margin_threshold':.15,
          'facies_trained':False,'texture_trained':False,'grain_size_trained':False,
          'scope':'Four source-labelled small-crop lithology classes; not facies, grain size, mineralogy or depositional environment',
          'source_dataset':'https://data.mendeley.com/datasets/b6rj29mrss/1','source_license':'CC BY 4.0',
          'attribution':'Alzubaidi et al., Lithology dataset, Mendeley Data v1; see source dataset citation',
          'selected_by':'validation_macro_f1_only','validation':reports[winner]['selection'],
          'test':reports[winner]['test'],'model_sha256':sha(args.output/winner/'model.pt'),
          'limitations':['Four lithology classes only. Other rocks may be out of distribution.',
                         'Softmax confidence is not proof of geological correctness.',
                         'No new independently collected blind wells; existing benchmark reused.',
                         'Synthetic facies and non-deployable facies research datasets excluded.']}
    publish=ROOT/'models/vision'/('vision_'+time.strftime('%Y%m%d_%H%M%S'))
    publish.mkdir(parents=True);shutil.copy2(args.output/winner/'model.pt',publish/'model.pt');write(publish/'config.json',card)
    assert sha(publish/'model.pt')==card['model_sha256']
    write(ROOT/'models/vision/current.json',{'folder':publish.name,'architecture':winner,'model_sha256':card['model_sha256']})
    comparison={'winner':winner,'selection_rule':'Highest validation macro-F1; test evaluated after checkpoint and architecture selection',
                'models':reports,'data_audit':audit,'published_folder':str(publish),'model_card':card}
    write(args.output/'comparison.json',comparison)
    write(ROOT/'research_reports/vision_20261104/comparison.json',comparison)
    print(json.dumps({'stage':'published','winner':winner,'models':reports,'folder':str(publish)}),flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--epochs',type=int,default=6);p.add_argument('--patience',type=int,default=2)
    p.add_argument('--batch',type=int,default=32);p.add_argument('--accumulation',type=int,default=2);p.add_argument('--workers',type=int,default=2)
    p.add_argument('--output',type=Path,default=ROOT/'models/vision/experiments/20261104')
    args=p.parse_args();args.output=args.output.resolve();assert args.output.is_relative_to(ROOT.resolve())
    if not torch.cuda.is_available():raise RuntimeError('CUDA required for this comparison')
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=True;torch.set_float32_matmul_precision('high')
    args.output.mkdir(parents=True,exist_ok=True);parts,audit=prepare(args.output)
    for name in ('efficientnet_b0','convnext_tiny'):train(name,parts,args)
    finish(parts,args,audit)
if __name__=='__main__':main()
