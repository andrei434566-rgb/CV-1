"""Add author-labelled coal using a frozen CNN and replay of all eight old groups.

Parent photographs (all their processed variants together) are split afresh.
The author's pickled model is inspected as DATA with pickletools, never loaded.
Coal textures are not claimed to be independently labelled coal-core specimens.
"""
from __future__ import annotations
import argparse, hashlib, io, json, pickletools, random, sys, time, zipfile
from collections import Counter, defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from torch import nn
from torch.utils.data import DataLoader
from PIL import Image
from app.domain.rock_types import COAL_CLASSES, COAL_NAMES_RU
from app.infrastructure.local_vision import VisionEngine
from app.infrastructure.supervised_learning import classifier_input_layer, classification_metrics
from train_rock_taxonomy import Images, init_worker, write, sha, SEED


def verify_source():
    folder=ROOT/'open_datasets/coal_zenodo10702879'
    archive=folder/'Coal.zip'
    md5=hashlib.md5(archive.read_bytes()).hexdigest()
    if md5!='54c58f9aae44d46c0801f3b880852f97':raise ValueError('Coal archive checksum mismatch')
    # No unpickling or execution of an external model, including its classes.
    with zipfile.ZipFile(folder/'author_weights.pt') as z:
        ops=list(pickletools.genops(z.read(next(n for n in z.namelist() if n.endswith('data.pkl')))))
    matches=[]
    for i,(_,arg,_) in enumerate(ops):
        if arg=='names':
            values=[a for o,a,p in ops[i+1:i+12] if o.name not in {'BINPUT','LONG_BINPUT'}]
            matches.append(values)
    if not any(v[:5]==[None,0,'coal',None,223] or v[:4]==[None,0,'coal',None] for v in matches):
        raise ValueError('Author class 0 -> coal mapping not verified; do not guess labels')
    return {'archive':str(archive),'archive_md5':md5,'author_class_id':0,'author_class':'coal',
            'mapping_source':'https://zenodo.org/records/10728156','mapping_method':'static pickle opcode inspection; no model loaded',
            'author_weights_sha256':sha(folder/'author_weights.pt'),
            'source':'https://zenodo.org/records/10702879','author':'Wenyu Wang','license':'CC BY 4.0'}


def prepare_coal(out):
    source=verify_source();groups=defaultdict(list);excluded=[]
    with zipfile.ZipFile(source['archive']) as z:
        names=set(z.namelist())
        for member in sorted(names):
            if '/labels/' not in member or not member.endswith('.txt'):continue
            image=member.replace('/labels/','/images/').removesuffix('.txt')+'.jpg'
            if image not in names:continue
            group=Path(image).stem.split('_')[0]
            if not group or group[0] not in 'ab' or not group[1:].isdigit():raise ValueError('Unknown parent-photo naming')
            with Image.open(io.BytesIO(z.read(image))) as im:w,h=im.size
            for line in z.read(member).decode('utf8').splitlines():
                if not line.strip():continue
                c,x,y,bw,bh=map(float,line.split())
                if c!=0:raise ValueError('Unexpected source class')
                box=[max(0,round((x-bw/2)*w)),max(0,round((y-bh/2)*h)),
                     min(w,round((x+bw/2)*w)),min(h,round((y+bh/2)*h))]
                if min(box[2]-box[0],box[3]-box[1])<32:continue
                groups[group].append({'archive':source['archive'],'member':image,'box':box,'parent_photo':group,
                                      'source':'zenodo_coal','label':8,'well':'','label_provenance':'author_yolo_class_0_bbox'})
        split_groups={s:[] for s in ('train','val','test')};rng=random.Random(SEED)
        for prefix in ('a','b'):
            chosen=sorted(g for g in groups if g.startswith(prefix));rng.shuffle(chosen);n=round(len(chosen)*.2)
            split_groups['val']+=chosen[:n];split_groups['test']+=chosen[n:n*2];split_groups['train']+=chosen[n*2:]
        parts={s:[] for s in split_groups};seen={}
        for split,chosen in split_groups.items():
            for group in chosen:
                rows=groups[group][:];rng.shuffle(rows);accepted=0
                for row in rows:
                    with Image.open(io.BytesIO(z.read(row['member']))) as im:crop=im.convert('RGB').crop(row['box'])
                    digest=hashlib.sha256(str(crop.size).encode()+crop.tobytes()).hexdigest()
                    if digest in seen:excluded.append({'member':row['member'],'reason':'duplicate crop','first':seen[digest]});continue
                    seen[digest]=group;row['crop_sha256']=digest;row['split']=split;parts[split].append(row);accepted+=1
                    if accepted>=24:break
    audit={'source':source,'parent_photo_groups':split_groups,'splits':{s:len(r) for s,r in parts.items()},
           'duplicates_excluded':excluded,'well_independent':False,
           'limitation':'Parent images are disjoint; physical specimen identities and boreholes are unavailable. Coal texture domain differs from core-box photos.'}
    write(out/'coal_dataset.json',parts);write(out/'coal_audit.json',audit)
    return parts,audit


class Regions(Images):
    def __getitem__(self,index):
        row=self.rows[index]
        if 'archive' in row:
            if row['archive'] not in self.archives:self.archives[row['archive']]=zipfile.ZipFile(row['archive'])
            data=io.BytesIO(self.archives[row['archive']].read(row['member']))
        else:data=row['image']
        with Image.open(data) as im:
            im=im.convert('RGB')
            if 'box' in row:im=im.crop(row['box'])
            return self.transform(im),row['label']


def features(engine,rows):
    batches=[];targets=[];captured=[]
    ld=DataLoader(Regions(rows),batch_size=64,num_workers=2,persistent_workers=False,pin_memory=True,worker_init_fn=init_worker)
    hook=classifier_input_layer(engine.model,engine.card['architecture']).register_forward_pre_hook(
        lambda module,inputs:captured.append(inputs[0].detach().float().cpu()))
    try:
        with torch.inference_mode():
            for i,(x,y) in enumerate(ld):
                with torch.autocast('cuda',dtype=torch.float16):engine.model(x.to('cuda',non_blocking=True))
                batches.append(captured.pop());targets.extend(y.tolist())
                if i%100==0:print(json.dumps({'features':min((i+1)*64,len(rows)),'of':len(rows)}),flush=True)
    finally:hook.remove()
    return torch.cat(batches),torch.tensor(targets)


def measure(logits,y):return classification_metrics(y.tolist(),logits.argmax(1).tolist(),list(COAL_CLASSES))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--base',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--data-base',type=Path)
    ap.add_argument('--prepare-only',action='store_true');args=ap.parse_args();out=args.output.resolve()
    if not out.is_relative_to(ROOT.resolve()):raise ValueError('Editable C project only; E backup is protected')
    out.mkdir(parents=True,exist_ok=True)
    coal,audit=prepare_coal(out);print(json.dumps(audit['splits']),flush=True)
    if args.prepare_only:return
    if (out/'config.json').exists():raise ValueError('Do not overwrite a completed model')
    torch.set_num_threads(4);torch.manual_seed(SEED)
    engine=VisionEngine(device='cuda');engine._load(args.base)
    if len(engine.card['classes']) not in (4,8):raise ValueError('Needs the approved four/eight-group base')
    data_base=args.data_base or args.base
    parts=json.loads((data_base/'dataset.json').read_text(encoding='utf8'))
    merged={s:parts[s]+coal[s] for s in parts};write(out/'dataset.json',merged)
    dataset_sha=hashlib.sha256(json.dumps(merged,sort_keys=True,ensure_ascii=False).encode('utf8')).hexdigest()
    embeddings={}
    for split in ('train','val','test'):
        cache=out/(split+'_features.pt')
        if cache.exists():
            saved=torch.load(cache,weights_only=True)
            if saved.get('base_sha256')!=engine.card['model_sha256'] or saved.get('dataset_sha256')!=dataset_sha:
                raise ValueError('Cached features belong to a different base or dataset; choose a new output folder')
            x,y=saved['features'],saved['targets']
        else:
            print('Extracting '+split,flush=True);x,y=features(engine,merged[split])
            torch.save({'features':x,'targets':y,'base_sha256':engine.card['model_sha256'],'dataset_sha256':dataset_sha},cache)
        embeddings[split]=(x.to('cuda'),y.to('cuda'))
    old=classifier_input_layer(engine.model,engine.card['architecture'])
    head=nn.Linear(old.in_features,len(COAL_CLASSES)).to('cuda')
    with torch.no_grad():
        if len(engine.card['classes'])==8:head.weight[:8].copy_(old.weight);head.bias[:8].copy_(old.bias)
        else:
            for src,dst in enumerate((0,5,2,7)):head.weight[dst].copy_(old.weight[src]);head.bias[dst].copy_(old.bias[src])
        head.weight[8].zero_();head.bias[8]=-3.
    x,y=embeddings['train'];vx,vy=embeddings['val'];counts=torch.bincount(y,minlength=9).float()
    mend_mask=torch.tensor([r['source']=='mendeley' for r in merged['val']],device='cuda')
    old_reference=json.loads((ROOT/'models/vision/vision_20261104_143516/config.json').read_text(encoding='utf8'))
    retention_floor=old_reference['validation']['validation_macro_f1']-(.005 if len(engine.card['classes'])==4 else .02)
    old_group_floor=engine.card['validation']['macro_f1']-.015 if len(engine.card['classes'])==8 else .97
    criterion=nn.CrossEntropyLoss(weight=(counts.sum()/counts).sqrt(),label_smoothing=.02,reduction='none')
    replay_weights=torch.tensor([3. if r['source']=='mendeley' else 1. for r in merged['train']],device='cuda')
    optimizer=torch.optim.AdamW(head.parameters(),lr=.001,weight_decay=.01)
    best=-1.;history=[]
    for epoch in range(35):
        head.train();losses=[]
        for indices in torch.randperm(len(y),device='cuda').split(256):
            optimizer.zero_grad(set_to_none=True)
            loss=(criterion(head(x[indices]),y[indices])*replay_weights[indices]).mean()
            loss.backward();optimizer.step();losses.append(float(loss.detach()))
        head.eval()
        with torch.inference_mode():vlogits=head(vx);result=measure(vlogits,vy)
        old_macro=sum(r['f1'] for r in result['per_class'][:8])/8
        mend_result=measure(vlogits[mend_mask],vy[mend_mask])
        mend_macro=sum(mend_result['per_class'][i]['f1'] for i in (0,2,5,7))/4
        # Choose on validation only, preserving the old groups as well as coal.
        eligible=(old_macro>=old_group_floor and result['per_class'][8]['f1']>=.85
                  and mend_macro>=retention_floor)
        row={'epoch':epoch+1,'validation':result,'old_groups_macro_f1':old_macro,
             'mendeley_common_macro_f1':mend_macro,'retention_floor':retention_floor,
             'eligible':eligible,'loss':sum(losses)/len(losses)}
        history.append(row)
        if eligible and result['macro_f1']>best:
            best=result['macro_f1'];torch.save(head.state_dict(),out/'head.pt');write(out/'best_validation.json',row)
        if epoch%5==0:print(json.dumps({'epoch':epoch+1,'val_macro_f1':result['macro_f1'],'eligible':eligible}),flush=True)
    write(out/'history.json',history)
    if best<0:raise RuntimeError('Coal extension failed validation gate; eight-group model remains unchanged')
    head.load_state_dict(torch.load(out/'head.pt',weights_only=True));head.eval()
    with torch.inference_mode():
        vlogits=head(vx);test_logits=head(embeddings['test'][0]);test_y=embeddings['test'][1]
    temperature=min([.8,1.,1.2,1.5,2.,2.5,3.],key=lambda t:float(nn.functional.cross_entropy(vlogits/t,vy)))
    metrics=measure(test_logits,test_y);write(out/'test_metrics.json',metrics)
    write(out/'test_by_source.json',{s:measure(test_logits[[i for i,r in enumerate(merged['test']) if r['source']==s]],
                                            test_y[[i for i,r in enumerate(merged['test']) if r['source']==s]])
                                    for s in sorted({r['source'] for r in merged['test']})})
    engine.model.classifier[2 if engine.card['architecture']=='convnext_tiny' else 1]=head
    torch.save({k:v.cpu() for k,v in engine.model.state_dict().items()},out/'model.pt')
    card=dict(engine.card);card.update(schema='cv1-discriminative-vision-v2',background_class='background',
        noncommercial_only=True,licence='Research only: includes CC BY-NC 4.0 data',
        classes=list(COAL_CLASSES),class_names_ru=list(COAL_NAMES_RU),
        model_sha256=sha(out/'model.pt'),temperature=temperature,validation=measure(vlogits,vy),test=metrics,
        training_method='Approved CNN encoder frozen; nine-way supervised linear head with replay',
        coal_parent_photo_disjoint=True,coal_core_validated=False,base_model_sha256=engine.card['model_sha256'],
        dataset_sha256=dataset_sha,mendeley_replay_weight=3.,selection='Validation only: old groups, coal and common-source retention gates')
    data_card=json.loads((data_base/'config.json').read_text(encoding='utf8'))
    card['sources']=data_card['sources']+[{'name':'Wenyu Wang · Coal','url':audit['source']['source'],'license':'CC BY 4.0'}]
    card['requested_classes_missing_direct_examples']=[x for x in data_card['requested_classes_missing_direct_examples'] if x!='Уголь']
    card['interbedding_trained']=False;card['well_independent_test']=False
    card['limitations']=data_card['limitations']+' '+audit['limitation'];write(out/'config.json',card)
    print(json.dumps({'done':True,'test_macro_f1':metrics['macro_f1'],'model':str(out)}),flush=True)


if __name__=='__main__':main()
