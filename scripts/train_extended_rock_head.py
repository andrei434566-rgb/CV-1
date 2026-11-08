"""Production candidate without the coal domain that failed real-core smoke review.

Reuses frozen ConvNeXt features and chooses the eight-way head on validation only.
"""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from torch import nn
from app.domain.rock_types import ROCK_CLASSES,ROCK_NAMES_RU
from app.infrastructure.local_vision import VisionEngine
from app.infrastructure.supervised_learning import classification_metrics,classifier_input_layer
from train_rock_taxonomy import write,sha,SEED

def measure(logits,y):return classification_metrics(y.tolist(),logits.argmax(1).tolist(),list(ROCK_CLASSES))

def main():
    source=ROOT/'models/vision/rock_taxonomy_convnext_20261106'
    out=ROOT/'models/vision/rock_taxonomy_core_20261106'
    if out.exists():raise ValueError('Do not overwrite an existing version')
    out.mkdir(parents=True);torch.set_num_threads(4);torch.manual_seed(SEED)
    engine=VisionEngine(device='cuda');engine._load(ROOT/'models/vision/vision_20261104_143516')
    parts=json.loads((source/'dataset.json').read_text(encoding='utf8'));emb={};rows={}
    for split in parts:
        data=torch.load(source/(split+'_features.pt'),weights_only=True)
        if data['base_sha256']!=engine.card['model_sha256']:raise ValueError('Wrong encoder cache')
        mask=data['targets']<8;emb[split]=(data['features'][mask].to('cuda'),data['targets'][mask].to('cuda'))
        rows[split]=[r for r in parts[split] if r['label']<8]
    old=classifier_input_layer(engine.model,'convnext_tiny');head=nn.Linear(old.in_features,8).to('cuda')
    with torch.no_grad():
        for src,dst in enumerate((0,5,2,7)):head.weight[dst].copy_(old.weight[src]);head.bias[dst].copy_(old.bias[src])
    x,y=emb['train'];vx,vy=emb['val'];counts=torch.bincount(y,minlength=8).float()
    weights=torch.tensor([3. if r['source']=='mendeley' else 1. for r in rows['train']],device='cuda')
    criterion=nn.CrossEntropyLoss(weight=(counts.sum()/counts).sqrt(),label_smoothing=.02,reduction='none')
    optimizer=torch.optim.AdamW(head.parameters(),lr=.001,weight_decay=.01)
    mend=torch.tensor([r['source']=='mendeley' for r in rows['val']],device='cuda')
    floor=engine.card['validation']['validation_macro_f1']-.005;best=-1.;history=[]
    for epoch in range(35):
        head.train()
        for idx in torch.randperm(len(y),device='cuda').split(256):
            optimizer.zero_grad(set_to_none=True);loss=(criterion(head(x[idx]),y[idx])*weights[idx]).mean();loss.backward();optimizer.step()
        head.eval()
        with torch.inference_mode():vlogits=head(vx);result=measure(vlogits,vy);mr=measure(vlogits[mend],vy[mend])
        common=sum(mr['per_class'][i]['f1'] for i in (0,2,5,7))/4
        eligible=result['macro_f1']>=.97 and common>=floor
        row={'epoch':epoch+1,'validation':result,'common_mendeley_validation_macro_f1':common,'eligible':eligible}
        history.append(row)
        if eligible and result['macro_f1']>best:
            best=result['macro_f1'];torch.save(head.state_dict(),out/'head.pt');write(out/'best_validation.json',row)
    write(out/'history.json',history)
    if best<0:raise RuntimeError('No head passed the retention gate; active model unchanged')
    head.load_state_dict(torch.load(out/'head.pt',weights_only=True));head.eval()
    with torch.inference_mode():vl=head(vx);tl=head(emb['test'][0]);ty=emb['test'][1]
    temperature=min([.8,1.,1.2,1.5,2.,2.5,3.],key=lambda t:float(nn.functional.cross_entropy(vl/t,vy)))
    by_source={}
    for s in sorted({r['source'] for r in rows['test']}):
        indices=[i for i,r in enumerate(rows['test']) if r['source']==s];by_source[s]=measure(tl[indices],ty[indices])
    write(out/'test_by_source.json',by_source);write(out/'test_metrics.json',measure(tl,ty));write(out/'dataset.json',rows)
    engine.model.classifier[2]=head;torch.save({k:v.cpu() for k,v in engine.model.state_dict().items()},out/'model.pt')
    card=json.loads((ROOT/'models/vision/rock_taxonomy_20261105/config.json').read_text(encoding='utf8'))
    card.update(architecture='convnext_tiny',classes=list(ROCK_CLASSES),class_names_ru=list(ROCK_NAMES_RU),
        model_sha256=sha(out/'model.pt'),temperature=temperature,validation=measure(vl,vy),test=measure(tl,ty),
        training_method='ConvNeXt-Tiny encoder from the approved 40000-crop model frozen; eight-way head on source-labelled replay',
        base_model_sha256=engine.card['model_sha256'],coal_core_validated=False,coal_excluded=True,
        coal_exclusion_reason='Coal texture model failed real-core domain smoke review; not a production rock class',
        well_independent_test=False,noncommercial_only=True)
    write(out/'config.json',card)
    m=by_source['mendeley'];print(json.dumps({'done':True,'test_macro_f1':card['test']['macro_f1'],
        'mendeley_accuracy':m['accuracy'],'mendeley_macro_f1':sum(m['per_class'][i]['f1'] for i in (0,2,5,7))/4,'folder':str(out)}),flush=True)

if __name__=='__main__':main()
