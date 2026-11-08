"""Fit small independent evidence probes; freeze every original CNN weight.

No unmentioned structure becomes negative. This is NOT a validated 16-type
classifier and is never activated as an app model by this research script.
"""
from pathlib import Path
import argparse,hashlib,json,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from PIL import Image
from torch import nn
from app.infrastructure.bedding_classifier import BeddingEngine,prepare_image
from app.infrastructure.bedding_expansion import TYPES,FEATURE_CONTRACT,validate_examples,known_targets,masked_balanced_loss,promotion_gate

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'models/bedding/extension_20261106_v2')
    p.add_argument('--epochs',type=int,default=120);args=p.parse_args();out=args.output.resolve()
    if not out.is_relative_to(ROOT.resolve()):raise ValueError('Output outside editable C workspace')
    if (out/'head.pt').exists():raise ValueError('Choose a new version; trained probes are immutable')
    out.mkdir(parents=True,exist_ok=True);started=time.perf_counter();torch.set_num_threads(4);torch.manual_seed(20261106)
    base=ROOT/'models/bedding/rf100_convnext_20261106';protected=[base/'model.pt',base/'config.json',ROOT/'models/vision/current.json']
    pointer=json.loads((ROOT/'models/vision/current.json').read_text(encoding='utf8'))
    protected+=[ROOT/'models/vision'/pointer['folder']/'model.pt']
    protected+=sorted((ROOT/'models/trained').rglob('*.pt'))
    before={str(f.relative_to(ROOT)):sha(f) for f in protected}
    source=json.loads((ROOT/'open_datasets/bedding_20261106/expansion/dataset.json').read_text(encoding='utf8'))
    rows=validate_examples(source['examples'])
    native=json.loads((base/'dataset.json').read_text(encoding='utf8'))
    for split,items in native.items():
        for i,item in enumerate(items):
            if item['label']!=2:continue # ONLY author's explicit Massive class, not guessed absence.
            rows.append(dict(item,id=f'massive_{split}_{i}',split=split,source_group='rf100:'+item['well'],source='rf100',
                             negative_origin='author_explicit_massive_structure'))
    for a,b in [('train','val'),('train','test'),('val','test')]:
        assert not {r['source_group'] for r in rows if r['split']==a}&{r['source_group'] for r in rows if r['split']==b}
    write(out/'dataset.json',{'examples':rows,'positive_source_inventory':str((ROOT/'open_datasets/bedding_20261106/expansion/dataset.json').relative_to(ROOT))})
    device='cuda' if torch.cuda.is_available() else 'cpu';engine=BeddingEngine(base,device=device)
    model=engine.model;model.requires_grad_(False);model.eval();captured=[]
    hook=model.classifier[2].register_forward_pre_hook(lambda m,x:captured.append(x[0].detach().float().cpu()))
    features=[];base_logits=[]
    try:
        for i in range(0,len(rows),24):
            batch=[]
            for row in rows[i:i+24]:
                with Image.open(ROOT/row['image']) as im:
                    im=im.convert('RGB')
                    if 'negative_origin' in row:
                        if sha(ROOT/row['image'])!=row['image_sha256']:raise ValueError('Changed source negative')
                        im=im.crop(row['box'])
                    batch.append(prepare_image(im))
            with torch.inference_mode():z=model(torch.stack(batch).to(device)).float().cpu()
            features.append(captured.pop());base_logits.append(z)
        reference_batch=torch.stack(batch).to(device)
        with torch.inference_mode():reference_before=model(reference_batch).cpu()
        captured.clear()
    finally:hook.remove()
    embedding=torch.nn.functional.normalize(torch.cat(features),dim=1)
    target,mask=known_targets(rows);train=torch.tensor([r['split']=='train' for r in rows])
    fitted=[bool(((target[train,j]>0)&(mask[train,j]>0)).any() and ((target[train,j]==0)&(mask[train,j]>0)).any()) for j in range(len(TYPES))]
    head=nn.Linear(embedding.shape[1],len(TYPES))
    with torch.no_grad():
        for j,ready in enumerate(fitted):
            if not ready:head.weight[j].zero_();head.bias[j].zero_()
    optimizer=torch.optim.AdamW(head.parameters(),lr=.03,weight_decay=.01)
    history=[]
    # Fixed schedule declared before any test predictions; no test-guided epoch selection.
    for epoch in range(args.epochs):
        optimizer.zero_grad();loss=masked_balanced_loss(head(embedding[train]),target[train],mask[train])
        if not torch.isfinite(loss):raise ValueError('Non-finite loss')
        loss.backward();optimizer.step()
        if epoch in (0,19,59,args.epochs-1):
            history.append({'epoch':epoch+1,'loss':float(loss.detach())});print(json.dumps(history[-1]),flush=True)
    torch.save(head.state_dict(),out/'head.pt');scores=head(embedding).sigmoid().detach();metrics={}
    threshold=.8
    for split in ('val','test'):
        indices=torch.tensor([r['split']==split for r in rows]);details=[]
        for j,name in enumerate(TYPES):
            known=indices&(mask[:,j]>0);truth=target[known,j]>0;pred=scores[known,j]>=threshold
            pos=int(truth.sum());neg=int((~truth).sum());tp=int((truth&pred).sum()) if fitted[j] else None;fp=int((~truth&pred).sum()) if fitted[j] else None
            details.append({'type':name,'known_positives':pos,'known_negatives':neg,'tp':tp,'fp':fp,
                            'fitted':fitted[j],
                            'positive_recall':tp/pos if pos and fitted[j] else None,'massive_false_positive_rate':fp/neg if neg and fitted[j] else None,
                            'all_types_discrimination_validated':False})
        metrics[split]=details
    with torch.inference_mode():reference_after=model(reference_batch).cpu()
    assert torch.equal(reference_before,reference_after),'Frozen original CNN outputs changed'
    assert before=={str(f.relative_to(ROOT)):sha(f) for f in protected},'Protected weights/pointers changed'
    write(out/'protected_models.json',{'files_before_and_after':before,'cnn_logits_identical':True,'frozen_backbone':True,
                                     'cnn_control_rows':len(reference_batch)})
    write(out/'predictions.json',[{'id':r['id'],'split':r['split'],'positive_type':r.get('positive_type'),
                                  'scores':{name:(float(s[j]) if fitted[j] else None) for j,name in enumerate(TYPES)}} for r,s in zip(rows,scores)])
    card={'schema':'cv1-bedding-partial-evidence-v1','generative':False,'base_model_sha256':engine.card['model_sha256'],
          'head_sha256':sha(out/'head.pt'),'base_architecture':'convnext_tiny','feature_contract':FEATURE_CONTRACT,
          'types':TYPES,'feature_dim':embedding.shape[1],'training':'frozen CNN + linear sigmoid probes; masked balanced BCE',
          'unknown_is_negative':False,'negative_scope':'Author-labelled massive regions exclude visible lamination; grading remains unknown.',
          'fitted_types':[name for name,ready in zip(TYPES,fitted) if ready],
          'unfitted_types':{name:'No verified negative labels: massive does not exclude grading.' for name,ready in zip(TYPES,fitted) if not ready},
          'epochs':args.epochs,'history':history,'threshold':threshold,'metrics':metrics,'deployment':'research_only',
          'promotion':promotion_gate(rows),'automatic_assignment':False,
          'limitation':'Small literature positive sample; no validation against diverse competing bedding types. No overall 16-class accuracy.',
          'elapsed_seconds':time.perf_counter()-started}
    write(out/'config.json',card);print(json.dumps({'finished':True,'types_fitted':sum(fitted),'automatic_assignment':False,
           'seconds':card['elapsed_seconds'],'positive_support':card['promotion']['positive_support']},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
