"""Independently reproduce v4 source crops, ridge equations, OOF and validation."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import train_unibo_visual_facies as source
from scripts.prepare_target_facies_20261031 import SOURCES,PHOTO_FILES,specifications,pixel_digest
from scripts.train_target_facies_head import weights_for,key,fit,codes,gates
from scripts.facies_head_v2_features import prepare_state,normalize
from scripts.partial_facies_ridge import label_system,mixed_metrics


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--raw',type=Path,default=ROOT/'open_datasets/target_facies_20261031_v4/raw')
    a=p.parse_args(); run=a.run.resolve()
    import torch
    from safetensors.torch import load_file
    from PIL import Image
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    cfg=json.loads((run/'run_config.json').read_text(encoding='utf-8'))
    result=json.loads((run/'result.json').read_text(encoding='utf-8'))
    frozen=json.loads((run/'selection_frozen.json').read_text(encoding='utf-8'))
    contract=json.loads((run/'MODEL_CONTRACT.json').read_text(encoding='utf-8'))
    for f,h in cfg['dependencies'].items():
        if source.digest(f)!=h: raise ValueError('Dependency changed: '+f)
    if max(frozen['candidates'],key=key)!=frozen['winner']: raise ValueError('Selection differs')
    if source.digest(run/'selection_frozen.json')!=result['selection_frozen_sha256']: raise ValueError('Selection changed')
    if source.digest(run/'facies_head.safetensors')!=result['head_sha256']: raise ValueError('Head changed')
    rows=source.read_rows(run/'train.jsonl'); val=source.read_rows(run/'val.jsonl')
    ref=Path(cfg['reference']); original_rows=source.read_rows(ref/'train.jsonl')
    if rows[:728]!=original_rows or val!=source.read_rows(ref/'val.jsonl'): raise ValueError('Old records changed')
    if [r['record_id'] for r in rows]!=frozen['training_record_ids']: raise ValueError('Feature order differs')
    for sid,meta in SOURCES.items():
        if source.digest(a.raw/(sid+'_pdf.pdf'))!=meta['sha256']: raise ValueError('PDF changed')
    images={}
    for name,(_,_,_,h) in PHOTO_FILES.items():
        if source.digest(a.raw/name)!=h: raise ValueError('Original image changed')
        with Image.open(a.raw/name) as im: images[name]=im.convert('RGB')
    specs=specifications()
    if len(specs)!=len(rows[728:]): raise ValueError('Extra count differs')
    for row,spec in zip(rows[728:],specs):
        sid=PHOTO_FILES[spec['file']][0]
        if row['source']!=SOURCES[sid] or row['allowed_codes']!=spec['allowed_codes'] or row['source_box']!=spec['box']: raise ValueError('Source mapping differs')
        if pixel_digest(images[spec['file']].crop(spec['box']))!=row['pixel_sha256']: raise ValueError('Original crop not reproduced')
    hashes={r['metadata']['image_sha256'] for r in val}
    for r in rows:
        if source.digest(r['image'])!=r['image_sha256'] or r['image_sha256'] in hashes: raise ValueError('Changed or duplicate image')
        hashes.add(r['image_sha256'])
    raw=load_file(str(run/'features.safetensors'))['train']; state=load_file(str(run/'facies_head.safetensors'))
    if not torch.equal(raw[:728],load_file(str(ref/'features.safetensors'))['train']): raise ValueError('Old feature cache differs')
    dims=(contract['qwen_feature_dimension'],contract['pre_merger_feature_dimension'],contract['texture_dimension'])
    rebuilt=prepare_state(raw,'qwen_post',*dims)
    if any(not torch.equal(state[k],v) for k,v in rebuilt.items()): raise ValueError('Normalizer differs')
    alpha,boost=result['selected']['alpha'],result['selected']['target_boost']
    y,precision,rotation=label_system([r['allowed_codes'] for r in rows])
    x=normalize(raw,state); w=weights_for(rows,boost)[:,None]*precision; beta=state['coefficients']@rotation
    rhs=x.T@(w*y); residual=x.T@(w*(x@beta))+alpha*beta-rhs
    relative=float(residual.norm()/rhs.norm())
    if relative>1e-10: raise ValueError('Ridge normal equations not satisfied')
    oof=['']*len(rows); mapping=frozen['fold_source_groups']; assignments=[mapping[r['source_group']] for r in rows]
    for fold in range(3):
        mask=torch.tensor([f!=fold for f in assignments]); held=~mask
        sub=[r for r,k in zip(rows,mask.tolist()) if k]
        if {r['source_group'] for r in sub}&{r['source_group'] for r,k in zip(rows,held.tolist()) if k}: raise ValueError('Group leakage')
        s=fit(raw[mask],sub,alpha,boost,dims)
        for i,c in zip(torch.where(held)[0].tolist(),codes(raw[held],s)): oof[i]=c
    if mixed_metrics(rows,oof,source)!=result['oof']: raise ValueError('OOF not reproduced')
    if oof!=[r['prediction'] for r in json.loads((run/'oof_answers.json').read_text(encoding='utf-8'))]: raise ValueError('Saved OOF differs')
    old_cfg=json.loads((ref/'run_config.json').read_text(encoding='utf-8'))
    original_ref=Path(old_cfg['reference'])
    vraw=load_file(str(original_ref/'validation_features.safetensors'))['raw_val'][:,:dims[0]]
    first_cfg=json.loads((original_ref/'run_config.json').read_text(encoding='utf-8'))
    if not torch.equal(vraw,load_file(str(Path(first_cfg['reference'])/'features.safetensors'))['val']): raise ValueError('Original validation cache differs')
    actual=source.classification_metrics([r['conversations'][1]['value'] for r in val],codes(vraw,state))
    if actual!=result['val'] or gates(actual)!=result['gates']: raise ValueError('Val metrics or gates differ')
    keep=torch.tensor([r.get('domain')!='outcrop_analogue' for r in rows]); sub=[r for r,k in zip(rows,keep.tolist()) if k]
    ablation=fit(raw[keep],sub,alpha,boost,dims)
    m=source.classification_metrics([r['conversations'][1]['value'] for r in val],codes(vraw,ablation))
    if m!=result['without_outcrop_fixed_selected_hyperparameters']: raise ValueError('Outcrop ablation differs')
    new=[(r,c) for r,c in zip(rows[728:],oof[728:])]
    per_new_group={g:dict(total=sum(r['source_group']==g for r,c in new),
        correct=sum(r['source_group']==g and c in r['allowed_codes'] for r,c in new)) for g in sorted({r['source_group'] for r,c in new})}
    audit=dict(status=result['status'],head_sha256=result['head_sha256'],source_crops_reconstructed_exactly=True,
        old_features_and_validation_unchanged=True,train_only_normalizer_reproduced=True,selected_group_oof_reproduced=True,
        partial_normal_equation_relative_residual=relative,new_source_group_oof=per_new_group,
        research_gates=result['gates'],val_correct=actual['correct'],val_total=actual['total'],
        installed_in_application=False,general_chat_weights_changed=False,geologist_approved=False,prepared_test_pixels_opened=0,
        audit_script_sha256=source.digest(__file__),
        limitations='Repeated development50; B source is one weathered outcrop, not new core. OOF used for selection, not independent final accuracy.')
    source.write_json(run/'audit_result.json',audit); source.write_json(run/'acceptance_review.json',audit)
    print(json.dumps(audit,ensure_ascii=False,indent=2))


if __name__=='__main__': main()
