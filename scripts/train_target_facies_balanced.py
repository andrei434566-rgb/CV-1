"""Fixed balanced continuation after v4's target-priority recall regression.

Reuses verified frozen visual features. Alpha1 and boost1 are the previous
v3 recipe, fixed before evaluating this run; no development-score search.
"""
import argparse
import json
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import train_unibo_visual_facies as source
from scripts.train_target_facies_head import fit,codes,gates,CRITERIA
from scripts.partial_facies_ridge import mixed_metrics


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--features-run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(); previous,output=a.features_run.resolve(),a.output.resolve()
    if output.exists(): raise ValueError('Choose a new experiment directory')
    previous_cfg=json.loads((previous/'run_config.json').read_text(encoding='utf-8'))
    previous_result=json.loads((previous/'result.json').read_text(encoding='utf-8'))
    for f,h in previous_cfg['dependencies'].items():
        if source.digest(f)!=h: raise ValueError('Feature provenance changed: '+f)
    cfg=dict(previous_cfg)
    cfg['recipe']=dict(profile='qwen_post',alpha=1.,target_boost=1.,outcrop_weight=.25,folds=3,
        selection='No hyperparameter search: fixed prior v3 alpha1, balanced boost1. v4 target recall priority rejected due swamp regression.',
        frozen_encoder=True,chat_weights_changed=False,
        validation='Previous50 unchanged; evaluated only after fixed-recipe OOF saved')
    cfg['criteria']=CRITERIA
    cfg['features_reference']=str(previous)
    cfg['dependencies']=dict(previous_cfg['dependencies'])
    cfg['dependencies'].update({str(f):source.digest(f) for f in (Path(__file__),previous/'features.safetensors',previous/'train.jsonl',previous/'val.jsonl',previous/'audit_result.json')})
    cfg['prospective_reason']='Target-priority v4 increases B/C recall but severely overpredicts them on F. Fixed balanced recipe checks the data addition without targeted class boost.'
    output.mkdir(parents=True); source.write_json(output/'run_config.json',cfg)
    for name in ('train.jsonl','val.jsonl','features.safetensors'):
        shutil.copy2(previous/name,output/name)
    shutil.copy2(__file__,output/'training_script_snapshot.py')
    rows=source.read_rows(output/'train.jsonl'); val=source.read_rows(output/'val.jsonl')
    import torch
    from safetensors.torch import load_file,save_file
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    raw=load_file(str(output/'features.safetensors'))['train']
    contract=json.loads((previous/'MODEL_CONTRACT.json').read_text(encoding='utf-8'))
    dims=(contract['qwen_feature_dimension'],contract['pre_merger_feature_dimension'],contract['texture_dimension'])
    mapping=json.loads((previous/'selection_frozen.json').read_text(encoding='utf-8'))['fold_source_groups']
    assignments=[mapping[r['source_group']] for r in rows]; oof=['']*len(rows)
    for fold in range(3):
        mask=torch.tensor([f!=fold for f in assignments]); held=~mask
        sub=[r for r,k in zip(rows,mask.tolist()) if k]
        if {r['source_group'] for r in sub}&{r['source_group'] for r,k in zip(rows,held.tolist()) if k}: raise ValueError('Group leakage')
        state=fit(raw[mask],sub,1.,1.,dims)
        for i,c in zip(torch.where(held)[0].tolist(),codes(raw[held],state)): oof[i]=c
    metrics=mixed_metrics(rows,oof,source); winner=dict(alpha=1.,target_boost=1.,metrics=metrics)
    frozen=dict(stage='before_validation_features_and_predictions',winner=winner,candidates=[winner],
        selection_note='Only one predeclared fixed recipe; not selected using v4 grid or val',
        fold_source_groups=mapping,training_record_ids=[r['record_id'] for r in rows])
    source.write_json(output/'selection_frozen.json',frozen)
    source.write_json(output/'oof_answers.json',[dict(record_id=r['record_id'],source_group=r['source_group'],
        allowed_codes=r['allowed_codes'],prediction=c,fold=f,external=r['external']) for r,c,f in zip(rows,oof,assignments)])
    state=fit(raw,rows,1.,1.,dims)
    save_file({k:v.contiguous() for k,v in state.items()},str(output/'facies_head.safetensors'))
    result=dict(role='balanced_B_C_fixed_recipe_frozen_Qwen_research_candidate',encoder_or_chat_weights_changed=False,
        head_sha256=source.digest(output/'facies_head.safetensors'),selection_frozen_sha256=source.digest(output/'selection_frozen.json'),
        selected=dict(alpha=1.,target_boost=1.,outcrop_weight=.25),oof=metrics,
        train=mixed_metrics(rows,codes(raw,state),source))
    ref=Path(cfg['reference']); ref_cfg=json.loads((ref/'run_config.json').read_text(encoding='utf-8'))
    vraw=load_file(str(Path(ref_cfg['reference'])/'validation_features.safetensors'))['raw_val'][:,:dims[0]]
    expected=[r['conversations'][1]['value'] for r in val]; pred=codes(vraw,state)
    result['val']=source.classification_metrics(expected,pred)
    old_pred=codes(vraw,load_file(str(ref/'facies_head.safetensors')))
    result['v3']=source.classification_metrics(expected,old_pred)
    result['paired_vs_v3']=dict(fixed=sum(a!=e and b==e for a,b,e in zip(old_pred,pred,expected)),
        introduced=sum(a==e and b!=e for a,b,e in zip(old_pred,pred,expected)))
    keep=torch.tensor([r.get('domain')!='outcrop_analogue' for r in rows]); sub=[r for r,k in zip(rows,keep.tolist()) if k]
    ablation=fit(raw[keep],sub,1.,1.,dims)
    result['without_outcrop_fixed_selected_hyperparameters']=source.classification_metrics(expected,codes(vraw,ablation))
    source.write_json(output/'val_answers.json',[dict(record_id=r['metadata']['record_id'],source_group=r['metadata']['source_group'],expected_code=e,prediction=c)
        for r,e,c in zip(val,expected,pred)])
    result['gates']=gates(result['val'])
    result['status']='source_development_gates_passed_research_only' if all(result['gates'].values()) else 'not_promoted_failed_prospective_gates'
    result['v4_diagnostic_reference']=dict(head_sha256=previous_result['head_sha256'],val_correct=previous_result['val']['correct'])
    source.write_json(output/'result.json',result)
    contract.update(head_sha256=result['head_sha256'],acceptance_status=result['status'],training_recipe=cfg['recipe'],
        selected_profile=dict(profile='qwen_post',**result['selected']))
    source.write_json(output/'MODEL_CONTRACT.json',contract)
    print(json.dumps(dict(selected=result['selected'],val=result['val'],gates=result['gates']),ensure_ascii=False))


if __name__=='__main__': main()
