"""Reproduce the frozen winner's group holdouts and audit the final fit."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import train_unibo_visual_facies as source
from scripts.facies_head_v2_features import (
    PROFILES,make_folds,prepare_state,normalize,sample_weights,ridge_coefficients,predict,
)
from scripts.train_qwen_visual_facies_head import predict_head


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    args=p.parse_args(); run=args.run.resolve()
    import torch
    from safetensors.torch import load_file
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    cfg=json.loads((run/'run_config.json').read_text(encoding='utf-8'))
    frozen=json.loads((run/'selection_frozen.json').read_text(encoding='utf-8'))
    result=json.loads((run/'result.json').read_text(encoding='utf-8'))
    contract=json.loads((run/'MODEL_CONTRACT.json').read_text(encoding='utf-8'))
    for path,expected in cfg['dependencies'].items():
        if source.digest(path)!=expected: raise ValueError('Run dependency changed: '+path)
    if source.digest(run/'selection_frozen.json')!=result['selection_frozen_sha256']: raise ValueError('Selection changed')
    if source.digest(run/'facies_head.safetensors')!=result['head_sha256']: raise ValueError('Head changed')
    if contract['head_sha256']!=result['head_sha256']: raise ValueError('Contract head differs')
    raw_train=source.read_rows(Path(cfg['data'])/'train.jsonl')
    raw_val=source.read_rows(Path(cfg['data'])/'val.jsonl')
    source.validate_facies_split(raw_train,raw_val)
    train,val=([source.transform_facies(r) for r in rows] for rows in (raw_train,raw_val))
    if train!=source.read_rows(run/'train.jsonl') or val!=source.read_rows(run/'val.jsonl'): raise ValueError('Run rows changed')
    if val!=source.read_rows(Path(cfg['reference'])/'val.jsonl'): raise ValueError('Validation changed')
    if [r['metadata']['record_id'] for r in train]!=frozen['training_record_ids']: raise ValueError('Feature row order changed')
    cache=load_file(str(run/'train_features.safetensors'))
    raw=torch.cat((cache['qwen_train'],cache['texture_train']),1)
    raw_val_features=load_file(str(run/'validation_features.safetensors'))['raw_val']
    labels=torch.tensor([source.CODES.index(r['conversations'][1]['value']) for r in train])
    groups=[r['metadata']['source_group'] for r in train]
    assignment,mapping=make_folds(labels.numpy(),groups)
    if mapping!=frozen['fold_source_groups']: raise ValueError('Fold assignment changed')
    winner=max(frozen['candidates'],key=lambda r:(r['all_classes_recalled'],r['metrics']['macro_f1'],
        r['mean_source_group_accuracy'],r['alpha'],-PROFILES.index(r['profile'])))
    if winner!=frozen['winner']: raise ValueError('Selection rule not reproduced')
    spec=(winner['profile'],result['qdim'],result['pre_dim'],result['texture_dim'])
    oof=torch.full_like(labels,-1)
    for fold in range(3):
        tr=torch.tensor(assignment!=fold); held=~tr
        tr_groups=[g for g,keep in zip(groups,tr.tolist()) if keep]
        if set(tr_groups)&{g for g,keep in zip(groups,held.tolist()) if keep}: raise ValueError('Group leakage')
        state=prepare_state(raw[tr],*spec)
        weights=sample_weights(labels[tr],tr_groups,winner['weighting'])
        state['coefficients']=ridge_coefficients(normalize(raw[tr],state),labels[tr],weights,winner['alpha'])
        oof[held]=predict(raw[held],state).argmax(1)
    oof_metrics=source.classification_metrics([source.CODES[c] for c in labels.tolist()],
        [source.CODES[c] for c in oof.tolist()])
    if oof_metrics!=winner['metrics']: raise ValueError('Selected OOF metrics not reproduced')
    source.write_json(run/'audited_oof_answers.json',[{'record_id':r['metadata']['record_id'],
        'source_group':groups[i],'fold':int(assignment[i]),'expected_code':source.CODES[int(labels[i])],
        'prediction':source.CODES[int(oof[i])]} for i,r in enumerate(train)])
    state=load_file(str(run/'facies_head.safetensors'))
    rebuilt=prepare_state(raw,*spec)
    for key in rebuilt:
        if not torch.equal(rebuilt[key],state[key]): raise ValueError('Train-only normalization not reproduced: '+key)
    x=normalize(raw,state); w=sample_weights(labels,groups,winner['weighting'])
    y=torch.nn.functional.one_hot(labels,6).double(); b=state['coefficients']
    rhs=x.T@(w[:,None]*y)
    residual=x.T@(w[:,None]*(x@b))+winner['alpha']*b-rhs
    relative_residual=float(residual.norm()/rhs.norm())
    if relative_residual>1e-10: raise ValueError('Ridge normal-equation residual too large')
    for split,rows,features in (('train',train,raw),('val',val,raw_val_features)):
        answers=json.loads((run/(split+'_answers.json')).read_text(encoding='utf-8'))
        scores=predict(features,state)
        if [a['record_id'] for a in answers]!=[r['metadata']['record_id'] for r in rows]: raise ValueError('Answer order changed')
        if [a['prediction'] for a in answers]!=[source.CODES[i] for i in scores.argmax(1).tolist()]: raise ValueError('Saved answers differ')
        recomputed=source.classification_metrics([a['expected_code'] for a in answers],[a['prediction'] for a in answers])
        if recomputed!=result[split]: raise ValueError('Metrics differ')
    reference=Path(cfg['reference'])
    old_features=load_file(str(reference/'features.safetensors'))['val']
    old_scores=predict_head(old_features,load_file(str(reference/'facies_head.safetensors')))
    old_codes=[source.CODES[i] for i in old_scores.argmax(1).tolist()]
    new_codes=[a['prediction'] for a in answers]
    expected=[a['expected_code'] for a in answers]
    paired={'fixed_old_errors':sum(o!=e and n==e for o,n,e in zip(old_codes,new_codes,expected)),
            'introduced_errors':sum(o==e and n!=e for o,n,e in zip(old_codes,new_codes,expected)),
            'changed_predictions':sum(o!=n for o,n in zip(old_codes,new_codes))}
    per_group={g:{'total':sum(a['source_group']==g for a in answers),
        'v1_correct':sum(a['source_group']==g and old_codes[i]==a['expected_code'] for i,a in enumerate(answers)),
        'v2_correct':sum(a['source_group']==g and a['prediction']==a['expected_code'] for a in answers)}
        for g in sorted({a['source_group'] for a in answers})}
    recalls=result['val']['per_class_recall']; c=cfg['criteria']
    gates={'accuracy':result['val']['correct']>=c['minimum_correct'],
        'macro_f1':result['val']['macro_f1']>=c['macro_f1_minimum'],
        'all_six_recalled':all(v>0 for v in recalls.values()),
        'prodelta_improved':recalls['C']*result['val']['expected_distribution']['C']>=c['prodelta_min_correct'],
        'PDF_preserved':recalls['B']*result['val']['expected_distribution']['B']>=c['poorly_drained_floodplain_min_correct']}
    if gates!=result['gates']: raise ValueError('Acceptance gates differ')
    status='accepted_source_development_research_only' if all(gates.values()) else 'not_promoted_failed_prospective_gates'
    audit={'status':status,'train_correct':result['train']['correct'],'val_correct':result['val']['correct'],
        'macro_f1':result['val']['macro_f1'],'paired_vs_v1':paired,'validation_per_group':per_group,
        'selected_oof_reproduced':True,'all_source_image_hashes_and_splits_verified':True,
        'selected_candidate_matches_train_only_rule':True,'final_fit_relative_normal_equation_residual':relative_residual,
        'head_sha256':result['head_sha256'],'selection_frozen_sha256':result['selection_frozen_sha256'],
        'research_gates':gates,'installed_in_application':False,'geologist_approved':False,
        'source_license':cfg['source_license'],'validation_note':'Repeated development set, three correlated proxy S groups; no final independent accuracy',
        'chat_or_encoder_weights_changed':False,'prepared_test_pixels_opened':0}
    source.write_json(run/'audit_result.json',audit)
    source.write_json(run/'acceptance_review.json',audit)
    contract.update({'pre_merger_feature_dimension':result['pre_dim'],'acceptance_status':status,
        'preprocessing_files_sha256':{Path(path).name:sha for path,sha in cfg['base_config_tokenizer_sha256'].items()}})
    source.write_json(run/'MODEL_CONTRACT.json',contract)
    print(json.dumps(audit,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__': main()
