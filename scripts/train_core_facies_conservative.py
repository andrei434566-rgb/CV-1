"""Local-only core-photo continuation with class-preservation acceptance.

No outcrops. No synthetic descriptions or pseudo-labels. No auto-promotion.
The Salpi source lacks a verified explicit distribution licence: keep all
mixed weights outside Git/release. Encoder and language model stay frozen.
"""
import argparse
from collections import Counter
import gc
import json
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import train_unibo_visual_facies as source
from scripts.facies_head_v2_features import prepare_state,normalize,predict
from scripts.partial_facies_ridge import row_weights,fit_partial,mixed_metrics
from scripts.conservative_facies_update import proximal_fit,training_gates,selection_key,preservation,PROTECTED
from scripts.prepare_core_facies_20261031_v6 import SOURCES,PHOTOS,pixel_digest

RECIPE=dict(proximities=[1.,10.,100.],new_weights=[.25,1.,4.],alpha=1.,end_face_weight=.25,
    seed=20261031,folds=3,profile='qwen_post',
    selection='Old728 group OOF only: require nondecreasing recall AND F1 for A,D,E,F, improved recall for BOTH B,C and macroF1 not lower; otherwise diagnostic only.',
    normalizer='Fit only on old train rows of each fold; no held/new-source statistics',
    objective='Original supervised/partial-label loss + new core labels + alpha*||beta||^2 + proximity*||beta-prior||^2',
    prior='Re-fit v3 alpha1, external_weight1 on old train subset only; no held labels in prior',
    validation='Previous50 unchanged, one evaluation after train selection frozen',
    distribution_approved=False,installed_in_application=False,encoder_or_chat_weights_changed=False)


def make_state(oldraw,oldrows,dims):
    s=prepare_state(oldraw,'qwen_post',*dims)
    w=row_weights([r['allowed_codes'] for r in oldrows],[r['external'] for r in oldrows],1.)
    s['coefficients']=fit_partial(normalize(oldraw,s),[r['allowed_codes'] for r in oldrows],w,1.)
    return s,w


def new_weights(oldrows,newrows,multiplier):
    import torch
    counts=Counter(r['allowed_codes'][0] for r in oldrows if len(r['allowed_codes'])==1);n=sum(counts.values())
    return torch.tensor([n/(6*counts[r['allowed_codes'][0]])*multiplier*(RECIPE['end_face_weight'] if r.get('domain')=='core_end_face' else 1.) for r in newrows],dtype=torch.float64)


def fit_update(oldraw,oldrows,newraw,newrows,dims,proximity,multiplier):
    import torch
    s,w=make_state(oldraw,oldrows,dims);prior=s['coefficients'].clone()
    x=normalize(torch.cat((oldraw,newraw)),s)
    allowed=[r['allowed_codes'] for r in oldrows+newrows]
    weights=torch.cat((w,new_weights(oldrows,newrows,multiplier)))
    s['coefficients']=proximal_fit(x,allowed,weights,prior,proximity)
    return s


def codes(raw,s):return [source.CODES[i] for i in predict(raw,s).argmax(1).tolist()]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,default=ROOT.parent/'vlm_core_v1/data/core_facies_20261031_v6_clean')
    p.add_argument('--reference',type=Path,default=ROOT.parent/'vlm_core_v1/outputs/core_qwen_web_partial_facies_20261031_v3')
    p.add_argument('--core-cache',type=Path,default=ROOT.parent/'vlm_core_v1/outputs/core_qwen_target_facies_20261031_v4')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();data,ref,cache,out=(v.resolve() for v in (a.data,a.reference,a.core_cache,a.output))
    if out.exists():raise ValueError('Choose new experiment directory')
    if not out.is_relative_to(ROOT.parent/'vlm_core_v1/outputs'):raise ValueError('Local-only weights must remain in ignored experiment outputs')
    old=source.read_rows(ref/'train.jsonl');val=source.read_rows(ref/'val.jsonl');previous=source.read_rows(cache/'train.jsonl')[728:739]
    extra=source.read_rows(data/'train.jsonl');contract=json.loads((ref/'MODEL_CONTRACT.json').read_text(encoding='utf-8'))
    summary=json.loads((data/'summary.json').read_text(encoding='utf-8'))
    if len(old)!=728 or len(previous)!=11 or len(extra)!=13 or len(val)!=50 or summary['sources']!=SOURCES:raise ValueError('Source data differ')
    from PIL import Image
    known={r['image_sha256'] for r in old}|{r['metadata']['image_sha256'] for r in val}
    pixels=set()
    # Compare decoded pixels against old train/val as well as newly prepared images.
    for r in old+val:
        with Image.open(r['image']) as im:pixels.add(pixel_digest(im.convert('RGB')))
    for r in previous+extra:
        if 'outcrop' in r.get('domain',''):raise ValueError('Outcrop not allowed')
        if source.digest(r['image'])!=r['image_sha256'] or r['image_sha256'] in known:raise ValueError('Changed/duplicate crop')
        with Image.open(r['image']) as im:h=pixel_digest(im.convert('RGB'))
        if h!=r['pixel_sha256'] or h in pixels:raise ValueError('Duplicate/changed source pixels')
        known.add(r['image_sha256']);pixels.add(h)
    for r in extra:
        if r['source']!=SOURCES[r['source_id']] or r['source_file_sha256']!=PHOTOS[r['source_file']]:raise ValueError('Source provenance differs')
    if source.package_versions()!=contract['package_versions']:raise ValueError('Extraction environment changed')
    ref_cfg=json.loads((ref/'run_config.json').read_text(encoding='utf-8'));base=Path(ref_cfg['base'])
    from scripts.classify_with_qwen_facies_head import streamed_sha
    if streamed_sha(base/'model.safetensors')!=contract['base_weights_sha256']:raise ValueError('Original base changed')
    for n,h in contract['preprocessing_files_sha256'].items():
        if source.digest(base/n)!=h:raise ValueError('Processor differs')
    out.mkdir(parents=True);rows=old+previous+extra
    source.write_rows(out/'train.jsonl',rows);source.write_rows(out/'val.jsonl',val)
    deps=[Path(__file__),ROOT/'scripts/conservative_facies_update.py',ROOT/'scripts/prepare_core_facies_20261031_v6.py',
        ROOT/'scripts/partial_facies_ridge.py',ROOT/'scripts/facies_head_v2_features.py',data/'summary.json',data/'train.jsonl',
        ref/'features.safetensors',ref/'train.jsonl',ref/'val.jsonl',ref/'facies_head.safetensors',ref/'selection_frozen.json',
        cache/'features.safetensors',cache/'train.jsonl',cache/'selection_frozen.json']
    cfg=dict(recipe=RECIPE,base=str(base),reference=str(ref),core_cache=str(cache),data=str(data),train_count=len(rows),old_count=len(old),previous_core_crops=len(previous),new_crops=len(extra),
        source_rights='UniBo NC-SA; old web sources CC BY; Salpi explicit distribution grant unverified. LOCAL ONLY, no weight/source redistribution.',
        dependencies={str(f):source.digest(f) for f in deps},distribution_approved=False,prepared_test_pixels_opened=0,installed_in_application=False)
    source.write_json(out/'run_config.json',cfg);shutil.copy2(__file__,out/'training_script_snapshot.py')
    from scripts.train_qwen_source_pilot import configure_offline_environment,_deny_network
    configure_offline_environment();sys.addaudithook(_deny_network)
    import torch
    from safetensors.torch import load_file,save_file
    torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.manual_seed(20261031)
    def log(event,**kw):
        obj=dict(event=event,**kw)
        with (out/'progress.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(obj)+'\n')
        print(json.dumps(obj),flush=True)
    oldraw=load_file(str(ref/'features.safetensors'))['train'];priorraw=load_file(str(cache/'features.safetensors'))['train'][728:739]
    if oldraw.shape!=(728,6144) or priorraw.shape!=(11,6144):raise ValueError('Cache shape differs')
    if not torch.cuda.is_available() or torch.cuda.mem_get_info()[0]<5*1024**3:raise RuntimeError('Need5GiB free CUDA')
    from transformers import AutoProcessor,Qwen3VLForConditionalGeneration
    from scripts.train_qwen_visual_facies_head import pool_features
    processor=AutoProcessor.from_pretrained(base,local_files_only=True)
    model=Qwen3VLForConditionalGeneration.from_pretrained(base,local_files_only=True,dtype=torch.float16,attn_implementation='sdpa',device_map='cuda')
    model.eval();model.requires_grad_(False);features=[]
    for i,r in enumerate(extra,1):
        with Image.open(r['image']) as im:image=im.convert('RGB');image.thumbnail((300,300))
        inputs=source.prompt_inputs(processor,image,source.PROMPT).to('cuda')
        with torch.inference_mode():
            visual=model.model.visual(inputs['pixel_values'].to(model.dtype),grid_thw=inputs['image_grid_thw'],return_dict=True);features.append(pool_features(visual).cpu())
        del inputs,visual
        log('new_features',row=i,total=len(extra))
    del model,processor;gc.collect();torch.cuda.empty_cache()
    newraw=torch.cat((priorraw,torch.stack(features)));newrows=previous+extra;raw=torch.cat((oldraw,newraw))
    save_file({'train':raw},str(out/'features.safetensors'))
    dims=(contract['qwen_feature_dimension'],contract['pre_merger_feature_dimension'],contract['texture_dimension'])
    mapping=json.loads((cache/'selection_frozen.json').read_text(encoding='utf-8'))['fold_source_groups']
    mapping={g:f for g,f in mapping.items() if g in {r['source_group'] for r in old+previous}}
    for g in sorted({r['source_group'] for r in extra}):
        label=next(r['allowed_codes'][0] for r in extra if r['source_group']==g)
        def cost(f):
            assigned=[r for r in rows if r['source_group'] in mapping and mapping[r['source_group']]==f]
            return sum(r['allowed_codes']==[label] for r in assigned),len(assigned),f
        mapping[g]=min(range(3),key=cost)
    assignments=[mapping[r['source_group']] for r in rows]
    base_oof=['']*len(old);folds=[]
    for f in range(3):
        om=torch.tensor([mapping[r['source_group']]!=f for r in old]);nm=torch.tensor([mapping[r['source_group']]!=f for r in newrows])
        oo=[r for r,k in zip(old,om.tolist()) if k];nn=[r for r,k in zip(newrows,nm.tolist()) if k]
        held=torch.tensor([i==f for i in assignments]);train_groups={r['source_group'] for r in oo+nn}
        if train_groups&{r['source_group'] for r,k in zip(rows,held.tolist()) if k}:raise ValueError('Source leakage')
        bs,_=make_state(oldraw[om],oo,dims)
        for i,c in zip(torch.where(~om)[0].tolist(),codes(oldraw[~om],bs)):base_oof[i]=c
        folds.append((om,nm,oo,nn,held))
    baseline_metrics=mixed_metrics(old,base_oof,source)['exact_six_class']
    saved_old=json.loads((ref/'oof_answers.json').read_text(encoding='utf-8'))
    if [r['prediction'] for r in saved_old]!=base_oof:raise ValueError('Cross-fitted baseline not reproduced')
    candidates=[];answers={}
    for proximity in RECIPE['proximities']:
        for multiplier in RECIPE['new_weights']:
            oof=['']*len(rows)
            for om,nm,oo,nn,held in folds:
                s=fit_update(oldraw[om],oo,newraw[nm],nn,dims,proximity,multiplier)
                for i,c in zip(torch.where(held)[0].tolist(),codes(raw[held],s)):oof[i]=c
            metric=mixed_metrics(old,oof[:728],source)['exact_six_class']
            item=dict(proximity=proximity,new_weight=multiplier,old_train_oof=metric,baseline_old_train_oof=baseline_metrics,
                training_gates=training_gates(metric,baseline_metrics),all_rows_oof=mixed_metrics(rows,oof,source))
            candidates.append(item);answers[proximity,multiplier]=oof
            log('cv',proximity=proximity,new_weight=multiplier,eligible=all(item['training_gates'].values()),recalls=metric['per_class_recall'])
    winner=max(candidates,key=selection_key);pr,mult=winner['proximity'],winner['new_weight'];oof=answers[pr,mult]
    frozen=dict(stage='before_validation_features_and_predictions',winner=winner,candidates=candidates,fold_source_groups=mapping,
        training_record_ids=[r['record_id'] for r in rows],baseline_old_train_oof=baseline_metrics,
        selection_status='eligible_candidate' if all(winner['training_gates'].values()) else 'NO_ELIGIBLE_UPDATE_DIAGNOSTIC_ONLY')
    source.write_json(out/'selection_frozen.json',frozen)
    source.write_json(out/'oof_answers.json',[dict(record_id=r['record_id'],source_group=r['source_group'],allowed_codes=r['allowed_codes'],prediction=c,fold=f) for r,c,f in zip(rows,oof,assignments)])
    s=fit_update(oldraw,old,newraw,newrows,dims,pr,mult);save_file({k:v.contiguous() for k,v in s.items()},str(out/'facies_head.safetensors'))
    base_state,_=make_state(oldraw,old,dims);stored_base=load_file(str(ref/'facies_head.safetensors'))
    delta=float((base_state['coefficients']-stored_base['coefficients']).abs().max())
    if delta>1e-10:raise ValueError('Full prior not reproduced')
    result=dict(head_sha256=source.digest(out/'facies_head.safetensors'),selection_frozen_sha256=source.digest(out/'selection_frozen.json'),
        selected=dict(proximity=pr,new_weight=mult),oof=winner,train=mixed_metrics(rows,codes(raw,s),source),
        baseline_head_coefficient_max_delta=delta,distribution_approved=False,installed_in_application=False,encoder_or_chat_weights_changed=False)
    # Frozen choice above; now a single evaluation on the unchanged repeated val.
    vraw=load_file(str(Path(ref_cfg['reference'])/'validation_features.safetensors'))['raw_val'][:,:dims[0]]
    target=[r['conversations'][1]['value'] for r in val];pred=codes(vraw,s);baseline=codes(vraw,stored_base)
    actual=source.classification_metrics(target,pred);bm=source.classification_metrics(target,baseline)
    checks=preservation(actual,bm);checks.update(B_improved=actual['per_class_recall']['B']>bm['per_class_recall']['B'],
        C_improved=actual['per_class_recall']['C']>bm['per_class_recall']['C'],accuracy=actual['correct']>=bm['correct'],macro_f1=actual['macro_f1']+1e-12>=bm['macro_f1'])
    introduced_protected=sum(e in PROTECTED and b==e and c!=e for e,b,c in zip(target,baseline,pred))
    checks['no_protected_correct_case_lost']=introduced_protected==0;checks['train_eligible']=all(winner['training_gates'].values())
    result.update(val=actual,baseline_val=bm,validation_gates=checks,introduced_protected_errors=introduced_protected,
        paired=dict(fixed=sum(b!=e and c==e for e,b,c in zip(target,baseline,pred)),introduced=sum(b==e and c!=e for e,b,c in zip(target,baseline,pred))))
    result['status']='research_gates_passed_requires_independent_validation' if all(checks.values()) else 'not_promoted_preservation_or_target_gate_failed'
    source.write_json(out/'result.json',result)
    source.write_json(out/'val_answers.json',[dict(record_id=r['metadata']['record_id'],expected_code=e,prediction=c,baseline_prediction=b) for r,e,c,b in zip(val,target,pred,baseline)])
    contract.update(type='frozen_qwen_prior_regularized_partial_ridge_head',head_sha256=result['head_sha256'],acceptance_status=result['status'],training_recipe=RECIPE,
        selected_profile=dict(profile='qwen_post',**result['selected']),training_source_license=cfg['source_rights'],distribution_approved=False)
    source.write_json(out/'MODEL_CONTRACT.json',contract);log('complete',selected=result['selected'],val=actual,checks=checks,status=result['status'])


if __name__=='__main__':main()
