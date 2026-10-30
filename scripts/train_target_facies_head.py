"""Train target B/C head on frozen local Qwen vision and caption-grounded data.

Selection is source-group CV on train only. No language/vision LoRA and no
automatic application promotion. The one outcrop photo has low fixed weight.
"""
import argparse
import gc
import json
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import train_unibo_visual_facies as source
from scripts.train_qwen_visual_facies_head import pool_features
from scripts.facies_head_v2_features import prepare_state,normalize,predict
from scripts.partial_facies_ridge import fit_partial,row_weights,mixed_metrics
from scripts.prepare_target_facies_20261031 import SOURCES,PHOTO_FILES,pixel_digest

ALPHAS=(.1,1.,10.,100.)
BOOSTS=(1.,2.,4.)
RECIPE=dict(profile='qwen_post',alphas=ALPHAS,target_class_boosts=BOOSTS,
    outcrop_weight=.25,old_external_weight=1.,seed=20261031,folds=3,
    selection='All six recalled, min(B,C) exact OOF recall, mean(B,C) recall, exact macroF1, mean group accuracy, larger alpha, smaller boost',
    partial_AB='A+B sum supervised; A-B direction not supervised',
    frozen_encoder=True,chat_weights_changed=False,
    input='Original RGB crop thumbnail300, original frozen Qwen3-VL-2B processor',
    validation='Previous50 unchanged, loaded only after train-only selection frozen')
CRITERIA=dict(minimum_correct=29,macro_f1_minimum=.5867044783173816,
    PDF_min_correct=3,prodelta_min_correct=3,FS_min_correct=10)


def weights_for(rows,boost):
    import torch
    weights=row_weights([r['allowed_codes'] for r in rows],[r['external'] for r in rows],1.)
    weights*=torch.tensor([boost if r['allowed_codes'] in (['B'],['C']) else 1. for r in rows],dtype=torch.float64)
    weights*=torch.tensor([RECIPE['outcrop_weight'] if r.get('domain')=='outcrop_analogue' else 1. for r in rows],dtype=torch.float64)
    return weights


def key(item):
    m=item['metrics']; rec=m['exact_six_class']['per_class_recall']
    return (all(v>0 for v in rec.values()),min(rec['B'],rec['C']),
        (rec['B']+rec['C'])/2,m['exact_six_class']['macro_f1'],
        m['mean_source_group_accuracy'],item['alpha'],-item['target_boost'])


def fit(raw,rows,alpha,boost,dims):
    state=prepare_state(raw,'qwen_post',*dims)
    state['coefficients']=fit_partial(normalize(raw,state),[r['allowed_codes'] for r in rows],weights_for(rows,boost),alpha)
    return state


def codes(raw,state):
    return [source.CODES[i] for i in predict(raw,state).argmax(1).tolist()]


def gates(metrics):
    rec=metrics['per_class_recall']
    return dict(accuracy=metrics['correct']>=CRITERIA['minimum_correct'],
        macro_f1=metrics['macro_f1']>=CRITERIA['macro_f1_minimum'],
        all_six_recalled=all(v>0 for v in rec.values()),
        PDF=round(rec['B']*8)>=CRITERIA['PDF_min_correct'],
        prodelta=round(rec['C']*10)>=CRITERIA['prodelta_min_correct'],
        FS=round(rec['E']*12)>=CRITERIA['FS_min_correct'])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,default=ROOT.parent/'vlm_core_v1/data/target_facies_20261031_v4_clean')
    p.add_argument('--reference',type=Path,default=ROOT.parent/'vlm_core_v1/outputs/core_qwen_web_partial_facies_20261031_v3')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(); data,reference,output=(x.resolve() for x in (a.data,a.reference,a.output))
    if output.exists(): raise ValueError('Choose a new output directory')
    old_cfg=json.loads((reference/'run_config.json').read_text(encoding='utf-8'))
    contract=json.loads((reference/'MODEL_CONTRACT.json').read_text(encoding='utf-8'))
    rows=source.read_rows(reference/'train.jsonl'); val=source.read_rows(reference/'val.jsonl')
    extra=source.read_rows(data/'train.jsonl'); summary=json.loads((data/'summary.json').read_text(encoding='utf-8'))
    if len(rows)!=728 or len(val)!=50 or len(extra)!=15 or summary['do_not_train']: raise ValueError('Unexpected preparation/reference')
    if summary['sources']!=SOURCES: raise ValueError('Source provenance differs')
    if source.package_versions()!=contract['package_versions']: raise ValueError('Feature extraction environment changed')
    base=Path(old_cfg['base'])
    from scripts.classify_with_qwen_facies_head import streamed_sha
    if streamed_sha(base/'model.safetensors')!=contract['base_weights_sha256']: raise ValueError('Original base differs')
    for name,expected in contract['preprocessing_files_sha256'].items():
        if source.digest(base/name)!=expected: raise ValueError('Original processor changed')
    from PIL import Image
    seen={r['image_sha256'] for r in rows}|{r['metadata']['image_sha256'] for r in val}
    seen_pixels={r['metadata']['pixel_sha256'] for r in val}|{r['pixel_sha256'] for r in rows if 'pixel_sha256' in r}
    for r in extra:
        if (r['source']!=SOURCES[r['source_id']] or r['source_file_sha256']!=PHOTO_FILES[r['source_file']][3]
            or r['source_group']!=r['source']['source_group'] or r['domain']!=r['source']['domain']
            or not r['training_approved'] or r['geologist_approved'] or r['split']!='train'): raise ValueError('Source review differs')
        if source.digest(r['image'])!=r['image_sha256'] or r['image_sha256'] in seen: raise ValueError('Duplicate or changed image')
        with Image.open(r['image']) as im: pixels=pixel_digest(im.convert('RGB'))
        if pixels!=r['pixel_sha256'] or pixels in seen_pixels: raise ValueError('Duplicate or changed pixels')
        seen.add(r['image_sha256']); seen_pixels.add(pixels)
    rows+=extra; output.mkdir(parents=True)
    source.write_rows(output/'train.jsonl',rows); source.write_rows(output/'val.jsonl',val)
    cache_file=reference/'features.safetensors'
    deps=[Path(__file__),ROOT/'scripts/prepare_target_facies_20261031.py',ROOT/'scripts/partial_facies_ridge.py',
        ROOT/'scripts/facies_head_v2_features.py',ROOT/'scripts/train_qwen_visual_facies_head.py',
        data/'train.jsonl',data/'summary.json',reference/'train.jsonl',reference/'val.jsonl',cache_file,reference/'MODEL_CONTRACT.json',reference/'selection_frozen.json']
    run=dict(recipe=RECIPE,criteria=CRITERIA,reference=str(reference),data=str(data),base=str(base),
        train_count=len(rows),val_count=len(val),new_count=len(extra),sources=SOURCES,
        source_license='UniBo CC BY-NC-SA4.0; caption-grounded public papers CC BY4.0; BossPoint current article license dated2025. Combined head research only CC BY-NC-SA4.0',
        dependencies={str(f):source.digest(f) for f in deps},
        encoder_or_chat_weights_changed=False,installed_in_application=False,prepared_test_pixels_opened=0)
    source.write_json(output/'run_config.json',run); shutil.copy2(__file__,output/'training_script_snapshot.py')
    from scripts.train_qwen_source_pilot import configure_offline_environment,_deny_network
    configure_offline_environment(); sys.addaudithook(_deny_network)
    import torch
    from safetensors.torch import load_file,save_file
    torch.set_num_threads(4); torch.set_num_interop_threads(1); torch.manual_seed(RECIPE['seed'])
    def log(event,**kw):
        event=dict(event=event,**kw)
        with (output/'progress.jsonl').open('a',encoding='utf-8') as f: f.write(json.dumps(event)+'\n')
        print(json.dumps(event),flush=True)
    cache=load_file(str(cache_file))['train']
    if cache.shape!=(728,6144): raise ValueError('Cached feature dimensions differ')
    if not torch.cuda.is_available() or torch.cuda.mem_get_info()[0]<5*1024**3: raise RuntimeError('Need5GiB free CUDA')
    from transformers import AutoProcessor,Qwen3VLForConditionalGeneration
    processor=AutoProcessor.from_pretrained(base,local_files_only=True)
    model=Qwen3VLForConditionalGeneration.from_pretrained(base,local_files_only=True,dtype=torch.float16,attn_implementation='sdpa',device_map='cuda')
    model.eval(); model.requires_grad_(False); features=[]
    for i,row in enumerate(extra,1):
        with Image.open(row['image']) as im: image=im.convert('RGB'); image.thumbnail((300,300))
        inputs=source.prompt_inputs(processor,image,source.PROMPT).to('cuda')
        with torch.inference_mode():
            visual=model.model.visual(inputs['pixel_values'].to(model.dtype),grid_thw=inputs['image_grid_thw'],return_dict=True)
            features.append(pool_features(visual).cpu())
        del inputs,visual
        log('new_features',row=i,total=len(extra))
    del model,processor; gc.collect(); torch.cuda.empty_cache()
    raw=torch.cat((cache,torch.stack(features)))
    save_file({'train':raw},str(output/'features.safetensors'))
    baseline=load_file(str(reference/'facies_head.safetensors'))
    baseline_codes=codes(raw[-len(extra):],baseline)
    source.write_json(output/'new_sources_before_training.json',dict(metrics=mixed_metrics(extra,baseline_codes,source),
        answers=[dict(record_id=r['record_id'],prediction=c) for r,c in zip(extra,baseline_codes)]))
    frozen_old=json.loads((reference/'selection_frozen.json').read_text(encoding='utf-8'))
    mapping=dict(frozen_old['fold_source_groups'])
    for g in sorted({r['source_group'] for r in extra}):
        group=[r for r in extra if r['source_group']==g]; label=group[0]['allowed_codes'][0]
        def cost(f):
            assigned=[r for r in rows if r['source_group'] in mapping and mapping[r['source_group']]==f]
            return (sum(r['allowed_codes']==[label] for r in assigned),len(assigned),f)
        mapping[g]=min(range(3),key=cost)
    assignments=[mapping[r['source_group']] for r in rows]
    dims=(contract['qwen_feature_dimension'],contract['pre_merger_feature_dimension'],contract['texture_dimension'])
    candidates=[]; predictions={}
    for alpha in ALPHAS:
        for boost in BOOSTS:
            oof=['']*len(rows)
            for f in range(3):
                mask=torch.tensor([i!=f for i in assignments]); held=~mask
                subset=[r for r,k in zip(rows,mask.tolist()) if k]
                if {r['source_group'] for r in subset}&{r['source_group'] for r,k in zip(rows,held.tolist()) if k}: raise ValueError('Group leakage')
                state=fit(raw[mask],subset,alpha,boost,dims)
                for i,c in zip(torch.where(held)[0].tolist(),codes(raw[held],state)): oof[i]=c
            if '' in oof: raise ValueError('Incomplete OOF')
            item=dict(alpha=alpha,target_boost=boost,metrics=mixed_metrics(rows,oof,source))
            candidates.append(item); predictions[alpha,boost]=oof
            log('cv_candidate',alpha=alpha,boost=boost,recalls=item['metrics']['exact_six_class']['per_class_recall'])
    winner=max(candidates,key=key); alpha,boost=winner['alpha'],winner['target_boost']
    frozen=dict(stage='before_validation_features_and_predictions',winner=winner,candidates=candidates,
        fold_source_groups=mapping,training_record_ids=[r['record_id'] for r in rows])
    source.write_json(output/'selection_frozen.json',frozen)
    oof=predictions[alpha,boost]
    source.write_json(output/'oof_answers.json',[dict(record_id=r['record_id'],source_group=r['source_group'],
        allowed_codes=r['allowed_codes'],prediction=c,fold=f,external=r['external']) for r,c,f in zip(rows,oof,assignments)])
    state=fit(raw,rows,alpha,boost,dims)
    save_file({k:v.contiguous() for k,v in state.items()},str(output/'facies_head.safetensors'))
    result=dict(role='target_B_C_frozen_Qwen_research_candidate',encoder_or_chat_weights_changed=False,
        head_sha256=source.digest(output/'facies_head.safetensors'),selection_frozen_sha256=source.digest(output/'selection_frozen.json'),
        selected=dict(alpha=alpha,target_boost=boost,outcrop_weight=RECIPE['outcrop_weight']),oof=winner['metrics'],
        train=mixed_metrics(rows,codes(raw,state),source))
    # Validation is evaluated once after train-only selection. Ablation is diagnostic only.
    original_ref=Path(old_cfg['reference'])
    vraw=load_file(str(original_ref/'validation_features.safetensors'))['raw_val'][:,:dims[0]]
    expected=[r['conversations'][1]['value'] for r in val]; vp=codes(vraw,state)
    result['val']=source.classification_metrics(expected,vp)
    old_pred=codes(vraw,baseline); result['v3']=source.classification_metrics(expected,old_pred)
    result['paired_vs_v3']=dict(fixed=sum(a!=e and b==e for a,b,e in zip(old_pred,vp,expected)),
        introduced=sum(a==e and b!=e for a,b,e in zip(old_pred,vp,expected)))
    mask=torch.tensor([r.get('domain')!='outcrop_analogue' for r in rows]); no_outcrop=[r for r,k in zip(rows,mask.tolist()) if k]
    ablation=fit(raw[mask],no_outcrop,alpha,boost,dims)
    result['without_outcrop_fixed_selected_hyperparameters']=source.classification_metrics(expected,codes(vraw,ablation))
    source.write_json(output/'val_answers.json',[dict(record_id=r['metadata']['record_id'],source_group=r['metadata']['source_group'],expected_code=e,prediction=c)
        for r,e,c in zip(val,expected,vp)])
    result['gates']=gates(result['val'])
    result['status']='source_development_gates_passed_research_only' if all(result['gates'].values()) else 'not_promoted_failed_prospective_gates'
    source.write_json(output/'result.json',result)
    contract.update(type='frozen_qwen_target_weighted_partial_ridge_head',head_sha256=result['head_sha256'],
        acceptance_status=result['status'],training_recipe=RECIPE,selected_profile=dict(profile='qwen_post',**result['selected']),
        training_source_license=run['source_license'],input_scope='One core-interval crop; six source classes. Training includes low-weight outcrop analogue, not general chat or92 project facies')
    source.write_json(output/'MODEL_CONTRACT.json',contract)
    source.write_json(output/'acceptance_review.json',dict(status=result['status'],installed_in_application=False,
        geologist_approved=False,head_sha256=result['head_sha256'],review='Numerical and provenance audit pending'))
    log('complete',val=result['val'],selected=result['selected'],gates=result['gates'])


if __name__=='__main__': main()
