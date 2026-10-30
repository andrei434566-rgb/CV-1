"""Train a new source-facies head with caption-grounded web crops.

The encoder is frozen. Generic floodplain examples supervise A+B, never
an invented drainage subclass. Hyperparameters use train-group CV only.
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
from scripts.train_qwen_visual_facies_head import pool_features,predict_head
from scripts.facies_head_v2_features import make_folds,prepare_state,normalize,predict
from scripts.partial_facies_ridge import fit_partial,row_weights,mixed_metrics
from scripts.prepare_web_facies_20261031 import pixel_digest

ALPHAS=(.1,1.,10.,100.)
EXTERNAL_WEIGHTS=(1.,4.)
RECIPE={'profile':'qwen_post','folds':3,'ridge_alphas':ALPHAS,'external_weights':EXTERNAL_WEIGHTS,
    'supervision':'Exact A..F, or (sA+sB-1)^2 + sC^2+sD^2+sE^2+sF^2 for generic floodplain',
    'partial_AB_difference':'unsupervised; no guessed drainage class',
    'selection':'All six exact OOF recalls nonzero, exact OOF macroF1, mean source-group admissible accuracy, larger alpha, smaller external weight',
    'encoder':'original frozen Qwen3-VL-2B, image side300, same pre/post mean/std cache as v2',
    'validation':'Previous50 unchanged and evaluated after frozen train-only selection',
    'seed':20261031,'commercial_deployment':False}


def cv(raw,rows,assignments,qdim,pre_dim,tdim,log):
    import torch
    all_results=[]; candidates={}
    for alpha in ALPHAS:
        for multiplier in EXTERNAL_WEIGHTS:
            predicted=['']*len(rows)
            for fold in range(3):
                tr=torch.tensor([f!=fold for f in assignments]); held=~tr
                subset=[r for r,keep in zip(rows,tr.tolist()) if keep]
                state=prepare_state(raw[tr],'qwen_post',qdim,pre_dim,tdim)
                allowed=[r['allowed_codes'] for r in subset]
                weights=row_weights(allowed,[r['external'] for r in subset],multiplier)
                state['coefficients']=fit_partial(normalize(raw[tr],state),allowed,weights,alpha)
                codes=[source.CODES[i] for i in predict(raw[held],state).argmax(1).tolist()]
                for i,p in zip(torch.where(held)[0].tolist(),codes): predicted[i]=p
            if '' in predicted: raise ValueError('Incomplete OOF predictions')
            metrics=mixed_metrics(rows,predicted,source)
            item={'alpha':alpha,'external_weight':multiplier,'metrics':metrics,
                'all_exact_classes_recalled':all(v>0 for v in metrics['exact_six_class']['per_class_recall'].values())}
            all_results.append(item); candidates[alpha,multiplier]=predicted
            log('cv_candidate',alpha=alpha,external_weight=multiplier,macro_f1=metrics['exact_six_class']['macro_f1'])
    winner=max(all_results,key=selection_key)
    return winner,all_results,candidates[winner['alpha'],winner['external_weight']]


def selection_key(r):
    return (r['all_exact_classes_recalled'],r['metrics']['exact_six_class']['macro_f1'],
        r['metrics']['mean_source_group_accuracy'],r['alpha'],-r['external_weight'])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--web-data',type=Path,default=ROOT.parent/'vlm_core_v1/data/web_facies_20261031_v3_clean')
    p.add_argument('--reference',type=Path,default=ROOT.parent/'vlm_core_v1/outputs/core_qwen_facies_groupcv_20261030_v2')
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args(); web,reference,output=(x.resolve() for x in (args.web_data,args.reference,args.output))
    if output.exists(): raise ValueError('Choose a new experiment directory')
    cfg=json.loads((reference/'run_config.json').read_text(encoding='utf-8'))
    contract=json.loads((reference/'MODEL_CONTRACT.json').read_text(encoding='utf-8'))
    old_data=Path(cfg['data'])
    original_train,original_val=(source.read_rows(old_data/(s+'.jsonl')) for s in ('train','val'))
    source.validate_facies_split(original_train,original_val)
    train,val=([source.transform_facies(r) for r in records] for records in (original_train,original_val))
    if train!=source.read_rows(reference/'train.jsonl') or val!=source.read_rows(reference/'val.jsonl'): raise ValueError('Reference row order differs')
    extra=source.read_rows(web/'train.jsonl'); summary=json.loads((web/'summary.json').read_text(encoding='utf-8'))
    if summary.get('do_not_train') or summary['crops']!=24: raise ValueError('Unapproved web preparation')
    if source.package_versions()!=contract['package_versions']: raise ValueError('Feature extraction environment differs')
    base=Path(cfg['base'])
    if source.digest(base/'model.safetensors')!=contract['base_weights_sha256']: raise ValueError('Original base weights differ')
    for path,expected in cfg['base_config_tokenizer_sha256'].items():
        if source.digest(path)!=expected: raise ValueError('Original processor/config differs')
    from PIL import Image
    from scripts.prepare_web_facies_20261031 import SOURCE_SHA,SOURCES
    seen={r['metadata']['pixel_sha256'] for r in train+val}; seen_bytes={r['metadata']['image_sha256'] for r in train+val}
    for r in extra:
        if (r['split']!='train' or r['training_approved'] is not True or r['geologist_approved'] is not False
            or r['source']!=SOURCES[r['source_id']] or r['source_file_sha256']!=SOURCE_SHA[r['source_file']]): raise ValueError('Web source approval differs')
        if source.digest(r['image'])!=r['image_sha256']: raise ValueError('Web crop hash differs')
        with Image.open(r['image']) as im:
            pixels=pixel_digest(im.convert('RGB'))
            if pixels!=r['pixel_sha256'] or list(im.size)!=r['image_size_px']: raise ValueError('Web crop pixels differ')
        if pixels in seen or r['image_sha256'] in seen_bytes: raise ValueError('Duplicate source crop')
        seen.add(pixels); seen_bytes.add(r['image_sha256'])
    rows=[{'record_id':r['metadata']['record_id'],'image':r['image'],'image_sha256':r['metadata']['image_sha256'],
        'allowed_codes':[r['conversations'][1]['value']],'source_group':r['metadata']['source_group'],'external':False} for r in train]
    rows += [dict(r,external=True) for r in extra]
    output.mkdir(parents=True)
    source.write_rows(output/'train.jsonl',rows); source.write_rows(output/'val.jsonl',val)
    run={'recipe':RECIPE,'base':str(base),'base_revision':contract['base_revision'],'base_weights_sha256':contract['base_weights_sha256'],
        'reference':str(reference),'web_data':str(web),'base_source_train_count':len(train),'additional_train_crops':len(extra),
        'train_count':len(rows),'val_count':len(val),'sources':summary['sources'],'source_license':'UniBo CC BY-NC-SA4.0; new cropped figures CC BY4.0; combined weights research only CC BY-NC-SA4.0',
        'encoder_or_chat_weights_changed':False,'installed_in_application':False,'prepared_test_pixels_opened':0,
        'criteria':{'minimum_correct':28,'macro_f1_minimum':.5161304436304436,'prodelta_min_correct':2,'PDF_min_correct':3,'FS_min_correct':10},
        'dependencies':{str(path):source.digest(path) for path in (Path(__file__),ROOT/'scripts/partial_facies_ridge.py',ROOT/'scripts/facies_head_v2_features.py',
            ROOT/'scripts/train_qwen_visual_facies_head.py',ROOT/'scripts/prepare_web_facies_20261031.py',
            web/'train.jsonl',web/'summary.json',reference/'train_features.safetensors',reference/'train.jsonl',reference/'val.jsonl',reference/'MODEL_CONTRACT.json')}}
    source.write_json(output/'run_config.json',run); shutil.copy2(__file__,output/'training_script_snapshot.py')
    from scripts.train_qwen_source_pilot import configure_offline_environment,_deny_network
    configure_offline_environment(); sys.addaudithook(_deny_network)
    import torch
    from safetensors.torch import load_file,save_file
    torch.set_num_threads(4); torch.set_num_interop_threads(1); torch.manual_seed(RECIPE['seed'])
    def log(event,**values):
        data={'event':event,**values}
        with (output/'progress.jsonl').open('a',encoding='utf-8') as stream: stream.write(json.dumps(data)+'\n')
        print(json.dumps(data),flush=True)
    cache=load_file(str(reference/'train_features.safetensors'))['qwen_train']
    if len(cache)!=len(train) or cache.shape[1]!=contract['qwen_feature_dimension']: raise ValueError('Qwen reference features differ')
    if not torch.cuda.is_available() or torch.cuda.mem_get_info()[0]<5*1024**3: raise RuntimeError('Needs5GiB free CUDA memory')
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
        if i%8==0: log('web_features',row=i,total=len(extra))
    del model,processor; gc.collect(); torch.cuda.empty_cache()
    raw=torch.cat((cache,torch.stack(features)),0)
    save_file({'train':raw},str(output/'features.safetensors'))
    labels=torch.tensor([source.CODES.index(r['allowed_codes'][0]) for r in rows[:len(train)]])
    assignments,mapping=make_folds(labels.numpy(),[r['source_group'] for r in rows[:len(train)]])
    assignments=assignments.tolist()
    # Keep the two new sources whole in folds chosen from train metadata only.
    c_counts=[sum(int(y)==2 and f==fold for y,f in zip(labels,assignments)) for fold in range(3)]
    sydney_fold=min(range(3),key=lambda f:(c_counts[f],f)); sizes=[assignments.count(f) for f in range(3)]
    sizes[sydney_fold]+=12; fujian_fold=min(range(3),key=lambda f:(sizes[f],f))
    mapping.update({'WEB_Sydney_FA7_unknown_wells':sydney_fold,'WEB_Fujian_LJ01':fujian_fold})
    assignments += [mapping[r['source_group']] for r in rows[len(train):]]
    for fold in range(3):
        tr_groups={r['source_group'] for r,f in zip(rows,assignments) if f!=fold}
        held_groups={r['source_group'] for r,f in zip(rows,assignments) if f==fold}
        if tr_groups&held_groups: raise ValueError('Source group leakage')
    qdim=contract['qwen_feature_dimension']; pre_dim=contract['pre_merger_feature_dimension']; tdim=contract['texture_dimension']
    winner,results,oof=cv(raw,rows,assignments,qdim,pre_dim,tdim,log)
    frozen={'stage':'before_validation_features_and_predictions','winner':winner,'candidates':results,'fold_source_groups':mapping,
        'training_record_ids':[r['record_id'] for r in rows]}
    source.write_json(output/'selection_frozen.json',frozen)
    source.write_json(output/'oof_answers.json',[{'record_id':r['record_id'],'source_group':r['source_group'],
        'allowed_codes':r['allowed_codes'],'prediction':p,'fold':f,'external':r['external']} for r,p,f in zip(rows,oof,assignments)])
    state=prepare_state(raw,'qwen_post',qdim,pre_dim,tdim)
    allowed=[r['allowed_codes'] for r in rows]; weights=row_weights(allowed,[r['external'] for r in rows],winner['external_weight'])
    state['coefficients']=fit_partial(normalize(raw,state),allowed,weights,winner['alpha'])
    save_file({k:v.contiguous() for k,v in state.items()},str(output/'facies_head.safetensors'))
    result={'role':'noncommercial_partial_caption_facies_head_research','encoder_or_chat_weights_changed':False,
        'head_sha256':source.digest(output/'facies_head.safetensors'),'selection_frozen_sha256':source.digest(output/'selection_frozen.json'),
        'selected':{k:winner[k] for k in ('alpha','external_weight')},'oof':winner['metrics'],'train':mixed_metrics(rows,[source.CODES[i] for i in predict(raw,state).argmax(1).tolist()],source)}
    # Only now load any validation feature tensor for scoring; never choose by it.
    raw_val=load_file(str(reference/'validation_features.safetensors'))['raw_val'][:,:qdim]
    predicted=[source.CODES[i] for i in predict(raw_val,state).argmax(1).tolist()]
    expected=[r['conversations'][1]['value'] for r in val]
    result['val']=source.classification_metrics(expected,predicted)
    source.write_json(output/'val_answers.json',[{'record_id':r['metadata']['record_id'],'source_group':r['metadata']['source_group'],
        'expected_code':e,'prediction':p} for r,e,p in zip(val,expected,predicted)])
    for name,path in (('v1',ROOT/'research_models/unibo_qwen_facies_head_v1'),('v2',reference)):
        s=load_file(str(path/'facies_head.safetensors'))
        scores=predict_head(raw_val,s) if name=='v1' else predict(raw_val,s)
        codes=[source.CODES[i] for i in scores.argmax(1).tolist()]
        result[name]=source.classification_metrics(expected,codes)
        result['paired_vs_'+name]={'fixed':sum(a!=e and b==e for a,b,e in zip(codes,predicted,expected)),
            'introduced':sum(a==e and b!=e for a,b,e in zip(codes,predicted,expected))}
    recalls=result['val']['per_class_recall']; criteria=run['criteria']; m=result['val']
    gates={'accuracy':m['correct']>=criteria['minimum_correct'],'macro_f1':m['macro_f1']>=criteria['macro_f1_minimum'],
        'all_six_recalled':all(v>0 for v in recalls.values()),'prodelta':recalls['C']*10>=criteria['prodelta_min_correct'],
        'PDF':recalls['B']*8>=criteria['PDF_min_correct'],'FS':recalls['E']*12>=criteria['FS_min_correct']}
    result['gates']=gates
    status='accepted_source_development_research_only' if all(gates.values()) else 'not_promoted_failed_prospective_gates'
    result['status']=status; source.write_json(output/'result.json',result)
    contract.update({'type':'frozen_qwen_partial_label_ridge_head','head_sha256':result['head_sha256'],
        'acceptance_status':status,'selected_profile':dict(profile='qwen_post',**result['selected']),
        'training_recipe':RECIPE,'training_source_license':run['source_license']})
    source.write_json(output/'MODEL_CONTRACT.json',contract)
    source.write_json(output/'acceptance_review.json',{'status':status,'installed_in_application':False,'geologist_approved':False,
        'gates':gates,'head_sha256':result['head_sha256'],'review':'Await independent numerical and source audit',
        'validation_note':'Repeated50 development crops, three proxy source groups; not final independent accuracy'})
    log('complete',correct=m['correct'],macro_f1=m['macro_f1'],selected=result['selected'],gates=gates)


if __name__=='__main__': main()
