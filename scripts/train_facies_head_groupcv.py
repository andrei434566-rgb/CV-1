"""Train-only source-group CV, expanded source coverage and optional texture.

The previous50 validation crops are evaluated only AFTER model selection
is frozen. Qwen remains frozen. This is a six-class research classifier.
"""
from collections import Counter
import argparse
import gc
import json
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import train_unibo_visual_facies as source
from scripts.train_qwen_visual_facies_head import pool_features,predict_head as predict_v1
from scripts.facies_head_v2_features import (
    texture_features,make_folds,sample_weights,prepare_state,normalize,ridge_coefficients,predict,
    PROFILES,ALPHAS,WEIGHTINGS,TEXTURE_VERSION,
)

RECIPE={'profiles':list(PROFILES),'ridge_alphas':list(ALPHAS),'weightings':list(WEIGHTINGS),'folds':3,
    'selection':'all six OOF recalls nonzero, then pooled OOF macroF1, mean S-group accuracy, larger lambda, baseline profile order',
    'qwen':'original frozen Qwen3-VL-2B vision, same pooled features asv1','image_side':300,
    'texture_version':TEXTURE_VERSION,'seed':20261030,'validation':'previous50 exactly unchanged, used once after selection freeze'}


def cross_validate(raw,labels,groups,qdim,pre_dim,tdim,log):
    import torch
    assignments,mapping=make_folds(labels.numpy(),groups)
    candidates={(p,w,a):torch.full((len(labels),),-1,dtype=torch.long) for p in PROFILES for w in WEIGHTINGS for a in ALPHAS}
    for profile in PROFILES:
        for fold in range(3):
            tr=torch.tensor(assignments!=fold); va=~tr
            state=prepare_state(raw[tr],profile,qdim,pre_dim,tdim)
            x=normalize(raw[tr],state); held=normalize(raw[va],state)
            kernel=x@x.T; cross=held@x.T
            y=labels[tr]; onehot=torch.nn.functional.one_hot(y,6).double()
            for weighting in WEIGHTINGS:
                weights=sample_weights(y,[g for g,keep in zip(groups,tr.tolist()) if keep],weighting)
                sw=weights.sqrt(); gram=kernel*sw[:,None]*sw[None,:]; rhs=onehot*sw[:,None]
                for alpha in ALPHAS:
                    dual=torch.linalg.solve(gram+alpha*torch.eye(len(y),dtype=torch.float64),rhs)
                    scores=(cross*sw[None,:])@dual
                    candidates[profile,weighting,alpha][va]=scores.argmax(-1)
            log('cv_fold',profile=profile,fold=fold,train=int(tr.sum()),held=int(va.sum()))
    results=[]
    references=[source.CODES[i] for i in labels.tolist()]
    for (profile,weighting,alpha),pred in candidates.items():
        if (pred<0).any(): raise ValueError('Incomplete out-of-fold predictions')
        m=source.classification_metrics(references,[source.CODES[i] for i in pred.tolist()])
        group_accuracy=sum(sum(int(pred[i])==int(labels[i]) for i,g in enumerate(groups) if g==s)/groups.count(s) for s in set(groups))/len(set(groups))
        results.append({'profile':profile,'weighting':weighting,'alpha':alpha,'metrics':m,
                        'all_classes_recalled':all(v>0 for v in m['per_class_recall'].values()),'mean_source_group_accuracy':group_accuracy})
    winner=max(results,key=lambda r:(r['all_classes_recalled'],r['metrics']['macro_f1'],r['mean_source_group_accuracy'],r['alpha'],-PROFILES.index(r['profile'])))
    return winner,results,mapping,candidates[winner['profile'],winner['weighting'],winner['alpha']]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,default=ROOT.parent/'vlm_core_v1/data/unibo_facies_expanded_20261030_v2')
    p.add_argument('--reference',type=Path,default=ROOT.parent/'vlm_core_v1/outputs/core_qwen_visual_facies_head_20261030_ridge')
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args(); args.data,args.reference,args.output=(x.resolve() for x in (args.data,args.reference,args.output))
    if args.output.exists(): raise ValueError('Choose a new experiment output')
    raw_train,raw_val=source.read_rows(args.data/'train.jsonl'),source.read_rows(args.data/'val.jsonl')
    source.validate_facies_split(raw_train,raw_val)
    train,val=[source.transform_facies(r) for r in raw_train],[source.transform_facies(r) for r in raw_val]
    old_train=source.read_rows(args.reference/'train.jsonl'); old_val=source.read_rows(args.reference/'val.jsonl')
    if val!=old_val: raise ValueError('Validation set differs from v1')
    contract=json.loads((args.reference/'MODEL_CONTRACT.json').read_text(encoding='utf-8'))
    parent_cfg=json.loads((args.reference/'run_config.json').read_text(encoding='utf-8'))
    if source.package_versions()!=contract['package_versions']: raise ValueError('Feature cache environment differs; recompute features before mixing versions')
    for path,sha in parent_cfg['base_config_tokenizer_sha256'].items():
        if source.digest(path)!=sha: raise ValueError('Original processor/config differs')
    base=Path(parent_cfg['base'])
    if source.digest(base/'model.safetensors')!=contract['base_weights_sha256']: raise ValueError('Original Qwen weights differ')
    args.output.mkdir(parents=True)
    source.write_rows(args.output/'train.jsonl',train); source.write_rows(args.output/'val.jsonl',val)
    cfg={'recipe':RECIPE,'base':str(base),'base_revision':contract['base_revision'],'base_weights_sha256':contract['base_weights_sha256'],
        'base_config_tokenizer_sha256':parent_cfg['base_config_tokenizer_sha256'],'reference':str(args.reference),'data':str(args.data),
        'source_license':'CC BY-NC-SA 4.0','role':'noncommercial_six_facies_groupcv_head_research','encoder_or_chat_weights_changed':False,
        'train_images':len(train),'validation_images':len(val),'prepared_test_pixels_opened':0,
        'package_versions':source.package_versions(),'python_version':sys.version,
        'dependencies':{str(x):source.digest(x) for x in (Path(__file__),ROOT/'scripts/facies_head_v2_features.py',
            ROOT/'scripts/train_qwen_visual_facies_head.py',ROOT/'scripts/train_unibo_visual_facies.py',args.data/'summary.json',
            args.data/'train.jsonl',args.data/'val.jsonl',args.reference/'features.safetensors',args.reference/'train.jsonl',
            args.reference/'val.jsonl',args.reference/'facies_head.safetensors',args.reference/'MODEL_CONTRACT.json')},
        'criteria':{'minimum_correct':28,'macro_f1_minimum':.4861304436304436+.03,'all_six_nonzero_recall':True,
                    'prodelta_min_correct':2,'poorly_drained_floodplain_min_correct':3},
        'notes':['Hyperparameters selected only from train-group OOF results','Same50 reused development examples, not a final independent test',
                 'New crops are correlated, S-groups not verified boreholes','No automatic92-class mapping or app activation']}
    source.write_json(args.output/'run_config.json',cfg)
    shutil.copy2(__file__,args.output/'training_script_snapshot.py')
    shutil.copy2(args.data/'LICENSE_SOURCES.md',args.output/'LICENSE_SOURCES.md')
    from scripts.train_qwen_source_pilot import configure_offline_environment,_deny_network
    configure_offline_environment(); sys.addaudithook(_deny_network)
    import torch
    from PIL import Image
    from safetensors import safe_open
    from safetensors.torch import save_file,load_file
    torch.set_num_threads(4); torch.set_num_interop_threads(1); torch.manual_seed(RECIPE['seed'])
    def log(event,**values):
        data={'event':event,**values}
        with (args.output/'progress.jsonl').open('a',encoding='utf-8') as f: f.write(json.dumps(data)+'\n')
        print(json.dumps(data),flush=True)
    with safe_open(str(args.reference/'features.safetensors'),framework='pt') as cache: cached=cache.get_tensor('train')
    index={r['metadata']['record_id']:i for i,r in enumerate(old_train)}
    qdim=cached.shape[1]; qvectors=[None]*len(train); missing=[]; textures=[]
    for i,row in enumerate(train):
        identity=row['metadata']['record_id']
        if identity in index:
            prior=old_train[index[identity]]
            if prior['metadata']['image_sha256']!=row['metadata']['image_sha256']: raise ValueError('Reused crop hash differs')
            qvectors[i]=cached[index[identity]]
        else: missing.append(i)
        with Image.open(row['image']) as im: textures.append(torch.from_numpy(texture_features(im)))
    if missing:
        from transformers import AutoProcessor,Qwen3VLForConditionalGeneration
        if not torch.cuda.is_available() or torch.cuda.mem_get_info()[0]<5*1024**3: raise RuntimeError('Needs5GiB free CUDA memory for new features')
        processor=AutoProcessor.from_pretrained(base,local_files_only=True)
        model=Qwen3VLForConditionalGeneration.from_pretrained(base,local_files_only=True,dtype=torch.float16,attn_implementation='sdpa',device_map='cuda')
        model.eval(); model.requires_grad_(False)
        for count,i in enumerate(missing,1):
            with Image.open(train[i]['image']) as im:
                image=im.convert('RGB'); image.thumbnail((300,300))
            inputs=source.prompt_inputs(processor,image,source.PROMPT).to('cuda')
            with torch.inference_mode():
                visual=model.model.visual(inputs['pixel_values'].to(model.dtype),grid_thw=inputs['image_grid_thw'],return_dict=True)
                qvectors[i]=pool_features(visual).cpu()
            del visual,inputs
            if count%25==0 or count==len(missing): log('new_features',row=count,total=len(missing))
        del model,processor; gc.collect(); torch.cuda.empty_cache()
    q=torch.stack(qvectors); texture=torch.stack(textures); raw=torch.cat((q,texture),1)
    if q.shape[1]!=contract['feature_dimension']: raise ValueError('Qwen feature width differs')
    labels=torch.tensor([source.CODES.index(r['conversations'][1]['value']) for r in train]); groups=[r['metadata']['source_group'] for r in train]
    pre_dim=2*json.loads((base/'config.json').read_text(encoding='utf-8'))['vision_config']['hidden_size']
    save_file({'qwen_train':q,'texture_train':texture},str(args.output/'train_features.safetensors'))
    winner,results,folds,oof=cross_validate(raw,labels,groups,qdim,pre_dim,texture.shape[1],log)
    frozen={'stage':'before_validation_features_and_predictions','winner':winner,'candidates':results,'fold_source_groups':folds,
            'training_record_ids':[r['metadata']['record_id'] for r in train]}
    source.write_json(args.output/'selection_frozen.json',frozen)
    log('selection_frozen',profile=winner['profile'],weighting=winner['weighting'],alpha=winner['alpha'],oof_macro_f1=winner['metrics']['macro_f1'])
    state=prepare_state(raw,winner['profile'],qdim,pre_dim,texture.shape[1])
    weights=sample_weights(labels,groups,winner['weighting'])
    state['coefficients']=ridge_coefficients(normalize(raw,state),labels,weights,winner['alpha'])
    save_file({k:v.contiguous() for k,v in state.items()},str(args.output/'facies_head.safetensors'))
    # Validation feature tensor and labels enter scoring only after selection freeze.
    with safe_open(str(args.reference/'features.safetensors'),framework='pt') as cache: qval=cache.get_tensor('val')
    tv=[]
    for row in val:
        with Image.open(row['image']) as im: tv.append(torch.from_numpy(texture_features(im)))
    raw_val=torch.cat((qval,torch.stack(tv)),1)
    save_file({'raw_val':raw_val},str(args.output/'validation_features.safetensors'))
    result={'role':cfg['role'],'source_license':cfg['source_license'],'installed_in_application':False,'geologist_approved':False,
        'encoder_or_chat_weights_changed':False,'selection_frozen_sha256':source.digest(args.output/'selection_frozen.json'),
        'winner':{k:winner[k] for k in ('profile','weighting','alpha')},'oof_metrics':winner['metrics'],
        'head_sha256':source.digest(args.output/'facies_head.safetensors'),'qdim':qdim,'pre_dim':pre_dim,'texture_dim':texture.shape[1]}
    for split,rows,features in (('train',train,raw),('val',val,raw_val)):
        scores=predict(features,state); predicted=scores.argmax(1).tolist()
        if not torch.allclose(scores,predict(features,load_file(str(args.output/'facies_head.safetensors')))): raise ValueError('Reloaded head differs')
        answers=[{'record_id':r['metadata']['record_id'],'source_group':r['metadata']['source_group'],'image_sha256':r['metadata']['image_sha256'],
            'expected_code':r['conversations'][1]['value'],'prediction':source.CODES[predicted[i]],'scores':dict(zip(source.CODES,scores[i].tolist()))} for i,r in enumerate(rows)]
        source.write_json(args.output/(split+'_answers.json'),answers)
        result[split]=source.classification_metrics([r['expected_code'] for r in answers],[r['prediction'] for r in answers])
    reference_scores=predict_v1(qval,load_file(str(args.reference/'facies_head.safetensors')))
    reference=source.classification_metrics([r['conversations'][1]['value'] for r in val],[source.CODES[i] for i in reference_scores.argmax(1).tolist()])
    if reference['correct']!=26: raise ValueError('Incumbent reference no longer matches26/50')
    result['incumbent']=reference; m=result['val']; recalls=m['per_class_recall']
    result['gates']={'accuracy':m['correct']>=28,'macro_f1':m['macro_f1']>=cfg['criteria']['macro_f1_minimum'],
        'all_six_recalled':all(x>0 for x in recalls.values()),'prodelta_improved':recalls['C']>=.2,'PDF_preserved':recalls['B']>=.375}
    source.write_json(args.output/'result.json',result)
    source.write_json(args.output/'acceptance_review.json',{'status':'research_candidate_requires_review_not_general_assistant',
        'installed_in_application':False,'geologist_approved':False,'research_gates':result['gates'],'source_license':cfg['source_license']})
    model_contract=dict(contract); model_contract.update({'type':'frozen_qwen_visual_texture_groupcv_ridge_head',
        'feature_contract_version':'qwen3vl_visual_texture_groupcv_v2','head_sha256':result['head_sha256'],
        'feature_dimension':int(len(state['mean'])),'qwen_feature_dimension':qdim,'texture_dimension':texture.shape[1],
        'texture_version':TEXTURE_VERSION,'selected_profile':result['winner'],'feature_spec':RECIPE})
    source.write_json(args.output/'MODEL_CONTRACT.json',model_contract)
    log('complete',correct=m['correct'],macro_f1=m['macro_f1'],profile=winner['profile'],all_gates=all(result['gates'].values()))


if __name__=='__main__': main()
