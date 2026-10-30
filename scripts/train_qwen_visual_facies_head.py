"""Train a small source-facies head on FROZEN Qwen visual features.

This is a separate image classifier, not a PEFT adapter or improved chat
model. All normalizer/linear weights use train only. One fixed ridge fit;
validation is read for evaluation only, never hyperparameter selection.
"""
import argparse
from collections import Counter
import gc
import json
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import train_unibo_visual_facies as source
from scripts.train_unibo_facies_discrimination import validate_parent_rows,PARENT_SHA

RECIPE={'encoder':'original frozen Qwen3-VL-2B vision, no LoRA','image_side':300,
        'features':'mean and population std of pre-merger and merged visual tokens',
        'normalization':'train mean/std floor1e-3, per-image L2, appended intercept',
        'classifier':'class-balanced multiclass ridge regression','ridge_lambda':1.0,
        'class_weights':'N/(6*n_class)','fit':'one fixed dual linear solve, float64',
        'seed':20261030,'activation':'none; closed-set argmax scores, not calibrated probabilities'}


def pool_features(vision_output):
    import torch
    vectors=[]
    for matrix in (vision_output.last_hidden_state,vision_output.pooler_output):
        if matrix.ndim!=2 or matrix.shape[0]<1: raise ValueError('Expected one image visual token matrix')
        matrix=matrix.float()
        vectors.extend((matrix.mean(0),matrix.std(0,unbiased=False)))
    result=torch.cat(vectors)
    if not torch.isfinite(result).all(): raise ValueError('Nonfinite visual features')
    return result


def normalized(features,mean,scale):
    import torch
    values=(features.double()-mean)/scale
    values=values/values.norm(dim=1,keepdim=True).clamp_min(1e-12)
    return torch.cat((values,torch.ones((len(values),1),dtype=values.dtype,device=values.device)),1)


def fit_head(features,labels):
    import torch
    if features.ndim!=2 or len(features)!=len(labels) or not torch.isfinite(features).all(): raise ValueError('Invalid train features')
    if set(labels.tolist())!=set(range(6)): raise ValueError('All six training classes required')
    values=features.double()
    mean=values.mean(0); scale=values.std(0,unbiased=False).clamp_min(1e-3)
    x=normalized(values,mean,scale)
    counts=torch.bincount(labels,minlength=6).double()
    weights=len(labels)/(6*counts[labels])
    weighted=x*weights.sqrt()[:,None]
    y=torch.nn.functional.one_hot(labels,num_classes=6).double()*weights.sqrt()[:,None]
    dual=torch.linalg.solve(weighted@weighted.T+RECIPE['ridge_lambda']*torch.eye(len(x),dtype=torch.float64),y)
    coefficients=weighted.T@dual
    if not torch.isfinite(coefficients).all(): raise ValueError('Nonfinite ridge solution')
    return {'mean':mean,'scale':scale,'coefficients':coefficients}


def predict_head(features,state):
    values=normalized(features,state['mean'],state['scale'])
    return values@state['coefficients']


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime',type=Path,default=ROOT.parent/'vlm_core_v1')
    parser.add_argument('--parent',type=Path,default=ROOT.parent/'vlm_core_v1/outputs/core_vlm_visual_facies_20261030_lora')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(); args.runtime,args.parent,args.output=(p.resolve() for p in (args.runtime,args.parent,args.output))
    if args.output.exists(): raise ValueError('Choose a new head experiment directory')
    parent_cfg=json.loads((args.parent/'run_config.json').read_text(encoding='utf-8'))
    records,val,_guard=validate_parent_rows(parent_cfg,args.parent,args.runtime)
    train=[r for r in records if r.get('research_task')=='unibo_six_choice_facies']
    args.output.mkdir(parents=True)
    source.write_rows(args.output/'train.jsonl',train); source.write_rows(args.output/'val.jsonl',val)
    config={'recipe':RECIPE,'base':parent_cfg['base'],'base_revision':parent_cfg['base_revision'],
        'base_config_tokenizer_sha256':parent_cfg['base_config_tokenizer_sha256'],
        'source_url':'https://amsacta.unibo.it/id/eprint/7308/','source_license':'CC BY-NC-SA 4.0',
        'train_images':len(train),'validation_images':len(val),'new_independent_photos':0,
        'script_sha256':source.digest(__file__),'parent_reference_sha256':PARENT_SHA,
        'manifest_sha256':{s:source.digest(args.output/(s+'.jsonl')) for s in ('train','val')},
        'criteria':{'accuracy_beats_16_of_50_and_24pct_majority':True,'macro_f1_above_0_3107512371':True,'all_six_nonzero_recall':True},
        'role':'separate_noncommercial_six_facies_head_on_frozen_qwen_vision',
        'prepared_test_pixels_opened':0,'validation_is_development':True,'encoder_or_chat_weights_changed':False,
        'limitations':['Six source classes, not92 project facies','Three correlated proxy source groups, not final independent test',
                       'Does not train descriptions or chat','Scores not calibrated confidence','No general-assistant activation']}
    source.write_json(args.output/'run_config.json',config)
    shutil.copy2(__file__,args.output/'training_script_snapshot.py')
    shutil.copy2(args.parent/'LICENSE_SOURCES.md',args.output/'LICENSE_SOURCES.md')
    from scripts.train_qwen_source_pilot import configure_offline_environment,_deny_network
    configure_offline_environment(); sys.addaudithook(_deny_network)
    import torch
    from PIL import Image
    from safetensors.torch import save_file,load_file
    from transformers import AutoProcessor,Qwen3VLForConditionalGeneration
    torch.set_num_threads(4); torch.set_num_interop_threads(1); torch.manual_seed(RECIPE['seed'])
    if not torch.cuda.is_available() or torch.cuda.mem_get_info()[0]<5*1024**3: raise RuntimeError('CUDA feature extraction requires at least5GiB free')
    processor=AutoProcessor.from_pretrained(config['base'],local_files_only=True)
    model=Qwen3VLForConditionalGeneration.from_pretrained(config['base'],local_files_only=True,dtype=torch.float16,attn_implementation='sdpa',device_map='cuda')
    model.eval(); model.requires_grad_(False)
    features={}
    for split,rows in (('train',train),('val',val)):
        vectors=[]
        for i,row in enumerate(rows,1):
            with Image.open(row['image']) as opened:
                image=opened.convert('RGB'); image.thumbnail((300,300))
            inputs=source.prompt_inputs(processor,image,source.PROMPT).to('cuda')
            with torch.inference_mode():
                output=model.model.visual(inputs['pixel_values'].to(model.dtype),grid_thw=inputs['image_grid_thw'],return_dict=True)
                vectors.append(pool_features(output).cpu())
            del output,inputs
            if i%25==0 or i==len(rows): print(json.dumps({'event':'features','split':split,'row':i,'total':len(rows)}),flush=True)
        features[split]=torch.stack(vectors)
    del model,processor; gc.collect(); torch.cuda.empty_cache()
    save_file(features,str(args.output/'features.safetensors'))
    labels=torch.tensor([source.CODES.index(r['conversations'][1]['value']) for r in train])
    state=fit_head(features['train'],labels)
    save_file({k:v.contiguous() for k,v in state.items()},str(args.output/'facies_head.safetensors'))
    reloaded=load_file(str(args.output/'facies_head.safetensors'))
    result={'encoder_or_chat_weights_changed':False,'installed_in_application':False,'geologist_approved':False,
            'role':config['role'],'source_license':config['source_license'],'feature_dimension':features['train'].shape[1],
            'head_sha256':source.digest(args.output/'facies_head.safetensors')}
    for split,rows in (('train',train),('val',val)):
        scores=predict_head(features[split],state)
        if not torch.allclose(scores,predict_head(features[split],reloaded)): raise ValueError('Saved head predictions differ')
        predictions=scores.argmax(-1).tolist()
        answers=[{'record_id':row['metadata']['record_id'],'source_group':row['metadata']['source_group'],
            'image_sha256':row['metadata']['image_sha256'],'expected_code':row['conversations'][1]['value'],
            'prediction':source.CODES[predictions[i]],'scores':dict(zip(source.CODES,scores[i].tolist()))} for i,row in enumerate(rows)]
        source.write_json(args.output/(split+'_answers.json'),answers)
        result[split]=source.classification_metrics([r['expected_code'] for r in answers],[r['prediction'] for r in answers])
    m=result['val']; result['research_gates']={'accuracy':m['correct']>16 and m['accuracy']>.24,'macro_f1':m['macro_f1']>.3107512371,'all_classes_recalled':all(x>0 for x in m['per_class_recall'].values())}
    source.write_json(args.output/'result.json',result)
    source.write_json(args.output/'MODEL_CONTRACT.json',{'type':'frozen_qwen_visual_ridge_head_not_peft','base_model_id':'Qwen/Qwen3-VL-2B-Instruct',
        'base_revision':config['base_revision'],'base_config_sha256':source.digest(Path(config['base'])/'config.json'),
        'feature_dimension':result['feature_dimension'],'feature_spec':RECIPE,'code_to_russian':dict(zip(source.CODES,source.LABELS)),
        'training_source_license':config['source_license'],'general_chat_or_description_model':False,'automatic_project_catalog_mapping':None})
    source.write_json(args.output/'acceptance_review.json',{'status':'research_head_requires_review_not_general_assistant',
        'installed_in_application':False,'geologist_approved':False,'source_license':config['source_license'],
        'research_gates':result['research_gates'],'head_sha256':result['head_sha256']})
    print(json.dumps({'event':'complete','train_correct':result['train']['correct'],'val_correct':m['correct'],'macro_f1':m['macro_f1'],'gates':result['research_gates']}),flush=True)


if __name__=='__main__': main()
