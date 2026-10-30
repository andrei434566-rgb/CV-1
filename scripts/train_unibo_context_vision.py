"""Local research: focal/context views and LoRA on last four vision blocks."""
from collections import Counter
from copy import deepcopy
import argparse
import gc
import json
import math
from pathlib import Path
import random
import re
import shutil
import sys
from time import perf_counter

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import train_unibo_visual_facies as source
from scripts.prepare_unibo_context import CONTEXT_PROMPT, CANVAS_SIZE, TARGET_BOX, FOCAL_SIDE, NEUTRAL, blank_context
from scripts.prepare_unibo_vlm_facies import pixel_digest
from scripts.train_unibo_facies_discrimination import discrimination_loss, validate_parent_rows, PARENT_SHA
from scripts.train_lithology_facies_continuation import complete_checkpoint

RECIPE={'epochs':2,'learning_rate':2e-5,'batch_size':1,'gradient_accumulation_steps':4,
        'seed':20261030,'facies_image_side':640,'replay_image_side':300,'vision_last_blocks':4,
        'lora_rank':8,'loss':'six-way first-token CE + 0.1 full-token SFT; raw lithology SFT replay',
        'attention':'sdpa','fp16':True,'save_steps':50,'torch_threads':4,'interop_threads':1}
CRITERIA={'macro_f1_min_gain':.05,'primary_beats_before_and_native_reference':True,
          'all_six_nonzero_recall':True,'raw_invalid_max':1,'lithology_max_lost':2,
          'scope':'noncommercial source-domain development research only, never automatic activation'}


def select_vision_targets(names,last=4):
    found={}
    for name in names:
        match=re.search(r'(?:^|\.)visual\.blocks\.(\d+)\.attn\.(qkv|proj)$',name)
        if match: found.setdefault(int(match.group(1)),{})[match.group(2)]=name
    if len(found)<last: raise ValueError('Not enough supported visual attention blocks')
    chosen=sorted(found)[-last:]
    if any(set(found[i])!={'qkv','proj'} for i in chosen): raise ValueError('Incomplete visual qkv/proj targets')
    return [found[i][part] for i in chosen for part in ('qkv','proj')]


def check_loaded_state(missing,unexpected,vision_targets):
    if unexpected: raise ValueError('Unexpected parent adapter tensors')
    missing_lora=[n for n in missing if 'lora_' in n]
    if len(missing_lora)!=2*len(vision_targets) or any('.visual.' not in n for n in missing_lora):
        raise ValueError('Parent language LoRA not loaded completely, or visual initialization differs')


def validate_context(train,val,old_train,old_val):
    from PIL import Image
    source_by_split={'train':{r['metadata']['record_id']:r for r in old_train},'val':{r['metadata']['record_id']:r for r in old_val}}
    split_sets={s:{f:set() for f in ('source_group','source_sha256','pixel_sha256','focal_record_id')} for s in ('train','val')}
    for split,rows in (('train',train),('val',val)):
        if len(rows)!=len(source_by_split[split]): raise ValueError('Context count differs from original focal data')
        for row in rows:
            meta=row['metadata']; old=source_by_split[split][meta['focal_record_id']]; prior=old['metadata']
            if (meta['split']!=split or meta['source_group']!=prior['source_group'] or meta['source_sha256']!=prior['source_sha256']
                or meta['source_class_code']!=prior['source_class_code'] or meta['focal_image_sha256']!=prior['image_sha256']
                or meta.get('geology_approved') is not False or meta.get('training_approved') is not (split=='train')
                or meta.get('source_license')!='CC BY-NC-SA 4.0' or meta.get('source_mask_passed_to_model') is not False
                or meta.get('class_text_drawn_on_input') is not False or meta.get('quality_scope')!='focal_crop_only_not_context'):
                raise ValueError('Context provenance/approval differs from original focal crop')
            if row['conversations']!=[{'from':'human','value':'<image>\n'+CONTEXT_PROMPT},old['conversations'][1]]:
                raise ValueError('Context prompt or target changed')
            if source.digest(row['image'])!=meta['image_sha256']: raise ValueError('Context file hash differs')
            with Image.open(row['image']) as im: panel=im.convert('RGB')
            with Image.open(old['image']) as im: focal=im.convert('RGB').resize((FOCAL_SIDE,FOCAL_SIDE),Image.Resampling.LANCZOS)
            if (panel.size!=CANVAS_SIZE or pixel_digest(panel)!=meta['pixel_sha256']
                    or panel.crop(TARGET_BOX).tobytes()!=focal.tobytes()):
                raise ValueError('Context focal panel or pixel hash differs')
            for field in split_sets[split]: split_sets[split][field].add(meta[field])
        if len(split_sets[split]['focal_record_id'])!=len(rows) or len(split_sets[split]['pixel_sha256'])!=len(rows):
            raise ValueError('Duplicate context example')
    for field in split_sets['train']:
        if split_sets['train'][field]&split_sets['val'][field]: raise ValueError('Context split leakage: '+field)


def prepare(parent,data,runtime,output):
    cfg=json.loads((parent/'run_config.json').read_text(encoding='utf-8'))
    old_rows,native,guard=validate_parent_rows(cfg,parent,runtime)
    old_train=[r for r in old_rows if r.get('research_task')=='unibo_six_choice_facies']
    train,val=source.read_rows(data/'train.jsonl'),source.read_rows(data/'val.jsonl')
    summary=json.loads((data/'summary.json').read_text(encoding='utf-8'))
    if summary.get('status')!='complete' or summary.get('prepared_test_pixels_opened')!=0: raise ValueError('Context preparation incomplete')
    validate_context(train,val,old_train,native)
    replay=[r for r in old_rows if r.get('research_task')!='unibo_six_choice_facies']
    records=train+replay
    random.Random(RECIPE['seed']).shuffle(records)
    ids={r['record_id'] for r in json.loads((parent/'post_hoc_training_fit.json').read_text(encoding='utf-8'))['answers']}
    fit=[r for r in train if r['metadata']['focal_record_id'] in ids]
    if len(fit)!=24 or Counter(r['conversations'][1]['value'] for r in fit)!=Counter(dict.fromkeys(source.CODES,4)):
        raise ValueError('Train-fit probe identities differ')
    for name,rows in (('train',records),('validation_facies',val),('validation_lithology',guard),('validation_native',native),('train_fit_probe',fit)):
        source.write_rows(output/(name+'.jsonl'),rows)
    paths=[Path(source.__file__),Path(__file__),ROOT/'scripts/prepare_unibo_context.py',ROOT/'scripts/prepare_unibo_vlm_facies.py',
           ROOT/'scripts/train_unibo_facies_discrimination.py',ROOT/'scripts/train_qwen_source_pilot.py',
           ROOT/'scripts/train_lithology_facies_continuation.py',ROOT/'app/infrastructure/preference_loss.py',
           ROOT/'app/infrastructure/open_dataset_import.py',runtime/'train_qwen3vl.py']
    external=[parent/'run_config.json',parent/'result.json',parent/'adapter/adapter_config.json',parent/'adapter/adapter_model.safetensors',
              parent/'post_hoc_training_fit.json',data/'train.jsonl',data/'val.jsonl',data/'summary.json']
    config={'recipe':RECIPE,'criteria':CRITERIA,'base':cfg['base'],'base_revision':cfg['base_revision'],
            'runtime':str(runtime),'parent':str(parent),'data':str(data),'parent_sha256':PARENT_SHA,
            'package_versions':source.package_versions(),'python_version':sys.version,
            'dependencies':{str(p):source.digest(p) for p in paths+external},
            'base_config_tokenizer_sha256':cfg['base_config_tokenizer_sha256'],'base_weights_identity':cfg['base_weights_identity'],
            'manifest_sha256':{p.name:source.digest(p) for p in output.glob('*.jsonl')},
            'class_weights':source.facies_weights(train),'prompt':CONTEXT_PROMPT,
            'source_url':'https://amsacta.unibo.it/id/eprint/7308/','source_license':'CC BY-NC-SA 4.0',
            'role':'isolated_noncommercial_context_and_visual_lora_research','unique_training_focal_images':480,
            'facies_focal_images':352,'replay_images':128,'validation_focal_images':50,'native_guard_images':64,
            'new_independent_photos':0,'prepared_test_pixels_opened':0,
            'limitations':['Source six classes only; no project92 mapping','S-group development validation, not independent final test',
                           'Input context, prompt, LR and trainable modules change together; no causal attribution to one factor',
                           'Context can contain other classes; label applies only to focal panel','No automatic activation or general description approval']}
    source.write_json(output/'run_config.json',config)
    shutil.copy2(__file__,output/'training_script_snapshot.py')
    shutil.copy2(parent/'LICENSE_SOURCES.md',output/'LICENSE_SOURCES.md')
    source.write_json(output/'acceptance_review.json',{'schema':'qwen_acceptance_review_v1',
        'status':'rejected_for_general_assistant_task_scope','installed_in_application':False,'geologist_approved':False,
        'research_evaluation_status':'prepared','source_license':config['source_license'],'permitted_role':config['role']})
    return config,records,val,guard,native,fit


def restore(output,parent,data,runtime):
    cfg=json.loads((output/'run_config.json').read_text(encoding='utf-8'))
    if cfg['recipe']!=RECIPE or cfg['criteria']!=CRITERIA: raise ValueError('Resume recipe changed')
    if (cfg['package_versions']!=source.package_versions() or cfg['python_version']!=sys.version
        or any(Path(cfg[k]).resolve()!=p for k,p in (('parent',parent),('data',data),('runtime',runtime)))): raise ValueError('Resume environment/path differs')
    for mapping in (cfg['dependencies'],cfg['base_config_tokenizer_sha256']):
        for path,sha in mapping.items():
            if source.digest(path)!=sha: raise ValueError('Resume dependency or data changed: '+path)
    for name,sha in cfg['manifest_sha256'].items():
        if source.digest(output/name)!=sha: raise ValueError('Resume manifest changed')
    if source.digest(output/'training_script_snapshot.py')!=source.digest(__file__): raise ValueError('Resume snapshot changed')
    if (Path(cfg['base'])/'model.safetensors').stat().st_size!=cfg['base_weights_identity']['bytes']: raise ValueError('Base weight size differs')
    old=validate_parent_rows(json.loads((parent/'run_config.json').read_text(encoding='utf-8')),parent,runtime)
    validate_context(source.read_rows(data/'train.jsonl'),source.read_rows(data/'val.jsonl'),[r for r in old[0] if r.get('research_task')=='unibo_six_choice_facies'],old[1])
    return (cfg,*(source.read_rows(output/(name+'.jsonl')) for name in ('train','validation_facies','validation_lithology','validation_native','train_fit_probe')))


def infer(model,processor,image,question,ids,generate=True):
    import torch
    inputs=source.prompt_inputs(processor,image,question).to('cuda')
    with torch.inference_mode():
        out=model(**inputs,logits_to_keep=1,use_cache=False)
        logits=out.logits[0,-1,[ids[c] for c in source.CODES]].float()
        probs=logits.softmax(-1).cpu().tolist(); forced=source.CODES[int(logits.argmax())]
        del out,logits
        if generate:
            generated=model.generate(**inputs,max_new_tokens=8,do_sample=False,use_cache=True)
            answer=processor.batch_decode(generated[:,inputs['input_ids'].shape[1]:],skip_special_tokens=True)[0]
        else: answer=None
    return {'forced_code':forced,'probabilities_A_to_F':dict(zip(source.CODES,probs)),
            'raw_answer':answer,'raw_code':source.raw_code(answer) if answer is not None else None}


def evaluate(model,processor,val,guard,native,fit,ids,output,stage,log):
    from PIL import Image
    model.eval(); model.config.use_cache=True
    answers=[]; native_answers=[]; lithology=[]; fit_answers=[]
    gray=infer(model,processor,Image.new('RGB',CANVAS_SIZE,NEUTRAL),CONTEXT_PROMPT,ids,False)
    for i,row in enumerate(val,1):
        with Image.open(row['image']) as im: image=im.convert('RGB')
        a={'record_id':row['metadata']['record_id'],'source_group':row['metadata']['source_group'],
           'focal_record_id':row['metadata']['focal_record_id'],'image_sha256':row['metadata']['image_sha256'],
           'expected_code':row['conversations'][1]['value'],**infer(model,processor,image,CONTEXT_PROMPT,ids)}
        a['context_blank_code']=infer(model,processor,blank_context(image),CONTEXT_PROMPT,ids,False)['forced_code']
        answers.append(a); source.write_json(output/(stage+'_facies_answers.json'),answers)
        log('evaluation',stage=stage,task='context',row=i,total=len(val))
    for i,row in enumerate(native,1):
        with Image.open(row['image']) as im:
            image=im.convert('RGB'); image.thumbnail((300,300))
        native_answers.append({'record_id':row['metadata']['record_id'],'expected_code':row['conversations'][1]['value'],
                               **infer(model,processor,image,source.PROMPT,ids,False)})
    source.write_json(output/(stage+'_native_answers.json'),native_answers)
    for i,row in enumerate(guard,1):
        answer=source.predict_lithology(model,processor,row)
        lithology.append({'record_id':row['metadata']['record_id'],'image_sha256':row['metadata']['image_sha256'],
            'expected':row['conversations'][1]['value'],'raw_answer':answer,'prediction':answer.casefold().strip().strip(' .,:;!"«»')})
        source.write_json(output/(stage+'_lithology_answers.json'),lithology)
        log('evaluation',stage=stage,task='lithology',row=i,total=len(guard))
    for row in fit:
        with Image.open(row['image']) as im: image=im.convert('RGB')
        fit_answers.append({'record_id':row['metadata']['record_id'],'expected_code':row['conversations'][1]['value'],
                            **infer(model,processor,image,CONTEXT_PROMPT,ids,False)})
    source.write_json(output/(stage+'_train_fit.json'),fit_answers)
    targets=[r['expected_code'] for r in answers]
    metrics={'facies':{'forced_choice':source.classification_metrics(targets,[r['forced_code'] for r in answers]),
                       'raw_generation':source.classification_metrics(targets,[r['raw_code'] for r in answers]),
                       'context_blank':source.classification_metrics(targets,[r['context_blank_code'] for r in answers]),
                       'gray_forced_choice':source.classification_metrics(targets,[gray['forced_code']]*len(targets))},
             'native_crop':source.classification_metrics([r['expected_code'] for r in native_answers],[r['forced_code'] for r in native_answers]),
             'raw_lithology_guard':source.classification_metrics([r['expected'] for r in lithology],[r['prediction'] for r in lithology],source.LITHOLOGY),
             'train_fit':source.classification_metrics([r['expected_code'] for r in fit_answers],[r['forced_code'] for r in fit_answers])}
    source.write_json(output/(stage+'_gray.json'),gray)
    source.write_json(output/(stage+'_metrics.json'),metrics)
    return metrics


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime',type=Path,default=ROOT.parent/'vlm_core_v1')
    parser.add_argument('--parent',type=Path,default=ROOT.parent/'vlm_core_v1/outputs/core_vlm_visual_facies_20261030_lora')
    parser.add_argument('--data',type=Path,default=ROOT.parent/'vlm_core_v1/data/unibo_context_vision_20261030_v2')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--prepare-only',action='store_true'); parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    args.runtime,args.parent,args.data,args.output=(p.resolve() for p in (args.runtime,args.parent,args.data,args.output))
    if args.resume:
        if (args.output/'result.json').exists(): raise ValueError('Completed run cannot resume')
        cfg,rows,val,guard,native,fit=restore(args.output,args.parent,args.data,args.runtime)
        checkpoint=complete_checkpoint(args.output/'trainer')
    else:
        if args.output.exists(): raise ValueError('Choose a new run directory')
        args.output.mkdir(parents=True)
        cfg,rows,val,guard,native,fit=prepare(args.parent,args.data,args.runtime,args.output); checkpoint=None
    if args.prepare_only:
        print(json.dumps({'status':'prepared','train':len(rows),'val':len(val)}),flush=True); return
    from scripts.train_qwen_source_pilot import configure_offline_environment,_deny_network,validate_training_tokens
    configure_offline_environment(); sys.addaudithook(_deny_network)
    import torch
    from PIL import Image
    from peft import LoraConfig,get_peft_model,set_peft_model_state_dict
    from safetensors.torch import load_file
    from transformers import AutoProcessor,Qwen3VLForConditionalGeneration,Trainer,TrainingArguments,TrainerCallback,set_seed
    sys.path.insert(0,str(args.runtime))
    from train_qwen3vl import CorePhotoDataset,SingleImageCollator
    if not torch.cuda.is_available() or torch.cuda.mem_get_info()[0]<6.5*1024**3: raise RuntimeError('Requires at least6.5GiB free CUDA memory')
    torch.set_num_threads(4); torch.set_num_interop_threads(1); set_seed(RECIPE['seed'])
    processor=AutoProcessor.from_pretrained(cfg['base'],local_files_only=True)
    class Dataset(torch.utils.data.Dataset):
        def __init__(self,records):
            self.records=records; self.photos=CorePhotoDataset(records,processor,640); self.replay=CorePhotoDataset(records,processor,300)
        def __len__(self): return len(self.records)
        def __getitem__(self,i):
            row=self.records[i]; facies=row.get('research_task')=='unibo_six_choice_facies'; code=row['conversations'][1]['value']
            result=(self.photos if facies else self.replay)[i]
            result['task_index']=torch.tensor(source.CODES.index(code) if facies else -1)
            result['example_weight']=torch.tensor(cfg['class_weights'].get(code,1.) if facies else 1.)
            return result
    token_ids={c:processor.tokenizer.encode(c,add_special_tokens=False) for c in source.CODES}
    if any(len(x)!=1 for x in token_ids.values()) or len({x[0] for x in token_ids.values()})!=6: raise ValueError('Distinct one-token A-F codes required')
    ids={c:x[0] for c,x in token_ids.items()}; dataset=Dataset(rows)
    audit={'rows':len(rows)+len(val),'max_sequence_tokens':0,'truncated_rows':0,'causal_first_token_checked':0}
    for row in rows+val:
        features=Dataset([row])[0]
        with Image.open(row['image']) as im:
            image=im.convert('RGB'); image.thumbnail((640,640) if row.get('research_task')=='unibo_six_choice_facies' else (300,300))
        prefix=source.prompt_inputs(processor,image,row['conversations'][0]['value'])['input_ids'][0].tolist()
        counts=validate_training_tokens(features['input_ids'].tolist(),prefix,features['labels'].tolist(),max_sequence_tokens=1024,identity=row['metadata']['record_id'])
        audit['max_sequence_tokens']=max(audit['max_sequence_tokens'],counts['sequence_tokens'])
        if row.get('research_task')=='unibo_six_choice_facies':
            if features['labels'][len(prefix)].item()!=ids[row['conversations'][1]['value']]: raise ValueError('Wrong first class token')
            audit['causal_first_token_checked']+=1
    source.write_json(args.output/'token_preflight.json',audit)
    model=Qwen3VLForConditionalGeneration.from_pretrained(cfg['base'],local_files_only=True,dtype=torch.float16,attn_implementation='sdpa',device_map='cuda')
    targets=select_vision_targets([name for name,_ in model.named_modules()],4)
    peft_config=LoraConfig.from_pretrained(args.parent/'adapter'); peft_config.inference_mode=False
    peft_config.init_lora_weights=True; peft_config.target_modules=sorted(set(peft_config.target_modules)|set(targets))
    model=get_peft_model(model,peft_config)
    loaded=set_peft_model_state_dict(model,load_file(str(args.parent/'adapter/adapter_model.safetensors')))
    check_loaded_state(loaded.missing_keys,loaded.unexpected_keys,targets)
    visual_params=[(n,p) for n,p in model.named_parameters() if '.visual.' in n and p.requires_grad]
    if len(visual_params)!=len(targets)*2 or any('lora_' not in n for n,p in visual_params): raise ValueError('Unexpected visual trainable parameters')
    if any(torch.count_nonzero(p).item() for n,p in visual_params if 'lora_B' in n): raise ValueError('Visual LoRA must start as zero update')
    source.write_json(args.output/'adapter_initialization.json',{'parent_sha256':PARENT_SHA,'language_missing_lora':0,'new_visual_targets':targets,
        'visual_lora_tensors':len(visual_params),'visual_B_initially_zero':True,'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad)})
    def log(event,**values):
        item={'event':event,**values}
        with (args.output/'progress.jsonl').open('a',encoding='utf-8') as stream: stream.write(json.dumps(item)+'\n')
        print(json.dumps(item),flush=True)
    before=evaluate(model,processor,val,guard,native,fit,ids,args.output,'before',log)
    model.train(); model.config.use_cache=False
    class LossTrainer(Trainer):
        def compute_loss(self,model,inputs,return_outputs=False,num_items_in_batch=None):
            inputs=dict(inputs); task=inputs.pop('task_index'); weight=inputs.pop('example_weight')
            loss,out,_=discrimination_loss(model,inputs,tuple(ids[c] for c in source.CODES),int(task.item()))
            loss=loss*weight.reshape(()).to(loss.device)
            return (loss,out) if return_outputs else loss
    class Callback(TrainerCallback):
        def on_log(self,arguments,state,control,logs=None,**kwargs): log('loss',step=state.global_step,**(logs or {}))
        def on_pre_optimizer_step(self,arguments,state,control,**kwargs):
            path=args.output/'vision_gradient_audit.json'
            if path.exists(): return
            active=[n for n,p in visual_params if p.grad is not None and 'lora_B' in n and torch.count_nonzero(p.grad).item()>0]
            if any(p.grad is not None and not torch.isfinite(p.grad).all().item() for n,p in visual_params):
                log('vision_gradient_scaler_retry',step=state.global_step)
                return
            if not active: raise RuntimeError('Visual LoRA gradients missing')
            source.write_json(path,{'step':state.global_step,'nonzero_visual_B_gradients':active,'finite':True})
    training_args=TrainingArguments(output_dir=str(args.output/'trainer'),num_train_epochs=2,per_device_train_batch_size=1,
        gradient_accumulation_steps=4,learning_rate=2e-5,fp16=True,gradient_checkpointing=True,gradient_checkpointing_kwargs={'use_reentrant':False},
        optim='adamw_torch',logging_steps=10,save_steps=50,save_total_limit=2,save_strategy='steps',remove_unused_columns=False,
        dataloader_num_workers=0,report_to='none',seed=RECIPE['seed'],data_seed=RECIPE['seed'],warmup_steps=8,disable_tqdm=True)
    trainer=LossTrainer(model=model,args=training_args,train_dataset=dataset,data_collator=SingleImageCollator(processor.tokenizer.pad_token_id),processing_class=processor,callbacks=[Callback()])
    trainer.model_accepts_loss_kwargs=False; started=perf_counter()
    log('training_started',epochs=2,images=len(rows),vision_targets=len(targets),resume=bool(checkpoint))
    training=trainer.train(resume_from_checkpoint=str(checkpoint) if checkpoint else None)
    if not (args.output/'vision_gradient_audit.json').exists(): raise RuntimeError('No finite nonzero visual gradient verified')
    adapter=args.output/'adapter'; trainer.save_model(str(adapter)); processor.save_pretrained(adapter)
    model.gradient_checkpointing_disable(); model.zero_grad(set_to_none=True)
    if trainer.optimizer: trainer.optimizer.state.clear(); trainer.optimizer=None
    trainer.lr_scheduler=None; gc.collect(); torch.cuda.empty_cache()
    after=evaluate(model,processor,val,guard,native,fit,ids,args.output,'candidate',log)
    old,new=before['facies']['forced_choice'],after['facies']['forced_choice']
    gates={'accuracy_gain':new['accuracy']>max(old['accuracy'],before['native_crop']['accuracy']),
        'macro_f1_gain':new['macro_f1']>=old['macro_f1']+.05,'all_six_recalled':all(v>0 for v in new['per_class_recall'].values()),
        'beats_gray':new['accuracy']>max(.24,after['facies']['gray_forced_choice']['accuracy']),
        'format_preserved':after['facies']['raw_generation']['invalid_outputs']<=1,
        'lithology_preserved':after['raw_lithology_guard']['correct']>=before['raw_lithology_guard']['correct']-2}
    result={'before':before,'candidate':after,'gates':gates,'training_metrics':training.metrics,
        'seconds_training_and_after_eval':perf_counter()-started,'peak_cuda_memory_bytes':torch.cuda.max_memory_allocated(),
        'adapter_sha256':source.digest(adapter/'adapter_model.safetensors'),'parent_sha256':PARENT_SHA,
        'role':cfg['role'],'source_license':cfg['source_license'],'installed_in_application':False,'geologist_approved':False}
    source.write_json(args.output/'result.json',result)
    review=json.loads((args.output/'acceptance_review.json').read_text(encoding='utf-8'))
    review.update({'research_evaluation_status':'complete','research_gates':gates,'adapter_sha256':result['adapter_sha256']})
    for dest in (args.output/'acceptance_review.json',adapter/'acceptance_review.json'): source.write_json(dest,review)
    shutil.copy2(args.output/'LICENSE_SOURCES.md',adapter/'LICENSE_SOURCES.md')
    log('complete',facies_correct=new['correct'],macro_f1=new['macro_f1'],native_correct=after['native_crop']['correct'],
        lithology_correct=after['raw_lithology_guard']['correct'],all_gates=all(gates.values()))


if __name__=='__main__': main()
