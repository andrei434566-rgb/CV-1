"""Train real visual LoRA and a residual six-facies head, language untouched.

One predeclared recipe, three grouped OOF runs and one final train run.
No checkpoint selection using validation; unresolved mixed data stay local.
"""
import argparse
import gc
import json
from pathlib import Path
import random
import shutil
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import train_unibo_visual_facies as source
from scripts.train_core_facies_conservative import make_state
from scripts.kernel_facies_head import grouped_weights,all_class_gates
from scripts.partial_facies_ridge import mixed_metrics
from scripts.vision_facies_adapter import load_original_vision,add_lora,batch_inputs,post_features,normalized_post,label_loss,adapter_state,restore_adapter

RECIPE=dict(epochs=3,batch_size=4,rank=8,lora_alpha=16,last_vision_blocks=4,lora_lr=1e-4,head_lr=5e-4,weight_decay=.01,
 logit_scale=5.,old_correct_exact_KL_weight=.3,head_delta_penalty=.1,seed=20261031,gradient_clip=1.,
 supervision='Weighted exact CE, -log(PA+PB) for generic floodplain. Teacher KL only on correctly classified OLD EXACT targets, not on partial targets.',
 weights='Class+source-group balanced on each train fold; external0.5, core end-face0.25',
 normalization='Old train subset only, Qwen post mean/std -> L2, fixed within run',
 selection='One fixed recipe; final epoch only. Three group OOF folds then frozen recipe before repeated val50.',
 image='Original RGB thumbnail300, unchanged original processor; no synthetic images or geometric/colour augmentation',
 language_weights_changed=False,vision_encoder_changed=True,distribution_approved=False,installed_in_application=False)


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--previous',type=Path,default=ROOT.parent/'vlm_core_v1/outputs/core_qwen_core_contrasts_20261031_v7');p.add_argument('--output',type=Path,required=True)
 p.add_argument('--stage',choices=('full','fold0','fold1','fold2','finalize'),default='full',help='Use one fresh process per fold, followed by finalize, to release native model memory.')
 a=p.parse_args();prev,out=a.previous.resolve(),a.output.resolve()
 if not out.is_relative_to(ROOT.parent/'vlm_core_v1/outputs'):raise ValueError('Choose a local experiment output')
 if out.exists() and a.stage in ('full','fold0'):raise ValueError('Choose a new local experiment output')
 if a.stage in ('fold1','fold2','finalize') and not (out/'run_config.json').is_file():raise ValueError('Initialize this experiment with fold0 first')
 preflight=ROOT.parent/'vlm_core_v1/outputs/vision_facies_adapter_v9_preflight.json'
 check=json.loads(preflight.read_text(encoding='utf-8'))
 if check['status']!='passed' or max(check['original_feature_max_deltas'])>1e-5:raise ValueError('Vision preflight not passed')
 cfgold=json.loads((prev/'run_config.json').read_text(encoding='utf-8'));ref=Path(cfgold['reference']);base=Path(cfgold['base']);contract=json.loads((prev/'MODEL_CONTRACT.json').read_text(encoding='utf-8'))
 rows=source.read_rows(prev/'train.jsonl');val=source.read_rows(prev/'val.jsonl')
 if len(rows)!=804 or len(val)!=50 or val!=source.read_rows(ref/'val.jsonl'):raise ValueError('Data changed')
 for f,h in cfgold['dependencies'].items():
  if source.digest(f)!=h:raise ValueError('Source dependency changed: '+f)
 if source.package_versions()!=contract['package_versions']:raise ValueError('Runtime differs')
 from scripts.classify_with_qwen_facies_head import streamed_sha
 if streamed_sha(base/'model.safetensors')!=contract['base_weights_sha256']:raise ValueError('Original base changed')
 for n,h in contract['preprocessing_files_sha256'].items():
  if source.digest(base/n)!=h:raise ValueError('Processor changed')
 if not out.exists():
  out.mkdir(parents=True);source.write_rows(out/'train.jsonl',rows);source.write_rows(out/'val.jsonl',val)
 deps=[Path(__file__),ROOT/'scripts/vision_facies_adapter.py',ROOT/'scripts/train_core_facies_conservative.py',ROOT/'scripts/kernel_facies_head.py',ROOT/'scripts/partial_facies_ridge.py',preflight,
  prev/'features.safetensors',prev/'train.jsonl',prev/'val.jsonl',prev/'selection_frozen.json',prev/'audit_result.json',ref/'oof_answers.json',ref/'facies_head.safetensors']
 cfg=dict(recipe=RECIPE,previous=str(prev),reference=str(ref),base=str(base),train_count=len(rows),validation_count=len(val),
  dependencies={str(f):source.digest(f) for f in deps},source_rights=cfgold['source_rights'],distribution_approved=False,installed_in_application=False,prepared_test_pixels_opened=0)
 if (out/'run_config.json').exists():
  if json.loads((out/'run_config.json').read_text(encoding='utf-8'))!=cfg:raise ValueError('Staged run recipe, source or dependency hash changed')
  if source.read_rows(out/'train.jsonl')!=rows or source.read_rows(out/'val.jsonl')!=val:raise ValueError('Staged rows changed')
 else:
  source.write_json(out/'run_config.json',cfg);shutil.copy2(__file__,out/'training_script_snapshot.py')
 from scripts.train_qwen_source_pilot import configure_offline_environment,_deny_network
 configure_offline_environment();sys.addaudithook(_deny_network)
 import torch,time
 from safetensors.torch import load_file,save_file
 from PIL import Image
 from transformers import AutoProcessor
 torch.set_num_threads(4);torch.set_num_interop_threads(1)
 if not torch.cuda.is_available() or torch.cuda.mem_get_info()[0]<4*1024**3:raise RuntimeError('Need4GiB free CUDA')
 processor=AutoProcessor.from_pretrained(base,local_files_only=True);raw=load_file(str(prev/'features.safetensors'))['train'];mapping=json.loads((prev/'selection_frozen.json').read_text(encoding='utf-8'))['fold_source_groups'];dims=(6144,2048,114)
 images=[]
 for r in rows:
  if source.digest(r['image'])!=r['image_sha256']:raise ValueError('Image changed')
  with Image.open(r['image']) as im:images.append(im.convert('RGB'))
 def log(event,**kw):
  data=dict(event=event,**kw)
  with (out/'progress.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(data)+'\n')
  print(json.dumps(data),flush=True)
 training_receipts=[]
 def fit_run(tag,indices,held,seed):
  torch.manual_seed(seed);random.seed(seed);trainrows=[rows[i] for i in indices]
  old_ids=[i for i in indices if i<728];old_rows=[rows[i] for i in old_ids]
  initial,_=make_state(raw[old_ids],old_rows,dims)
  norm={k:initial[k].float().cuda() for k in ('mean','scale')};coeff=initial['coefficients'].float().cuda()
  # Cache teacher scores from original vision; do not read held labels to train.
  with torch.no_grad():teacher=normalized_post(raw[indices,2048:].cuda(),norm)@coeff
  teacher_codes=teacher.argmax(1).tolist();protect=torch.tensor([i<728 and len(r['allowed_codes'])==1 and 'ABCDEF'[c]==r['allowed_codes'][0] for i,r,c in zip(indices,trainrows,teacher_codes)],device='cuda')
  teacher_probs=torch.softmax(teacher*RECIPE['logit_scale'],1).detach()
  weights=grouped_weights(trainrows,'class_group',.5).float().cuda();vision=load_original_vision(base);modules=add_lora(vision,RECIPE['rank'],RECIPE['lora_alpha'],RECIPE['last_vision_blocks'])
  delta=torch.nn.Parameter(torch.zeros_like(coeff));vparams=[p for p in vision.parameters() if p.requires_grad]
  optimizer=torch.optim.AdamW([dict(params=vparams,lr=RECIPE['lora_lr']),dict(params=[delta],lr=RECIPE['head_lr'])],weight_decay=RECIPE['weight_decay'])
  scaler=torch.amp.GradScaler('cuda',init_scale=1024.);start=time.perf_counter();epoch_stats=[];first_grad=None
  for epoch in range(RECIPE['epochs']):
   order=list(range(len(indices)));random.Random(seed+epoch).shuffle(order);total=0.;seen=0
   for offset in range(0,len(order),RECIPE['batch_size']):
    local=order[offset:offset+RECIPE['batch_size']];actual=[indices[i] for i in local];inp=batch_inputs(processor,[images[i] for i in actual],'cuda');optimizer.zero_grad(set_to_none=True)
    with torch.autocast('cuda',dtype=torch.float16):features=post_features(vision,inp)
    scores=normalized_post(features,norm)@(coeff+delta);logits=scores*RECIPE['logit_scale'];labels=[rows[i]['allowed_codes'] for i in actual]
    loss=label_loss(logits,labels,weights[local]);keep=protect[local]
    if keep.any():loss=loss+RECIPE['old_correct_exact_KL_weight']*torch.nn.functional.kl_div(torch.log_softmax(logits[keep].float(),1),teacher_probs[local][keep],reduction='batchmean')
    loss=loss+RECIPE['head_delta_penalty']*delta.square().sum()
    if not torch.isfinite(loss):raise ValueError('Nonfinite visual training loss')
    scaler.scale(loss).backward();scaler.unscale_(optimizer)
    if first_grad is None:
     first_grad={n:float(p.grad.norm()) for n,p in vision.named_parameters() if p.requires_grad and n.endswith('lora_b') and p.grad is not None}
     if not first_grad or max(first_grad.values())<=0 or any(not torch.isfinite(p.grad).all() for p in vparams+[delta] if p.grad is not None):raise ValueError('Missing/nonfinite first gradients')
    torch.nn.utils.clip_grad_norm_(vparams+[delta],RECIPE['gradient_clip']);scaler.step(optimizer);scaler.update();total+=float(loss.detach())*len(local);seen+=len(local)
   stat=dict(epoch=epoch+1,mean_loss=total/seen,examples=seen);epoch_stats.append(stat);log('epoch',tag=tag,**stat,elapsed_seconds=time.perf_counter()-start)
  state=adapter_state(vision,delta,norm,coeff);folder=out/tag;folder.mkdir();save_file(state,str(folder/'adapter.safetensors'))
  if max(float(p.abs().max()) for n,p in state.items() if n.endswith('lora_b'))==0:raise ValueError('Visual adapter remained zero')
  # Reload saved adapter before any evaluation; same exact tensors required.
  reloaded=load_file(str(folder/'adapter.safetensors'));restore_adapter(vision,reloaded)
  if any(not torch.equal(state[k],reloaded[k]) for k in state):raise ValueError('Saved adapter changed')
  receipt=dict(tag=tag,seed=seed,training_ids=[rows[i]['record_id'] for i in indices],held_ids=[rows[i]['record_id'] for i in held],
   normalizer_training_ids=[rows[i]['record_id'] for i in old_ids],teacher_protected_exact_rows=int(protect.sum()),epochs=epoch_stats,
   modules=modules,trainable_vision_parameters=sum(p.numel() for p in vparams),trainable_head_parameters=delta.numel(),
   first_gradient_norms=first_grad,adapter_sha256=source.digest(folder/'adapter.safetensors'),seconds=time.perf_counter()-start)
  source.write_json(folder/'receipt.json',receipt);training_receipts.append(receipt)
  return vision,coeff+delta.detach(),norm,state
 def evaluate(vision,coeff,norm,indices,image_list):
  predictions=[];scores=[]
  with torch.inference_mode():
   for offset in range(0,len(indices),4):
    batch=indices[offset:offset+4];inp=batch_inputs(processor,[image_list[i] for i in batch],'cuda');features=post_features(vision,inp)
    s=normalized_post(features,norm)@coeff;predictions.extend('ABCDEF'[i] for i in s.argmax(1).tolist());scores.extend(s.cpu().tolist())
  return predictions,scores
 oof=['']*len(rows)
 for fold in range(3):
  if a.stage=='finalize':
   folder=out/('fold'+str(fold));receipt=json.loads((folder/'receipt.json').read_text(encoding='utf-8'))
   if source.digest(folder/'adapter.safetensors')!=receipt['adapter_sha256']:raise ValueError('Saved fold changed')
   saved=json.loads((folder/'held_answers.json').read_text(encoding='utf-8'));held=[i for i,r in enumerate(rows) if mapping[r['source_group']]==fold]
   if receipt['held_ids']!=[rows[i]['record_id'] for i in held] or [s['record_id'] for s in saved]!=receipt['held_ids']:raise ValueError('Saved fold records differ')
   for i,s in zip(held,saved):oof[i]=s['prediction']
   training_receipts.append(receipt)
   continue
  if a.stage!='full' and a.stage!='fold'+str(fold):continue
  indices=[i for i,r in enumerate(rows) if mapping[r['source_group']]!=fold];held=[i for i,r in enumerate(rows) if mapping[r['source_group']]==fold]
  if {rows[i]['source_group'] for i in indices}&{rows[i]['source_group'] for i in held}:raise ValueError('Group leakage')
  vision,coeff,norm,_=fit_run('fold'+str(fold),indices,held,RECIPE['seed']+fold)
  pred,_=evaluate(vision,coeff,norm,held,images)
  for i,c in zip(held,pred):oof[i]=c
  source.write_json(out/('fold'+str(fold))/'held_answers.json',[dict(record_id=rows[i]['record_id'],prediction=c,allowed_codes=rows[i]['allowed_codes']) for i,c in zip(held,pred)])
  del vision,coeff,norm;gc.collect();torch.cuda.empty_cache();log('fold_complete',fold=fold)
 if a.stage in ('fold0','fold1','fold2'):return
 baseline=mixed_metrics(rows[:728],[r['prediction'] for r in json.loads((ref/'oof_answers.json').read_text(encoding='utf-8'))],source)['exact_six_class'];oldm=mixed_metrics(rows[:728],oof[:728],source)['exact_six_class'];train_gates=all_class_gates(oldm,baseline)
 frozen=dict(stage='before_validation_pixels_and_predictions',recipe=RECIPE,baseline_old_oof=baseline,old_oof=oldm,all_class_gates=train_gates,fold_source_groups=mapping,training_record_ids=[r['record_id'] for r in rows])
 source.write_json(out/'selection_frozen.json',frozen);source.write_json(out/'oof_answers.json',[dict(record_id=r['record_id'],source_group=r['source_group'],allowed_codes=r['allowed_codes'],prediction=c,fold=mapping[r['source_group']]) for r,c in zip(rows,oof)])
 vision,coeff,norm,state=fit_run('final',list(range(len(rows))),[],RECIPE['seed']+3)
 train_pred,_=evaluate(vision,coeff,norm,list(range(len(rows))),images)
 # First open validation image pixels only now, after OOF/recipe frozen and final fitting.
 valimages=[]
 for r in val:
  with Image.open(r['image']) as im:valimages.append(im.convert('RGB'))
 vp,vs=evaluate(vision,coeff,norm,list(range(len(val))),valimages);expected=[r['conversations'][1]['value'] for r in val];vm=source.classification_metrics(expected,vp)
 rcfg=json.loads((ref/'run_config.json').read_text(encoding='utf-8'));baseline_val=load_file(str(Path(rcfg['reference'])/'validation_features.safetensors'))['raw_val'][:,:6144]
 from scripts.facies_head_v2_features import predict
 bp=['ABCDEF'[i] for i in predict(baseline_val,load_file(str(ref/'facies_head.safetensors'))).argmax(1).tolist()];bm=source.classification_metrics(expected,bp)
 gates=all_class_gates(vm,bm);gates.update(train_eligible=all(train_gates.values()),accuracy=vm['correct']>=bm['correct'])
 result=dict(role='visual_LoRA_plus_residual_facies_head_not_general_chat',vision_encoder_changed=True,language_weights_changed=False,head_sha256=source.digest(out/'final/adapter.safetensors'),selection_frozen_sha256=source.digest(out/'selection_frozen.json'),
  old_oof=oldm,baseline_old_oof=baseline,oof=mixed_metrics(rows,oof,source),train=mixed_metrics(rows,train_pred,source),val=vm,baseline_val=bm,
  training_gates=train_gates,validation_gates=gates,distribution_approved=False,installed_in_application=False,training_receipts=training_receipts,
  status='research_all_class_gates_passed_requires_independent_validation' if all(gates.values()) else 'not_promoted_all_class_gate_failed')
 source.write_json(out/'result.json',result);source.write_json(out/'val_answers.json',[dict(record_id=r['metadata']['record_id'],expected_code=e,prediction=c,baseline_prediction=b,scores=s) for r,e,c,b,s in zip(val,expected,vp,bp,vs)])
 contract.update(type='qwen3vl_vision_lora_residual_facies_classifier',feature_contract_version='qwen3vl_vision_lora_facies_v1',adapter_sha256=result['head_sha256'],head_sha256=result['head_sha256'],training_recipe=RECIPE,distribution_approved=False,acceptance_status=result['status'],
  vision_encoder_changed=True,general_chat_or_description_model=False,feature_spec='Original vision + LoRA last4 qkv/proj, post token mean/std4096 -> fixed old-only normalizer -> base ridge + trained head delta',training_source_license=cfg['source_rights'])
 source.write_json(out/'MODEL_CONTRACT.json',contract);log('complete',val=vm,gates=gates,status=result['status'])


if __name__=='__main__':main()
