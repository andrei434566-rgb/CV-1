"""Fresh-load replay of all saved visual adapters, splits and acceptance gates."""
import argparse
import gc
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import train_unibo_visual_facies as source
from scripts.train_core_facies_conservative import make_state
from scripts.facies_head_v2_features import predict
from scripts.kernel_facies_head import all_class_gates
from scripts.partial_facies_ridge import mixed_metrics
from scripts.vision_facies_adapter import load_original_vision,add_lora,restore_adapter,batch_inputs,post_features,normalized_post


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
 p.add_argument('--stage',choices=('full','fold0','fold1','fold2','final','aggregate'),default='full')
 a=p.parse_args();run=a.run.resolve()
 cfg=json.loads((run/'run_config.json').read_text(encoding='utf-8'));result=json.loads((run/'result.json').read_text(encoding='utf-8'));frozen=json.loads((run/'selection_frozen.json').read_text(encoding='utf-8'));contract=json.loads((run/'MODEL_CONTRACT.json').read_text(encoding='utf-8'))
 for f,h in cfg['dependencies'].items():
  if source.digest(f)!=h:raise ValueError('Training dependency changed: '+f)
 from scripts.classify_with_qwen_facies_head import streamed_sha
 base=Path(cfg['base']);ref=Path(cfg['reference']);prev=Path(cfg['previous'])
 if streamed_sha(base/'model.safetensors')!=contract['base_weights_sha256']:raise ValueError('Original base modified')
 if source.digest(run/'final/adapter.safetensors')!=result['head_sha256'] or source.digest(run/'selection_frozen.json')!=result['selection_frozen_sha256']:raise ValueError('Final adapter/selection changed')
 rows=source.read_rows(run/'train.jsonl');val=source.read_rows(run/'val.jsonl')
 if rows!=source.read_rows(prev/'train.jsonl') or val!=source.read_rows(ref/'val.jsonl') or [r['record_id'] for r in rows]!=frozen['training_record_ids']:raise ValueError('Training/validation rows changed')
 from scripts.train_qwen_source_pilot import configure_offline_environment,_deny_network
 configure_offline_environment();sys.addaudithook(_deny_network)
 import torch
 from safetensors.torch import load_file
 from transformers import AutoProcessor
 from PIL import Image
 torch.set_num_threads(4);torch.set_num_interop_threads(1)
 if a.stage!='aggregate' and not torch.cuda.is_available():raise RuntimeError('CUDA replay required')
 processor=AutoProcessor.from_pretrained(base,local_files_only=True) if a.stage!='aggregate' else None
 raw=load_file(str(prev/'features.safetensors'))['train'];mapping=frozen['fold_source_groups'];recipe=cfg['recipe'];dims=(6144,2048,114)
 images=[]
 for r in rows:
  if source.digest(r['image'])!=r['image_sha256']:raise ValueError('Train source image changed')
  if a.stage!='aggregate':
   with Image.open(r['image']) as im:images.append(im.convert('RGB'))
 for r in val:
  if source.digest(r['image'])!=r['metadata']['image_sha256']:raise ValueError('Validation image changed')
 oof=['']*len(rows);receipts=[]
 def evaluate(model,state,indices,image_list):
  coeff=(state['head_base']+state['head_delta']).cuda();norm={k:state[k].cuda() for k in ('mean','scale')};answers=[];scores=[]
  with torch.inference_mode():
   for offset in range(0,len(indices),4):
    batch=indices[offset:offset+4];features=post_features(model,batch_inputs(processor,[image_list[i] for i in batch],'cuda'))
    value=normalized_post(features,norm)@coeff;answers.extend('ABCDEF'[i] for i in value.argmax(1).tolist());scores.extend(value.cpu().tolist())
  return answers,scores
 for tag in ('fold0','fold1','fold2','final'):
  if a.stage not in ('full','aggregate',tag):continue
  folder=run/tag;receipt=json.loads((folder/'receipt.json').read_text(encoding='utf-8'));state=load_file(str(folder/'adapter.safetensors'))
  if source.digest(folder/'adapter.safetensors')!=receipt['adapter_sha256'] or any(not torch.isfinite(v).all() for v in state.values()):raise ValueError('Adapter hash/finiteness differs')
  if tag=='final':indices=list(range(len(rows)));held=[]
  else:
   fold=int(tag[-1]);indices=[i for i,r in enumerate(rows) if mapping[r['source_group']]!=fold];held=[i for i,r in enumerate(rows) if mapping[r['source_group']]==fold]
  if [rows[i]['record_id'] for i in indices]!=receipt['training_ids'] or [rows[i]['record_id'] for i in held]!=receipt['held_ids']:raise ValueError('Training/held split differs')
  if {rows[i]['source_group'] for i in indices}&{rows[i]['source_group'] for i in held}:raise ValueError('Group leakage')
  oldids=[i for i in indices if i<728]
  if [rows[i]['record_id'] for i in oldids]!=receipt['normalizer_training_ids']:raise ValueError('Normalizer included held/new data')
  initial,_=make_state(raw[oldids],[rows[i] for i in oldids],dims)
  for k in ('mean','scale'):
   if not torch.equal(state[k],initial[k].float()):raise ValueError('Old-only normalizer differs')
  if not torch.equal(state['head_base'],initial['coefficients'].float()):raise ValueError('Prior coefficients differ')
  if len(receipt['epochs'])!=recipe['epochs'] or any(e['examples']!=len(indices) or not 0<=e['mean_loss']<1e6 for e in receipt['epochs']):raise ValueError('Epoch/finite loss receipt differs')
  if receipt['trainable_vision_parameters']!=196608 or receipt['trainable_head_parameters']!=24582 or max(receipt['first_gradient_norms'].values())<=0:raise ValueError('Visual learning receipt differs')
  if max(float(v.abs().max()) for k,v in state.items() if k.endswith('lora_b'))==0:raise ValueError('Visual adapter remained zero')
  answerfile=folder/'held_answers.json' if tag!='final' else run/'val_answers.json'
  saved=json.loads(answerfile.read_text(encoding='utf-8'))
  expected_ids=receipt['held_ids'] if tag!='final' else [r['metadata']['record_id'] for r in val]
  if [r['record_id'] for r in saved]!=expected_ids:raise ValueError('Prediction records changed')
  if a.stage=='aggregate':
   replay=json.loads((folder/'replay_audit.json').read_text(encoding='utf-8'))
   required=dict(adapter_sha256=receipt['adapter_sha256'],training_receipt_sha256=source.digest(folder/'receipt.json'),answers_sha256=source.digest(answerfile),audit_script_sha256=source.digest(__file__))
   if any(replay.get(k)!=v for k,v in required.items()) or replay.get('fresh_inference_reproduced') is not True:raise ValueError('Fresh replay receipt changed/missing')
   if tag!='final':
    saved=json.loads(answerfile.read_text(encoding='utf-8'))
    for i,r in zip(held,saved):oof[i]=r['prediction']
   else:
    saved=json.loads(answerfile.read_text(encoding='utf-8'));vp=[r['prediction'] for r in saved];delta=replay['maximum_val_score_delta']
  else:
   model=load_original_vision(base);modules=add_lora(model,recipe['rank'],recipe['lora_alpha'],recipe['last_vision_blocks']);restore_adapter(model,state)
   if modules!=receipt['modules']:raise ValueError('Adapted module scope differs')
  if a.stage!='aggregate' and tag!='final':
   pred,_=evaluate(model,state,held,images);saved=json.loads((folder/'held_answers.json').read_text(encoding='utf-8'))
   if pred!=[r['prediction'] for r in saved]:raise ValueError('Fresh held predictions differ')
   for i,c in zip(held,pred):oof[i]=c
  elif a.stage!='aggregate':
   valimages=[]
   for r in val:
    with Image.open(r['image']) as im:valimages.append(im.convert('RGB'))
   vp,vs=evaluate(model,state,list(range(len(val))),valimages);saved=json.loads((run/'val_answers.json').read_text(encoding='utf-8'))
   if vp!=[r['prediction'] for r in saved]:raise ValueError('Fresh final val predictions differ')
   delta=float((torch.tensor(vs)-torch.tensor([r['scores'] for r in saved])).abs().max())
   if delta>1e-5:raise ValueError('Fresh final scores differ: '+str(delta))
  replay=dict(tag=tag,adapter_sha256=receipt['adapter_sha256'],training_receipt_sha256=source.digest(folder/'receipt.json'),answers_sha256=source.digest(answerfile),audit_script_sha256=source.digest(__file__),old_only_prior_verified=True,train_held_source_groups_disjoint=True,nonzero_visual_update=True,fresh_inference_reproduced=True)
  if tag=='final':replay['maximum_val_score_delta']=delta
  receipts.append(replay)
  if a.stage!='aggregate':
   source.write_json(folder/'replay_audit.json',replay);del model
  del state;gc.collect();torch.cuda.empty_cache();print(json.dumps(dict(event='replayed',tag=tag)),flush=True)
 if a.stage not in ('full','aggregate'):return
 if oof!=[r['prediction'] for r in json.loads((run/'oof_answers.json').read_text(encoding='utf-8'))]:raise ValueError('Complete OOF differs')
 old=mixed_metrics(rows[:728],oof[:728],source)['exact_six_class'];baseline=mixed_metrics(rows[:728],[r['prediction'] for r in json.loads((ref/'oof_answers.json').read_text(encoding='utf-8'))],source)['exact_six_class']
 if old!=result['old_oof'] or baseline!=result['baseline_old_oof'] or mixed_metrics(rows,oof,source)!=result['oof'] or all_class_gates(old,baseline)!=result['training_gates']:raise ValueError('OOF/selection metrics differ')
 expected=[r['conversations'][1]['value'] for r in val];vm=source.classification_metrics(expected,vp);rcfg=json.loads((ref/'run_config.json').read_text(encoding='utf-8'));vr=load_file(str(Path(rcfg['reference'])/'validation_features.safetensors'))['raw_val'][:,:6144]
 bp=['ABCDEF'[i] for i in predict(vr,load_file(str(ref/'facies_head.safetensors'))).argmax(1).tolist()];bm=source.classification_metrics(expected,bp);gates=all_class_gates(vm,bm);gates.update(train_eligible=all(result['training_gates'].values()),accuracy=vm['correct']>=bm['correct'])
 if vm!=result['val'] or bm!=result['baseline_val'] or gates!=result['validation_gates']:raise ValueError('Validation/gates differ')
 audit=dict(status=result['status'],adapter_sha256=result['head_sha256'],original_base_sha256_unchanged=True,language_model_loaded=False,language_weights_changed=False,
  real_visual_parameter_updates_verified=True,group_fold_receipts=receipts,all_oof_predictions_replayed=True,validation_predictions_and_scores_replayed=True,maximum_val_score_delta=delta,
  validation_gates=gates,distribution_approved=False,installed_in_application=False,prepared_test_pixels_opened=0,audit_script_sha256=source.digest(__file__),
  limitation='Three correlated source-proxy folds and repeatedly used val50, not independent field accuracy or92-facies/full-description validation')
 source.write_json(run/'audit_result.json',audit);print(json.dumps(audit,indent=2),flush=True)


if __name__=='__main__':main()
