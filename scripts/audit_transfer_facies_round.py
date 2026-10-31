"""Fresh feature and grouped-fit replay for research rounds11-16."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import train_unibo_visual_facies as source
from scripts.facies_head_v2_features import predict,normalize
from scripts.train_core_facies_conservative import make_state
from scripts.partial_facies_ridge import mixed_metrics
from scripts.kernel_facies_head import all_class_gates
from scripts.gray_facies_head import blend_scores
from scripts.facies_transfer_evaluation import controls

NAMES={11:'core_qwen_gray_facies_20261031_v11',12:'core_qwen_dino_facies_20261031_v12',13:'core_qwen_guarded_facies_20261031_v13',14:'core_qwen_residual_facies_20261031_v14',15:'core_qwen_residual_guard_facies_20261031_v15',16:'core_qwen_paired_context_facies_20261031_v16'}


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--round',type=int,choices=sorted(NAMES),required=True);p.add_argument('--skip-feature-replay',action='store_true');a=p.parse_args();number=a.round;outputs=ROOT.parent/'vlm_core_v1/outputs';run=outputs/NAMES[number]
 def read(path):return json.loads(path.read_text(encoding='utf-8'))
 cfg,result,frozen,contract=[read(run/f) for f in ('run_config.json','result.json','selection_frozen.json','MODEL_CONTRACT.json')]
 for f,h in cfg['dependencies'].items():
  if source.digest(f)!=h:raise ValueError('Training dependency changed: '+f)
 original=outputs/'core_qwen_new_well_facies_20261031_v10';ref=Path(cfg['reference']);d12=outputs/NAMES[12]
 rows,val,held=[source.read_rows(run/f) for f in ('train.jsonl','val.jsonl','external_holdout.jsonl')]
 for rr,name in ((rows,'train.jsonl'),(val,'val.jsonl'),(held,'external_holdout.jsonl')):
  if rr!=source.read_rows(original/name):raise ValueError('Original row order changed')
 if [r['record_id'] for r in rows]!=frozen['training_ids']:raise ValueError('Frozen training IDs differ')
 for r in rows+val+held:
  if source.digest(r['image'])!=r.get('image_sha256',r.get('metadata',{}).get('image_sha256')):raise ValueError('Source image changed')
 from scripts.classify_with_qwen_facies_head import streamed_sha
 base=Path(cfg.get('qwen_base',cfg.get('base','')))
 if streamed_sha(base/'model.safetensors')!=contract['base_weights_sha256']:raise ValueError('Original Qwen base changed')
 for f,h in contract['preprocessing_files_sha256'].items():
  if source.digest(base/f)!=h:raise ValueError('Qwen processor changed')
 if number!=11:
  dino=Path(cfg['dino'])
  for f,h in cfg['dino_contract']['sha256'].items():
   if source.digest(dino/f)!=h:raise ValueError('DINO model/processor changed')
 from scripts.train_qwen_source_pilot import configure_offline_environment,_deny_network
 configure_offline_environment();sys.addaudithook(_deny_network)
 import torch
 from safetensors.torch import load_file
 torch.set_num_threads(4);torch.set_num_interop_threads(1)
 rgb=load_file(str(original/'features.safetensors'))['train'];teacher=load_file(str(ref/'facies_head.safetensors'));mapping=frozen['fold_source_groups'];recipe=cfg['recipe'];features={};feature_delta=0.
 # Re-extract every feature, not a small favourable sample, for the new input families.
 if number==11:
  from scripts.vision_facies_adapter import load_original_vision,batch_inputs
  from scripts.gray_facies_head import grayscale_view
  from scripts.train_qwen_visual_facies_head import pool_features
  from transformers import AutoProcessor
  from PIL import Image
  if not a.skip_feature_replay:model=load_original_vision(base);processor=AutoProcessor.from_pretrained(base,local_files_only=True)
  def extract(rr):
   fresh=[]
   for i,r in enumerate(rr):
    with Image.open(r['image']) as im:inp=batch_inputs(processor,[grayscale_view(im)],'cuda')
    with torch.inference_mode():fresh.append(pool_features(model(inp['pixel_values'],grid_thw=inp['grid'],return_dict=True)).cpu())
    if (i+1)%100==0:print(json.dumps(dict(event='gray_replay',done=i+1,total=len(rr))),flush=True)
   return torch.stack(fresh)
  gray=load_file(str(run/'gray_train.safetensors'))['gray_train'];ev=load_file(str(run/'gray_evaluation.safetensors'))
  if not a.skip_feature_replay:
   feature_delta=max(float((extract(rr)-expected).abs().max()) for rr,expected in ((rows,gray),(val,ev['gray_val']),(held,ev['gray_external'])));del model,processor;torch.cuda.empty_cache()
  features={'gray':gray}
 elif number==12:
  from scripts.dino_facies_head import extract_features
  features=load_file(str(run/'dino_train.safetensors'));ev=load_file(str(run/'dino_evaluation.safetensors'))
  if not a.skip_feature_replay:
   for rr,expected in ((rows,features),(val,{k.removeprefix('val__'):v for k,v in ev.items() if k.startswith('val__')}),(held,{k.removeprefix('external__'):v for k,v in ev.items() if k.startswith('external__')})):
    fresh=extract_features(rr,dino);feature_delta=max(feature_delta,max(float((fresh[k]-v).abs().max()) for k,v in expected.items()));print(json.dumps(dict(event='dino_replayed',images=len(rr))),flush=True)
 elif number==16:
  from scripts.dino_facies_head import extract_features
  data=Path(cfg['context']);summary=read(data/'summary.json')
  if summary['do_not_train'] or summary['masks_read']!=0:raise ValueError('Rejected context')
  features={'paired':load_file(str(run/'paired_train.safetensors'))['paired_train']};ev=load_file(str(run/'paired_evaluation.safetensors'));focal=load_file(str(d12/'dino_train.safetensors'));fev=load_file(str(d12/'dino_evaluation.safetensors'))
  if not a.skip_feature_replay:
   for split,expected,ff in (('train',features['paired'],focal['letterbox']),('val',ev['paired_val'],fev['val__letterbox']),('external',ev['paired_external'],fev['external__letterbox'])):
    rr=source.read_rows(data/(split+'.jsonl'));fresh=extract_features(rr,dino)['letterbox'];feature_delta=max(feature_delta,float((torch.cat((ff,fresh),1)-expected).abs().max()));print(json.dumps(dict(event='context_replayed',split=split)),flush=True)
 else:
  if not (d12/'audit_result.json').is_file():raise ValueError('Audit DINO features first')
  features=load_file(str(d12/'dino_train.safetensors'));ev=load_file(str(d12/'dino_evaluation.safetensors'))
 if feature_delta>1e-5:raise ValueError('Fresh features differ: '+str(feature_delta))
 folds=[];teacher_oof=torch.empty((len(rows),6),dtype=torch.float64)
 for f in range(3):
  ids=[i for i,r in enumerate(rows) if mapping[r['source_group']]!=f];test=[i for i,r in enumerate(rows) if mapping[r['source_group']]==f];old=[i for i in ids if i<728]
  if {rows[i]['source_group'] for i in ids}&{rows[i]['source_group'] for i in test}:raise ValueError('Group leakage')
  t,_=make_state(rgb[old],[rows[i] for i in old],(6144,2048,114));ts=predict(rgb[test],t);teacher_oof[test]=ts;folds.append((ids,test,t,ts))
 baseline=mixed_metrics(rows[:728],['ABCDEF'[i] for i in teacher_oof[:728].argmax(1).tolist()],source)['exact_six_class']
 if baseline!=result['baseline_old_oof'] or ['ABCDEF'[i] for i in teacher_oof[:728].argmax(1).tolist()]!=[r['prediction'] for r in read(ref/'oof_answers.json')]:raise ValueError('Old-only teacher OOF differs')
 def codes(x):return ['ABCDEF'[i] for i in x.argmax(1).tolist()]
 candidate_count=0;receipts=[]
 if number in (11,12,16):
  if number==11:
   from scripts.gray_facies_head import fit_gray as fithead,candidate_key,pack_state,predict_features
   views=['gray'];weight_key='gray_weight';weights=recipe['gray_weights']
  elif number==12:
   from scripts.dino_facies_head import fit_head as fithead,candidate_key,pack_state,predict_features
   views=recipe['views'];weight_key='dino_weight';weights=recipe['dino_weights']
  else:
   from scripts.paired_context_facies_head import fit_head as fithead
   from scripts.train_paired_context_v16 import candidate_key
   from scripts.dino_facies_head import pack_state,predict_features
   views=['paired'];weight_key='dino_weight';weights=recipe['dino_weights']
  candidates=[];allpred={}
  for view in views:
   for alpha in recipe['alphas']:
    ds=[predict(features[view][test],fithead(features[view][ids],[rows[i] for i in ids],alpha)) for ids,test,t,ts in folds]
    for weight in weights:
     pred=['']*len(rows)
     for (_,test,t,ts),ss in zip(folds,ds):
      for i,c in zip(test,codes(blend_scores(ts,ss,weight))):pred[i]=c
     m=mixed_metrics(rows[:728],pred[:728],source)['exact_six_class'];mm=mixed_metrics(rows,pred,source);item=dict(alpha=alpha,old_oof=m,baseline_old_oof=baseline,all_class_gates=all_class_gates(m,baseline),mean_group_accuracy=mm['mean_source_group_accuracy']);item[weight_key]=weight
     if number==12:item['view']=view
     candidates.append(item);allpred[view,alpha,weight]=pred
  if candidates!=frozen['candidates']:raise ValueError('Train-only candidate metrics differ')
  winner=max(candidates,key=candidate_key)
  if winner!=frozen['winner']:raise ValueError('Winner selection differs')
  view=winner.get('view',views[0]);oof=allpred[view,winner['alpha'],winner[weight_key]];state=pack_state(teacher,fithead(features[view],rows,winner['alpha']),winner[weight_key]);saved=load_file(str(run/'facies_head.safetensors'))
  if any(not torch.equal(v,saved[k]) for k,v in state.items()):raise ValueError('Saved fitted head differs')
  candidate_count=len(candidates)
  def infer_control(raw,split):return codes(predict_features(raw,ev['gray_'+split] if number==11 else ev['paired_'+split] if number==16 else ev[('val__' if split=='val' else 'external__')+view],state))
  trainp=codes(predict_features(rgb,features[view],state))
 else:
  from scripts.residual_facies_head import infer
  def residual_state(tag,indices,test):
   parent=run if number==14 else Path(cfg['parent']);rr=read(parent/tag/'receipt.json');st=load_file(str(parent/tag/'residual.safetensors'));rc=cfg['recipe'] if number==14 else cfg['recipe']['residual_recipe']
   if rr['training_ids']!=[rows[i]['record_id'] for i in indices] or rr['held_ids']!=[rows[i]['record_id'] for i in test] or rr['epochs']!=rc['epochs'] or len(rr['epoch_mean_losses'])!=rc['epochs'] or rr['first_output_gradient_norm']<=0 or rr['trainable_parameters']!=74566:raise ValueError('Neural training receipt differs')
   if source.digest(parent/tag/'residual.safetensors')!=rr['head_sha256'] or any(not torch.isfinite(v).all() for v in st.values()):raise ValueError('Neural weights changed')
   raw=features[rc['view']][indices].double()
   if not torch.equal(st['norm__mean'],raw.mean(0)) or not torch.equal(st['norm__scale'],raw.std(0,unbiased=False).clamp_min(1e-3)):raise ValueError('Neural normalizer included held data')
   receipts.append(dict(tag=tag,head_sha256=rr['head_sha256'],actual_gradient_verified=True,train_only_normalizer_verified=True));return st,rc
  scores=torch.empty((len(rows),6),dtype=torch.float32 if number in (14,15) else torch.float64)
  if number in (13,15):
   from scripts.guarded_facies_head import guard_sets,guarded_predictions,selection_key
  for f,(ids,test,t,ts) in enumerate(folds):
   if number==13:
    from scripts.dino_facies_head import fit_head
    sel=cfg['fixed_dino_selection'];scores[test]=blend_scores(ts,predict(features[sel['view']][test],fit_head(features[sel['view']][ids],[rows[i] for i in ids],sel['alpha'])),sel['dino_weight'])
   else:
    st,rc=residual_state('fold'+str(f),ids,test);scores[test]=infer(features[rc['view']][test],ts,st,rc)
    if float((scores[test].double()-ts).abs().max())>rc['cap']+1e-6:raise ValueError('Neural correction exceeded cap')
  if number in (13,15):
   candidates=[];preds={}
   for gs in guard_sets():
    pp=guarded_predictions(teacher_oof,scores,gs);m=mixed_metrics(rows[:728],pp[:728],source)['exact_six_class'];mm=mixed_metrics(rows,pp,source);item=dict(guard_classes=list(gs),old_oof=m,baseline_old_oof=baseline,all_class_gates=all_class_gates(m,baseline),mean_group_accuracy=mm['mean_source_group_accuracy']);candidates.append(item);preds[gs]=pp
   if candidates!=frozen['candidates'] or max(candidates,key=selection_key)!=frozen['winner']:raise ValueError('Guard policy selection differs')
   guards=frozen['winner']['guard_classes'];oof=preds[tuple(guards)];candidate_count=len(candidates);policy=read(run/'policy.json')
   if policy['guard_classes']!=guards or source.digest(run/'policy.json')!=result['policy_sha256']:raise ValueError('Guard policy changed')
  else:oof=codes(scores)
  if number==13:
   from scripts.dino_facies_head import pack_state,predict_features
   state=pack_state(teacher,fit_head(features[sel['view']],rows,sel['alpha']),sel['dino_weight']);saved=load_file(str(run/'facies_head.safetensors'))
   if any(not torch.equal(v,saved[k]) for k,v in state.items()):raise ValueError('Guarded fusion head differs')
   def infer_control(raw,split):
    ts=predict(raw,teacher);return guarded_predictions(ts,predict_features(raw,ev[('val__' if split=='val' else 'external__')+sel['view']],state),guards)
   trainp=guarded_predictions(predict(rgb,teacher),predict_features(rgb,features[sel['view']],state),guards)
  else:
   state,rc=residual_state('final',list(range(len(rows))),[])
   def infer_control(raw,split):
    ts=predict(raw,teacher);cs=infer(ev[('val__' if split=='val' else 'external__')+rc['view']],ts,state,rc)
    return codes(cs) if number==14 else guarded_predictions(ts,cs,guards)
   trainp=codes(infer(features[rc['view']],predict(rgb,teacher),state,rc)) if number==14 else None
 if oof!=[r['prediction'] for r in read(run/'oof_answers.json')]:raise ValueError('Fresh grouped predictions differ')
 oldm=mixed_metrics(rows[:728],oof[:728],source)['exact_six_class'];tg=all_class_gates(oldm,baseline)
 if oldm!=result['old_oof'] or tg!=result['training_gates'] or mixed_metrics(rows,oof,source)!=result['oof']:raise ValueError('Grouped metrics/gates differ')
 if trainp is not None and mixed_metrics(rows,trainp,source)!=result['train']:raise ValueError('Training metrics differ')
 rcfg=read(ref/'run_config.json');vr=load_file(str(Path(rcfg['reference'])/'validation_features.safetensors'))['raw_val'][:,:6144];hr=load_file(str(original/'external_features.safetensors'))['external_holdout'];vp,hp=infer_control(vr,'val'),infer_control(hr,'external');bp,hb=codes(predict(vr,teacher)),codes(predict(hr,teacher));evaluation=controls(val,held,vp,bp,hp,hb,tg)
 for k,v in evaluation.items():
  if v!=result[k]:raise ValueError('Control metric/gate differs: '+k)
 for name,pred in (('val',vp),('external',hp)):
  if pred!=[r['prediction'] for r in read(run/(name+'_answers.json'))]:raise ValueError('Control answers differ')
 head=run/('final/residual.safetensors' if number==14 else 'residual.safetensors' if number==15 else 'facies_head.safetensors')
 if source.digest(head)!=result['head_sha256'] or source.digest(run/'selection_frozen.json')!=result['selection_frozen_sha256']:raise ValueError('Final head/selection changed')
 audit=dict(status=result['status'],round=number,head_sha256=result['head_sha256'],train_only_selection_candidates_reproduced=candidate_count,all_grouped_and_control_predictions_reproduced=True,
  all_class_and_per_well_gates_reproduced=True,source_images_verified=888,fresh_feature_max_delta=feature_delta if not a.skip_feature_replay else None,fresh_feature_images=888 if number in (11,12,16) and not a.skip_feature_replay else 0,
  feature_replay_explicitly_skipped=a.skip_feature_replay,
  feature_parent_audited=number not in (11,12,16),neural_fold_receipts=receipts,distribution_approved=False,installed_in_application=False,prepared_test_pixels_opened=0,
  controls_reused_from_prior_development=True,audit_script_sha256=source.digest(__file__),limitation='Six source classes, correlated crops and reused controls; not independent field accuracy or92-class/full-description validation')
 source.write_json(run/'audit_result.json',audit);print(json.dumps(audit,indent=2),flush=True)


if __name__=='__main__':main()
