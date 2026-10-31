"""Train paired focal/context DINO heads, selected only on source-group training OOF."""
import argparse
import json
from pathlib import Path
import shutil
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import train_unibo_visual_facies as source
from scripts.dino_facies_head import extract_features,pack_state,predict_features
from scripts.paired_context_facies_head import fit_head
from scripts.gray_facies_head import blend_scores
from scripts.facies_head_v2_features import predict
from scripts.train_core_facies_conservative import make_state
from scripts.partial_facies_ridge import mixed_metrics
from scripts.kernel_facies_head import all_class_gates
from scripts.facies_transfer_evaluation import controls

RECIPE=dict(alphas=[.1,1.,10.,100.],dino_weights=[.25,.5,1.],view='letterbox',context='Fixed3x along-core geometry, no masks/labels; same assigned original photo.100 external train fallbacks retain focal only.',
 features='Focal DINO1152 + context DINO1152, six separately normalized384 blocks; train-only mean/std',
 fit='Class+source-group balanced partial ridge; external0.5,endface0.25; generic floodplain supervises A+B only',
 rgb='Exact original Qwen v3 teacher, re-fit on old728 TRAIN subset per fold',
 selection='12 candidates by all-six old OOF gates, worst recall/F1 gain, mean group accuracy, macroF1, smaller DINO weight, larger alpha. Never use control predictions to select.',
 pretrained_encoders_changed=False,language_weights_changed=False,distribution_approved=False,installed_in_application=False)


def candidate_key(item):
 m,b=item['old_oof'],item['baseline_old_oof']
 return (all(item['all_class_gates'].values()),min(m['per_class_recall'][c]-b['per_class_recall'][c] for c in 'ABCDEF'),min(m['per_class_f1'][c]-b['per_class_f1'][c] for c in 'ABCDEF'),item['mean_group_accuracy'],m['macro_f1'],-item['dino_weight'],item['alpha'])


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--context',type=Path,default=ROOT.parent/'vlm_core_v1/data/paired_context_20261031_v16_along_core');a=p.parse_args();out,data=a.output.resolve(),a.context.resolve();parent=ROOT.parent/'vlm_core_v1/outputs/core_qwen_dino_facies_20261031_v12'
 if out.exists() or not out.is_relative_to(ROOT.parent/'vlm_core_v1/outputs'):raise ValueError('Choose new local output')
 pcfg=json.loads((parent/'run_config.json').read_text(encoding='utf-8'));ref,original=Path(pcfg['reference']),Path(pcfg['previous']);contract=json.loads((parent/'MODEL_CONTRACT.json').read_text(encoding='utf-8'));summary=json.loads((data/'summary.json').read_text(encoding='utf-8'))
 if summary['do_not_train'] or summary['masks_read']!=0 or summary['context_counts']!={'train':722,'val':50,'external':16}:raise ValueError('Context not approved for this experiment')
 for f,h in pcfg['dependencies'].items():
  if source.digest(f)!=h:raise ValueError('Parent dependency changed')
 rows,val,held=[source.read_rows(original/f) for f in ('train.jsonl','val.jsonl','external_holdout.jsonl')];ctx={s:source.read_rows(data/(s+'.jsonl')) for s in ('train','val','external')}
 for s,rr in (('train',rows),('val',val),('external',held)):
  if [r.get('record_id',r.get('metadata',{}).get('record_id')) for r in rr]!=[r['record_id'] for r in ctx[s]]:raise ValueError('Context/focal order changed')
  for r,c in zip(rr,ctx[s]):
   if source.digest(r['image'])!=c['focal_image_sha256']:raise ValueError('Focal changed')
 if source.package_versions()!=contract['package_versions']:raise ValueError('Runtime changed')
 out.mkdir(parents=True)
 for name,rr in (('train',rows),('val',val),('external_holdout',held)):source.write_rows(out/(name+'.jsonl'),rr)
 deps=[Path(__file__),ROOT/'scripts/paired_context_facies_head.py',ROOT/'scripts/prepare_paired_context_v16.py',ROOT/'scripts/dino_facies_head.py',ROOT/'scripts/gray_facies_head.py',ROOT/'scripts/facies_transfer_evaluation.py',ROOT/'scripts/kernel_facies_head.py',ROOT/'scripts/partial_facies_ridge.py',ROOT/'scripts/facies_head_v2_features.py',ROOT/'scripts/train_core_facies_conservative.py',parent/'dino_train.safetensors',parent/'run_config.json',original/'features.safetensors',original/'selection_frozen.json',ref/'facies_head.safetensors',ref/'oof_answers.json',data/'train.jsonl',data/'val.jsonl',data/'external.jsonl',data/'summary.json']
 cfg=dict(recipe=RECIPE,previous=str(parent),original=str(original),reference=str(ref),context=str(data),qwen_base=pcfg['qwen_base'],dino=pcfg['dino'],dino_contract=pcfg['dino_contract'],train_count=len(rows),validation_count=len(val),external_count=len(held),dependencies={str(f):source.digest(f) for f in deps},source_rights=pcfg['source_rights'],distribution_approved=False,installed_in_application=False,prepared_test_pixels_opened=0)
 source.write_json(out/'run_config.json',cfg);shutil.copy2(__file__,out/'training_script_snapshot.py')
 from scripts.train_qwen_source_pilot import configure_offline_environment,_deny_network
 configure_offline_environment();sys.addaudithook(_deny_network)
 import torch
 from safetensors.torch import load_file,save_file
 torch.set_num_threads(4);torch.set_num_interop_threads(1)
 def log(event,**kw):
  v=dict(event=event,**kw)
  with (out/'progress.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(v)+'\n')
  if event!='dino_features' or kw['done']%80==0 or kw['done']==kw['total']:print(json.dumps(v),flush=True)
 context_features=extract_features(ctx['train'],Path(cfg['dino']),log)['letterbox'];focal=load_file(str(parent/'dino_train.safetensors'))['letterbox'];df=torch.cat((focal,context_features),1);save_file({'paired_train':df},str(out/'paired_train.safetensors'));rgb=load_file(str(original/'features.safetensors'))['train'];mapping=json.loads((original/'selection_frozen.json').read_text(encoding='utf-8'))['fold_source_groups'];baseline=mixed_metrics(rows[:728],[r['prediction'] for r in json.loads((ref/'oof_answers.json').read_text(encoding='utf-8'))],source)['exact_six_class']
 folds=[]
 for f in range(3):
  ids=[i for i,r in enumerate(rows) if mapping[r['source_group']]!=f];test=[i for i,r in enumerate(rows) if mapping[r['source_group']]==f];old=[i for i in ids if i<728]
  if {rows[i]['source_group'] for i in ids}&{rows[i]['source_group'] for i in test}:raise ValueError('Group leakage')
  teacher,_=make_state(rgb[old],[rows[i] for i in old],(6144,2048,114));folds.append((ids,test,predict(rgb[test],teacher)))
 candidates=[];predictions={}
 for alpha in RECIPE['alphas']:
  ds=[predict(df[test],fit_head(df[ids],[rows[i] for i in ids],alpha)) for ids,test,ts in folds]
  for weight in RECIPE['dino_weights']:
   pred=['']*len(rows)
   for (_,test,ts),scores in zip(folds,ds):
    for i,c in zip(test,blend_scores(ts,scores,weight).argmax(1).tolist()):pred[i]='ABCDEF'[c]
   oldm=mixed_metrics(rows[:728],pred[:728],source)['exact_six_class'];allm=mixed_metrics(rows,pred,source);item=dict(alpha=alpha,dino_weight=weight,old_oof=oldm,baseline_old_oof=baseline,all_class_gates=all_class_gates(oldm,baseline),mean_group_accuracy=allm['mean_source_group_accuracy']);candidates.append(item);predictions[alpha,weight]=pred;log('candidate',alpha=alpha,dino_weight=weight,old_correct=oldm['correct'],macro_f1=oldm['macro_f1'],eligible=all(item['all_class_gates'].values()))
 winner=max(candidates,key=candidate_key);oof=predictions[winner['alpha'],winner['dino_weight']];teacher=load_file(str(ref/'facies_head.safetensors'));state=pack_state(teacher,fit_head(df,rows,winner['alpha']),winner['dino_weight']);save_file({k:v.contiguous() for k,v in state.items()},str(out/'facies_head.safetensors'))
 frozen=dict(stage='before_context_control_features_and_predictions',winner=winner,candidates=candidates,fold_source_groups=mapping,training_ids=[r['record_id'] for r in rows],head_sha256=source.digest(out/'facies_head.safetensors'));source.write_json(out/'selection_frozen.json',frozen);source.write_json(out/'oof_answers.json',[dict(record_id=r['record_id'],prediction=c,allowed_codes=r['allowed_codes'],source_group=r['source_group'],fold=mapping[r['source_group']]) for r,c in zip(rows,oof)])
 ev=load_file(str(parent/'dino_evaluation.safetensors'));vdf=torch.cat((ev['val__letterbox'],extract_features(ctx['val'],Path(cfg['dino']))['letterbox']),1);hdf=torch.cat((ev['external__letterbox'],extract_features(ctx['external'],Path(cfg['dino']))['letterbox']),1);save_file({'paired_val':vdf,'paired_external':hdf},str(out/'paired_evaluation.safetensors'))
 rcfg=json.loads((ref/'run_config.json').read_text(encoding='utf-8'));vr=load_file(str(Path(rcfg['reference'])/'validation_features.safetensors'))['raw_val'][:,:6144];hr=load_file(str(original/'external_features.safetensors'))['external_holdout']
 def codes(s):return ['ABCDEF'[i] for i in s.argmax(1).tolist()]
 vp,bp=codes(predict_features(vr,vdf,state)),codes(predict(vr,teacher));hp,hb=codes(predict_features(hr,hdf,state)),codes(predict(hr,teacher));evaluation=controls(val,held,vp,bp,hp,hb,winner['all_class_gates'])
 result=dict(status='research_all_gates_passed_requires_independent_wells' if all(evaluation['validation_gates'].values()) else 'not_promoted_all_class_or_new_well_gate_failed',selected={k:winner[k] for k in ('alpha','dino_weight')},head_sha256=source.digest(out/'facies_head.safetensors'),selection_frozen_sha256=source.digest(out/'selection_frozen.json'),old_oof=winner['old_oof'],baseline_old_oof=baseline,training_gates=winner['all_class_gates'],oof=mixed_metrics(rows,oof,source),train=mixed_metrics(rows,codes(predict_features(rgb,df,state)),source),**evaluation,distribution_approved=False,installed_in_application=False,pretrained_encoders_changed=False,language_weights_changed=False,controls_reused_from_previous_development=True)
 source.write_json(out/'result.json',result)
 for name,rr,pred,basepred in [('val',val,vp,bp),('external',held,hp,hb)]:source.write_json(out/(name+'_answers.json'),[dict(record_id=r.get('record_id',r.get('metadata',{}).get('record_id')),well=r.get('well'),expected_code=r['allowed_codes'][0] if 'allowed_codes' in r else r['conversations'][1]['value'],prediction=c,baseline_prediction=b) for r,c,b in zip(rr,pred,basepred)])
 contract.update(type='qwen_dino_paired_focal_context',feature_contract_version='qwen_dino_paired_context_facies_v1',head_sha256=result['head_sha256'],acceptance_status=result['status'],training_recipe=RECIPE,selected_profile=result['selected'],distribution_approved=False)
 source.write_json(out/'MODEL_CONTRACT.json',contract);log('complete',old_oof=winner['old_oof']['correct'],val=result['val']['correct'],external=result['external_holdout']['exact_six_class']['correct'],status=result['status'])


if __name__=='__main__':main()
