"""Fit a fixed DINO head and choose teacher class guards on training-group OOF only."""
import argparse
import json
from pathlib import Path
import shutil
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import train_unibo_visual_facies as source
from scripts.dino_facies_head import fit_head,pack_state,predict_features
from scripts.gray_facies_head import blend_scores
from scripts.guarded_facies_head import guard_sets,guarded_predictions,selection_key
from scripts.facies_head_v2_features import predict
from scripts.train_core_facies_conservative import make_state
from scripts.kernel_facies_head import all_class_gates
from scripts.partial_facies_ridge import mixed_metrics

RECIPE=dict(guard_subsets='All64 subsets of six teacher-predicted classes; selected by train OOF only',
 dino_recipe='Fixed train-selected v12 view/alpha/weight; no joint refit/search against controls',
 guard_semantics='Keep all teacher predictions of a guarded class, including incorrect ones. Other predictions use the trained Qwen/DINO blend.',
 selection='All-six recall/F1 nondecreasing and macroF1 higher, worst recall/F1 gain, macroF1, mean group accuracy, fewer guards, deterministic lexical tie-break',
 risk='64 policy candidates on same source-proxy OOF can overfit. Controls already used in prior development, not new blind validation.',
 pretrained_encoders_changed=False,language_weights_changed=False,distribution_approved=False,installed_in_application=False)


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output.resolve();prev=ROOT.parent/'vlm_core_v1/outputs/core_qwen_dino_facies_20261031_v12'
 if out.exists() or not out.is_relative_to(ROOT.parent/'vlm_core_v1/outputs'):raise ValueError('Choose new local output')
 pcfg=json.loads((prev/'run_config.json').read_text(encoding='utf-8'));pfrozen=json.loads((prev/'selection_frozen.json').read_text(encoding='utf-8'));selection={k:pfrozen['winner'][k] for k in ('view','alpha','dino_weight')};ref=Path(pcfg['reference']);original=Path(pcfg['previous']);contract=json.loads((prev/'MODEL_CONTRACT.json').read_text(encoding='utf-8'))
 for f,h in pcfg['dependencies'].items():
  if source.digest(f)!=h:raise ValueError('Previous dependency changed')
 rows,val,held=[source.read_rows(prev/f) for f in ('train.jsonl','val.jsonl','external_holdout.jsonl')]
 if len(rows)!=822 or len(val)!=50 or len(held)!=16 or source.package_versions()!=contract['package_versions']:raise ValueError('Runtime/data changed')
 out.mkdir(parents=True)
 for name,rr in (('train',rows),('val',val),('external_holdout',held)):source.write_rows(out/(name+'.jsonl'),rr)
 deps=[Path(__file__),ROOT/'scripts/guarded_facies_head.py',ROOT/'scripts/dino_facies_head.py',ROOT/'scripts/gray_facies_head.py',ROOT/'scripts/kernel_facies_head.py',ROOT/'scripts/facies_head_v2_features.py',ROOT/'scripts/train_core_facies_conservative.py',ROOT/'scripts/partial_facies_ridge.py',prev/'dino_train.safetensors',prev/'selection_frozen.json',prev/'run_config.json',original/'features.safetensors',ref/'facies_head.safetensors',ref/'oof_answers.json']
 cfg=dict(recipe=RECIPE,fixed_dino_selection=selection,previous=str(prev),reference=str(ref),original=str(original),dino=pcfg['dino'],dino_contract=pcfg['dino_contract'],qwen_base=pcfg['qwen_base'],train_count=822,validation_count=50,external_count=16,
 dependencies={str(f):source.digest(f) for f in deps},source_rights=pcfg['source_rights'],distribution_approved=False,installed_in_application=False,prepared_test_pixels_opened=0)
 source.write_json(out/'run_config.json',cfg);shutil.copy2(__file__,out/'training_script_snapshot.py')
 from scripts.train_qwen_source_pilot import configure_offline_environment,_deny_network
 configure_offline_environment();sys.addaudithook(_deny_network)
 import torch
 from safetensors.torch import load_file,save_file
 torch.set_num_threads(4);torch.set_num_interop_threads(1)
 rgb=load_file(str(original/'features.safetensors'))['train'];df=load_file(str(prev/'dino_train.safetensors'))[selection['view']];mapping=pfrozen['fold_source_groups'];ts=torch.empty((len(rows),6),dtype=torch.float64);ds=torch.empty_like(ts)
 for f in range(3):
  ids=[i for i,r in enumerate(rows) if mapping[r['source_group']]!=f];test=[i for i,r in enumerate(rows) if mapping[r['source_group']]==f];old=[i for i in ids if i<728]
  if {rows[i]['source_group'] for i in ids}&{rows[i]['source_group'] for i in test}:raise ValueError('Group leakage')
  teacher,_=make_state(rgb[old],[rows[i] for i in old],(6144,2048,114));ts[test]=predict(rgb[test],teacher);ds[test]=blend_scores(ts[test],predict(df[test],fit_head(df[ids],[rows[i] for i in ids],selection['alpha'])),selection['dino_weight'])
 baseline=mixed_metrics(rows[:728],['ABCDEF'[i] for i in ts[:728].argmax(1).tolist()],source)['exact_six_class']
 if ['ABCDEF'[i] for i in ts[:728].argmax(1).tolist()]!=[r['prediction'] for r in json.loads((ref/'oof_answers.json').read_text(encoding='utf-8'))]:raise ValueError('Teacher OOF differs')
 candidates=[];predictions={}
 for guards in guard_sets():
  pred=guarded_predictions(ts,ds,guards);m=mixed_metrics(rows[:728],pred[:728],source)['exact_six_class'];mm=mixed_metrics(rows,pred,source)
  item=dict(guard_classes=list(guards),old_oof=m,baseline_old_oof=baseline,all_class_gates=all_class_gates(m,baseline),mean_group_accuracy=mm['mean_source_group_accuracy'])
  candidates.append(item);predictions[guards]=pred
 winner=max(candidates,key=selection_key);guards=winner['guard_classes'];oof=predictions[tuple(guards)]
 teacher=load_file(str(ref/'facies_head.safetensors'));state=pack_state(teacher,fit_head(df,rows,selection['alpha']),selection['dino_weight']);save_file({k:v.contiguous() for k,v in state.items()},str(out/'facies_head.safetensors'))
 # Save the learned policy and its exact head before reading control features.
 policy=dict(version='qwen_dino_teacher_class_guard_v1',guard_classes=guards,dino_selection=selection,head_sha256=source.digest(out/'facies_head.safetensors'))
 source.write_json(out/'policy.json',policy);frozen=dict(stage='before_reading_control_features_or_predictions',winner=winner,candidates=candidates,fold_source_groups=mapping,training_ids=[r['record_id'] for r in rows],policy_sha256=source.digest(out/'policy.json'))
 source.write_json(out/'selection_frozen.json',frozen);source.write_json(out/'oof_answers.json',[dict(record_id=r['record_id'],prediction=c,allowed_codes=r['allowed_codes'],source_group=r['source_group'],fold=mapping[r['source_group']]) for r,c in zip(rows,oof)]);save_file({'teacher':ts,'dino_blend':ds},str(out/'oof_scores.safetensors'))
 evaluation=load_file(str(prev/'dino_evaluation.safetensors'));rcfg=json.loads((ref/'run_config.json').read_text(encoding='utf-8'));vr=load_file(str(Path(rcfg['reference'])/'validation_features.safetensors'))['raw_val'][:,:6144];hr=load_file(str(original/'external_features.safetensors'))['external_holdout']
 vts,hts=predict(vr,teacher),predict(hr,teacher);vp=guarded_predictions(vts,predict_features(vr,evaluation['val__'+selection['view']],state),guards);hp=guarded_predictions(hts,predict_features(hr,evaluation['external__'+selection['view']],state),guards);bp=['ABCDEF'[i] for i in vts.argmax(1).tolist()];hb=['ABCDEF'[i] for i in hts.argmax(1).tolist()];expected=[r['conversations'][1]['value'] for r in val];vm,bm=source.classification_metrics(expected,vp),source.classification_metrics(expected,bp);hm,hbm=mixed_metrics(held,hp,source),mixed_metrics(held,hb,source)
 gates=all_class_gates(vm,bm);gates.update(train_eligible=all(winner['all_class_gates'].values()),accuracy=vm['correct']>=bm['correct'],no_previously_correct_case_lost=not any(b==e and c!=e for e,b,c in zip(expected,bp,vp)))
 wg={}
 for well in sorted({r['well'] for r in held}):
  for c in sorted({r['allowed_codes'][0] for r in held if r['well']==well}):wg[well+'_'+c]=sum(p==c for r,p in zip(held,hp) if r['well']==well and r['allowed_codes']==[c])>=sum(p==c for r,p in zip(held,hb) if r['well']==well and r['allowed_codes']==[c])
 wg['at_least_one_gain']=hm['exact_six_class']['correct']>hbm['exact_six_class']['correct'];gates['new_wells_eligible']=all(wg.values())
 trainp=guarded_predictions(predict(rgb,teacher),predict_features(rgb,df,state),guards)
 result=dict(status='research_all_gates_passed_requires_independent_wells' if all(gates.values()) else 'not_promoted_all_class_or_new_well_gate_failed',selected=dict(guard_classes=guards,**selection),eligible_policy_count=sum(all(c['all_class_gates'].values()) for c in candidates),head_sha256=policy['head_sha256'],policy_sha256=source.digest(out/'policy.json'),selection_frozen_sha256=source.digest(out/'selection_frozen.json'),old_oof=winner['old_oof'],baseline_old_oof=baseline,training_gates=winner['all_class_gates'],oof=mixed_metrics(rows,oof,source),train=mixed_metrics(rows,trainp,source),val=vm,baseline_val=bm,external_holdout=hm,baseline_external_holdout=hbm,validation_gates=gates,new_well_gates=wg,distribution_approved=False,installed_in_application=False,pretrained_encoders_changed=False,language_weights_changed=False,controls_reused_from_previous_development=True)
 source.write_json(out/'result.json',result)
 for name,rr,pred,basepred in [('val',val,vp,bp),('external',held,hp,hb)]:source.write_json(out/(name+'_answers.json'),[dict(record_id=r.get('record_id',r.get('metadata',{}).get('record_id')),well=r.get('well'),expected_code=r['allowed_codes'][0] if 'allowed_codes' in r else r['conversations'][1]['value'],prediction=c,baseline_prediction=b) for r,c,b in zip(rr,pred,basepred)])
 contract.update(type='qwen_dinov2_guarded_fusion',feature_contract_version=policy['version'],head_sha256=result['head_sha256'],acceptance_status=result['status'],training_recipe=RECIPE,selected_profile=result['selected'],policy_sha256=result['policy_sha256'],distribution_approved=False)
 source.write_json(out/'MODEL_CONTRACT.json',contract);print(json.dumps(dict(status=result['status'],selected=result['selected'],eligible_policies=result['eligible_policy_count'],old_oof=winner['old_oof']['correct'],val=vm['correct'],external=hm['exact_six_class']['correct'],failed_gates=[k for k,v in gates.items() if not v]),indent=2),flush=True)


if __name__=='__main__':main()
