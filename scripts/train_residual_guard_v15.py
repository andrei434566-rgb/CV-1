"""Train-group-only guard selection for the fixed learned v14 fusion head."""
import argparse
import json
from pathlib import Path
import shutil
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import train_unibo_visual_facies as source
from scripts.residual_facies_head import infer
from scripts.guarded_facies_head import guard_sets,guarded_predictions,selection_key
from scripts.facies_head_v2_features import predict
from scripts.train_core_facies_conservative import make_state
from scripts.partial_facies_ridge import mixed_metrics
from scripts.kernel_facies_head import all_class_gates


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output.resolve();parent=ROOT.parent/'vlm_core_v1/outputs/core_qwen_residual_facies_20261031_v14'
 if out.exists() or not out.is_relative_to(ROOT.parent/'vlm_core_v1/outputs'):raise ValueError('Choose a new local output')
 pcfg=json.loads((parent/'run_config.json').read_text(encoding='utf-8'));pr=json.loads((parent/'result.json').read_text(encoding='utf-8'));contract=json.loads((parent/'MODEL_CONTRACT.json').read_text(encoding='utf-8'));prev,ref,original=(Path(pcfg[k]) for k in ('previous','reference','original'));recipe=pcfg['recipe']
 for f,h in pcfg['dependencies'].items():
  if source.digest(f)!=h:raise ValueError('Parent dependency changed')
 rows,val,held=[source.read_rows(parent/f) for f in ('train.jsonl','val.jsonl','external_holdout.jsonl')];out.mkdir(parents=True)
 for name,rr in (('train',rows),('val',val),('external_holdout',held)):source.write_rows(out/(name+'.jsonl'),rr)
 deps=[Path(__file__),ROOT/'scripts/guarded_facies_head.py',ROOT/'scripts/residual_facies_head.py',ROOT/'scripts/facies_head_v2_features.py',ROOT/'scripts/train_core_facies_conservative.py',ROOT/'scripts/kernel_facies_head.py',ROOT/'scripts/partial_facies_ridge.py',parent/'run_config.json',parent/'result.json',parent/'selection_frozen.json',prev/'dino_train.safetensors',original/'features.safetensors',ref/'facies_head.safetensors']
 deps.extend(parent/f'fold{i}/residual.safetensors' for i in range(3));deps.append(parent/'final/residual.safetensors')
 cfg=dict(recipe=dict(policy='All64 teacher-class guard subsets, train-only OOF selection; keeps incorrect teacher decisions too',selection='All-class gates, worst recall/F1 gain, macroF1, source-group accuracy, fewer guards, lexical tie-break',residual_recipe=recipe),parent=str(parent),previous=str(prev),reference=str(ref),original=str(original),qwen_base=pcfg['qwen_base'],dino=pcfg['dino'],dino_contract=pcfg['dino_contract'],dependencies={str(f):source.digest(f) for f in deps},source_rights=pcfg['source_rights'],train_count=len(rows),validation_count=len(val),external_count=len(held),distribution_approved=False,installed_in_application=False,prepared_test_pixels_opened=0)
 source.write_json(out/'run_config.json',cfg);shutil.copy2(__file__,out/'training_script_snapshot.py')
 from scripts.train_qwen_source_pilot import configure_offline_environment,_deny_network
 configure_offline_environment();sys.addaudithook(_deny_network)
 import torch
 from safetensors.torch import load_file,save_file
 torch.set_num_threads(4);torch.set_num_interop_threads(1)
 rgb=load_file(str(original/'features.safetensors'))['train'];df=load_file(str(prev/'dino_train.safetensors'))[recipe['view']];mapping=json.loads((parent/'selection_frozen.json').read_text(encoding='utf-8'))['fold_source_groups'];teacher_scores=torch.empty((len(rows),6),dtype=torch.float64);candidate_scores=torch.empty((len(rows),6),dtype=torch.float32)
 for f in range(3):
  ids=[i for i,r in enumerate(rows) if mapping[r['source_group']]!=f];test=[i for i,r in enumerate(rows) if mapping[r['source_group']]==f];old=[i for i in ids if i<728]
  teacher,_=make_state(rgb[old],[rows[i] for i in old],(6144,2048,114));teacher_scores[test]=predict(rgb[test],teacher);candidate_scores[test]=infer(df[test],teacher_scores[test],load_file(str(parent/f'fold{f}/residual.safetensors')),recipe)
 baseline=mixed_metrics(rows[:728],['ABCDEF'[i] for i in teacher_scores[:728].argmax(1).tolist()],source)['exact_six_class'];candidates=[];predictions={}
 for guards in guard_sets():
  pred=guarded_predictions(teacher_scores,candidate_scores,guards);m=mixed_metrics(rows[:728],pred[:728],source)['exact_six_class'];mm=mixed_metrics(rows,pred,source);item=dict(guard_classes=list(guards),old_oof=m,baseline_old_oof=baseline,all_class_gates=all_class_gates(m,baseline),mean_group_accuracy=mm['mean_source_group_accuracy']);candidates.append(item);predictions[guards]=pred
 winner=max(candidates,key=selection_key);guards=winner['guard_classes'];oof=predictions[tuple(guards)];shutil.copy2(parent/'final/residual.safetensors',out/'residual.safetensors')
 policy=dict(version='qwen_dino_residual_class_guard_v1',guard_classes=guards,recipe=recipe,head_sha256=source.digest(out/'residual.safetensors'));source.write_json(out/'policy.json',policy)
 frozen=dict(stage='before_reading_controls',winner=winner,candidates=candidates,fold_source_groups=mapping,training_ids=[r['record_id'] for r in rows],policy_sha256=source.digest(out/'policy.json'));source.write_json(out/'selection_frozen.json',frozen);save_file({'teacher':teacher_scores,'residual':candidate_scores},str(out/'oof_scores.safetensors'));source.write_json(out/'oof_answers.json',[dict(record_id=r['record_id'],prediction=c,allowed_codes=r['allowed_codes'],source_group=r['source_group'],fold=mapping[r['source_group']]) for r,c in zip(rows,oof)])
 teacher=load_file(str(ref/'facies_head.safetensors'));state=load_file(str(out/'residual.safetensors'));ev=load_file(str(prev/'dino_evaluation.safetensors'));rcfg=json.loads((ref/'run_config.json').read_text(encoding='utf-8'));vr=load_file(str(Path(rcfg['reference'])/'validation_features.safetensors'))['raw_val'][:,:6144];hr=load_file(str(original/'external_features.safetensors'))['external_holdout'];vt,ht=predict(vr,teacher),predict(hr,teacher)
 vp=guarded_predictions(vt,infer(ev['val__'+recipe['view']],vt,state,recipe),guards);hp=guarded_predictions(ht,infer(ev['external__'+recipe['view']],ht,state,recipe),guards);bp=['ABCDEF'[i] for i in vt.argmax(1).tolist()];hb=['ABCDEF'[i] for i in ht.argmax(1).tolist()];expected=[r['conversations'][1]['value'] for r in val];vm,bm=source.classification_metrics(expected,vp),source.classification_metrics(expected,bp);hm,hbm=mixed_metrics(held,hp,source),mixed_metrics(held,hb,source)
 gates=all_class_gates(vm,bm);gates.update(train_eligible=all(winner['all_class_gates'].values()),accuracy=vm['correct']>=bm['correct'],no_previously_correct_case_lost=not any(b==e and c!=e for e,b,c in zip(expected,bp,vp)));wg={}
 for well in sorted({r['well'] for r in held}):
  for c in sorted({r['allowed_codes'][0] for r in held if r['well']==well}):wg[well+'_'+c]=sum(p==c for r,p in zip(held,hp) if r['well']==well and r['allowed_codes']==[c])>=sum(p==c for r,p in zip(held,hb) if r['well']==well and r['allowed_codes']==[c])
 wg['at_least_one_gain']=hm['exact_six_class']['correct']>hbm['exact_six_class']['correct'];gates['new_wells_eligible']=all(wg.values())
 result=dict(status='research_all_gates_passed_requires_independent_wells' if all(gates.values()) else 'not_promoted_all_class_or_new_well_gate_failed',selected=dict(guard_classes=guards),eligible_policy_count=sum(all(c['all_class_gates'].values()) for c in candidates),head_sha256=policy['head_sha256'],policy_sha256=source.digest(out/'policy.json'),selection_frozen_sha256=source.digest(out/'selection_frozen.json'),old_oof=winner['old_oof'],baseline_old_oof=baseline,training_gates=winner['all_class_gates'],oof=mixed_metrics(rows,oof,source),val=vm,baseline_val=bm,external_holdout=hm,baseline_external_holdout=hbm,validation_gates=gates,new_well_gates=wg,distribution_approved=False,installed_in_application=False,pretrained_encoders_changed=False,language_weights_changed=False,controls_reused_from_previous_development=True)
 source.write_json(out/'result.json',result)
 for name,rr,pred,basepred in [('val',val,vp,bp),('external',held,hp,hb)]:source.write_json(out/(name+'_answers.json'),[dict(record_id=r.get('record_id',r.get('metadata',{}).get('record_id')),well=r.get('well'),expected_code=r['allowed_codes'][0] if 'allowed_codes' in r else r['conversations'][1]['value'],prediction=c,baseline_prediction=b) for r,c,b in zip(rr,pred,basepred)])
 contract.update(type='qwen_dino_bounded_residual_class_guard',feature_contract_version=policy['version'],head_sha256=result['head_sha256'],acceptance_status=result['status'],training_recipe=cfg['recipe'],selected_profile=result['selected'],policy_sha256=result['policy_sha256'],distribution_approved=False)
 source.write_json(out/'MODEL_CONTRACT.json',contract);print(json.dumps(dict(status=result['status'],guard_classes=guards,eligible=result['eligible_policy_count'],old_oof=winner['old_oof']['correct'],val=vm['correct'],external=hm['exact_six_class']['correct'],failed_gates=[k for k,v in gates.items() if not v]),indent=2),flush=True)


if __name__=='__main__':main()
