"""Select one raw-score margin threshold from training-source OOF, never controls."""
import argparse
import json
from pathlib import Path
import shutil
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import train_unibo_visual_facies as source
from scripts.margin_facies_gate import predictions,key
from scripts.paired_context_facies_head import fit_head
from scripts.dino_facies_head import predict_features
from scripts.gray_facies_head import blend_scores
from scripts.facies_head_v2_features import predict
from scripts.train_core_facies_conservative import make_state
from scripts.partial_facies_ridge import mixed_metrics
from scripts.kernel_facies_head import all_class_gates
from scripts.facies_transfer_evaluation import controls

THRESHOLDS=[0.,.005,.01,.02,.03,.05,.08,.12,.2]


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output.resolve();parent=ROOT.parent/'vlm_core_v1/outputs/core_qwen_paired_context_facies_20261031_v16'
 if out.exists() or not out.is_relative_to(ROOT.parent/'vlm_core_v1/outputs'):raise ValueError('Choose new local output')
 pcfg=json.loads((parent/'run_config.json').read_text(encoding='utf-8'));pfrozen=json.loads((parent/'selection_frozen.json').read_text(encoding='utf-8'));sel={k:pfrozen['winner'][k] for k in ('alpha','dino_weight')};original,ref=Path(pcfg['original']),Path(pcfg['reference']);contract=json.loads((parent/'MODEL_CONTRACT.json').read_text(encoding='utf-8'))
 for f,h in pcfg['dependencies'].items():
  if source.digest(f)!=h:raise ValueError('Parent dependency changed')
 rows,val,held=[source.read_rows(parent/f) for f in ('train.jsonl','val.jsonl','external_holdout.jsonl')];out.mkdir(parents=True)
 for name,rr in (('train',rows),('val',val),('external_holdout',held)):source.write_rows(out/(name+'.jsonl'),rr)
 deps=[Path(__file__),ROOT/'scripts/margin_facies_gate.py',ROOT/'scripts/paired_context_facies_head.py',ROOT/'scripts/dino_facies_head.py',ROOT/'scripts/gray_facies_head.py',ROOT/'scripts/facies_transfer_evaluation.py',ROOT/'scripts/kernel_facies_head.py',ROOT/'scripts/facies_head_v2_features.py',ROOT/'scripts/train_core_facies_conservative.py',ROOT/'scripts/partial_facies_ridge.py',parent/'paired_train.safetensors',parent/'facies_head.safetensors',parent/'run_config.json',parent/'selection_frozen.json',original/'features.safetensors',ref/'facies_head.safetensors']
 cfg=dict(recipe=dict(thresholds=THRESHOLDS,selection='Old source-group OOF only: all-class gates, corrected-2*introduced, worst F1 gain, macroF1, group accuracy, higher threshold',scope='Raw uncalibrated margin gate, no new encoder training; parent head fixed before controls'),parent=str(parent),original=str(original),reference=str(ref),context=pcfg['context'],qwen_base=pcfg['qwen_base'],dino=pcfg['dino'],dino_contract=pcfg['dino_contract'],fixed_parent_selection=sel,dependencies={str(f):source.digest(f) for f in deps},train_count=len(rows),validation_count=len(val),external_count=len(held),source_rights=pcfg['source_rights'],distribution_approved=False,installed_in_application=False,prepared_test_pixels_opened=0)
 source.write_json(out/'run_config.json',cfg);shutil.copy2(__file__,out/'training_script_snapshot.py')
 from scripts.train_qwen_source_pilot import configure_offline_environment,_deny_network
 configure_offline_environment();sys.addaudithook(_deny_network)
 import torch
 from safetensors.torch import load_file,save_file
 torch.set_num_threads(4);torch.set_num_interop_threads(1)
 rgb=load_file(str(original/'features.safetensors'))['train'];df=load_file(str(parent/'paired_train.safetensors'))['paired_train'];mapping=pfrozen['fold_source_groups'];ts=torch.empty((len(rows),6),dtype=torch.float64);cs=torch.empty_like(ts)
 for f in range(3):
  ids=[i for i,r in enumerate(rows) if mapping[r['source_group']]!=f];test=[i for i,r in enumerate(rows) if mapping[r['source_group']]==f];old=[i for i in ids if i<728]
  teacher,_=make_state(rgb[old],[rows[i] for i in old],(6144,2048,114));ts[test]=predict(rgb[test],teacher);cs[test]=blend_scores(ts[test],predict(df[test],fit_head(df[ids],[rows[i] for i in ids],sel['alpha'])),sel['dino_weight'])
 base=['ABCDEF'[i] for i in ts.argmax(1).tolist()];baseline=mixed_metrics(rows[:728],base[:728],source)['exact_six_class'];candidates=[];saved={}
 for threshold in THRESHOLDS:
  pred=predictions(ts,cs,threshold);m=mixed_metrics(rows[:728],pred[:728],source)['exact_six_class'];mm=mixed_metrics(rows,pred,source)
  exact=[(r['allowed_codes'][0],b,c) for r,b,c in zip(rows[:728],base[:728],pred[:728]) if len(r['allowed_codes'])==1]
  item=dict(threshold=threshold,old_oof=m,baseline_old_oof=baseline,all_class_gates=all_class_gates(m,baseline),mean_group_accuracy=mm['mean_source_group_accuracy'],corrected=sum(b!=e and c==e for e,b,c in exact),introduced=sum(b==e and c!=e for e,b,c in exact));candidates.append(item);saved[threshold]=pred
 winner=max(candidates,key=key);threshold=winner['threshold'];oof=saved[threshold];shutil.copy2(parent/'facies_head.safetensors',out/'facies_head.safetensors')
 policy=dict(version='qwen_dino_context_margin_facies_v1',threshold=threshold,head_sha256=source.digest(out/'facies_head.safetensors'));source.write_json(out/'policy.json',policy)
 frozen=dict(stage='before_reading_controls',winner=winner,candidates=candidates,fold_source_groups=mapping,training_ids=[r['record_id'] for r in rows],policy_sha256=source.digest(out/'policy.json'));source.write_json(out/'selection_frozen.json',frozen);save_file({'teacher':ts,'candidate':cs},str(out/'oof_scores.safetensors'));source.write_json(out/'oof_answers.json',[dict(record_id=r['record_id'],prediction=c,allowed_codes=r['allowed_codes'],source_group=r['source_group'],fold=mapping[r['source_group']]) for r,c in zip(rows,oof)])
 state=load_file(str(out/'facies_head.safetensors'));teacher=load_file(str(ref/'facies_head.safetensors'));ev=load_file(str(parent/'paired_evaluation.safetensors'));rcfg=json.loads((ref/'run_config.json').read_text(encoding='utf-8'));vr=load_file(str(Path(rcfg['reference'])/'validation_features.safetensors'))['raw_val'][:,:6144];hr=load_file(str(original/'external_features.safetensors'))['external_holdout'];vt,ht=predict(vr,teacher),predict(hr,teacher);vp=predictions(vt,predict_features(vr,ev['paired_val'],state),threshold);hp=predictions(ht,predict_features(hr,ev['paired_external'],state),threshold);bp=['ABCDEF'[i] for i in vt.argmax(1).tolist()];hb=['ABCDEF'[i] for i in ht.argmax(1).tolist()];evaluation=controls(val,held,vp,bp,hp,hb,winner['all_class_gates'])
 result=dict(status='research_all_gates_passed_requires_independent_wells' if all(evaluation['validation_gates'].values()) else 'not_promoted_all_class_or_new_well_gate_failed',selected=dict(threshold=threshold,**sel),eligible_policy_count=sum(all(c['all_class_gates'].values()) for c in candidates),head_sha256=policy['head_sha256'],policy_sha256=source.digest(out/'policy.json'),selection_frozen_sha256=source.digest(out/'selection_frozen.json'),old_oof=winner['old_oof'],baseline_old_oof=baseline,training_gates=winner['all_class_gates'],oof=mixed_metrics(rows,oof,source),**evaluation,distribution_approved=False,installed_in_application=False,pretrained_encoders_changed=False,language_weights_changed=False,controls_reused_from_previous_development=True)
 source.write_json(out/'result.json',result)
 for name,rr,pred,basepred in [('val',val,vp,bp),('external',held,hp,hb)]:source.write_json(out/(name+'_answers.json'),[dict(record_id=r.get('record_id',r.get('metadata',{}).get('record_id')),well=r.get('well'),expected_code=r['allowed_codes'][0] if 'allowed_codes' in r else r['conversations'][1]['value'],prediction=c,baseline_prediction=b) for r,c,b in zip(rr,pred,basepred)])
 contract.update(type='qwen_dino_context_margin_gate',feature_contract_version=policy['version'],head_sha256=result['head_sha256'],policy_sha256=result['policy_sha256'],acceptance_status=result['status'],training_recipe=cfg['recipe'],selected_profile=result['selected'],distribution_approved=False);source.write_json(out/'MODEL_CONTRACT.json',contract)
 print(json.dumps(dict(status=result['status'],threshold=threshold,old_oof=winner['old_oof']['correct'],val=result['val']['correct'],external=result['external_holdout']['exact_six_class']['correct'],failed_gates=[k for k,v in result['validation_gates'].items() if not v]),indent=2),flush=True)


if __name__=='__main__':main()
