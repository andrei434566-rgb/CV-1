"""Replay every train-only margin candidate and both previously used controls."""
import argparse
import json
from pathlib import Path
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


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);a=p.parse_args();run=a.run.resolve()
 def read(p):return json.loads(p.read_text(encoding='utf-8'))
 cfg,result,frozen,policy=[read(run/f) for f in ('run_config.json','result.json','selection_frozen.json','policy.json')]
 parent,original,ref=(Path(cfg[k]) for k in ('parent','original','reference'))
 if not (parent/'audit_result.json').is_file():raise ValueError('Parent context features must be audited first')
 for f,h in cfg['dependencies'].items():
  if source.digest(f)!=h:raise ValueError('Dependency changed')
 if source.digest(run/'policy.json')!=result['policy_sha256'] or source.digest(run/'facies_head.safetensors')!=result['head_sha256'] or source.digest(parent/'facies_head.safetensors')!=result['head_sha256']:raise ValueError('Policy or parent head differs')
 from scripts.train_qwen_source_pilot import configure_offline_environment,_deny_network
 configure_offline_environment();sys.addaudithook(_deny_network)
 import torch
 from safetensors.torch import load_file
 torch.set_num_threads(4);torch.set_num_interop_threads(1)
 rows,val,held=[source.read_rows(run/f) for f in ('train.jsonl','val.jsonl','external_holdout.jsonl')]
 for rr,name in ((rows,'train.jsonl'),(val,'val.jsonl'),(held,'external_holdout.jsonl')):
  if rr!=source.read_rows(parent/name):raise ValueError('Rows changed')
 rgb=load_file(str(original/'features.safetensors'))['train'];df=load_file(str(parent/'paired_train.safetensors'))['paired_train'];mapping=frozen['fold_source_groups'];sel=cfg['fixed_parent_selection'];ts=torch.empty((len(rows),6),dtype=torch.float64);cs=torch.empty_like(ts)
 for f in range(3):
  ids=[i for i,r in enumerate(rows) if mapping[r['source_group']]!=f];test=[i for i,r in enumerate(rows) if mapping[r['source_group']]==f];old=[i for i in ids if i<728]
  if {rows[i]['source_group'] for i in ids}&{rows[i]['source_group'] for i in test}:raise ValueError('Group leakage')
  teacher,_=make_state(rgb[old],[rows[i] for i in old],(6144,2048,114));ts[test]=predict(rgb[test],teacher);cs[test]=blend_scores(ts[test],predict(df[test],fit_head(df[ids],[rows[i] for i in ids],sel['alpha'])),sel['dino_weight'])
 scores=load_file(str(run/'oof_scores.safetensors'))
 if not torch.equal(ts,scores['teacher']) or not torch.equal(cs,scores['candidate']):raise ValueError('Cross-fitted scores differ')
 base=['ABCDEF'[i] for i in ts.argmax(1).tolist()];baseline=mixed_metrics(rows[:728],base[:728],source)['exact_six_class'];candidates=[];saved={}
 for threshold in cfg['recipe']['thresholds']:
  pred=predictions(ts,cs,threshold);m=mixed_metrics(rows[:728],pred[:728],source)['exact_six_class'];mm=mixed_metrics(rows,pred,source);exact=[(r['allowed_codes'][0],b,c) for r,b,c in zip(rows[:728],base[:728],pred[:728]) if len(r['allowed_codes'])==1]
  item=dict(threshold=threshold,old_oof=m,baseline_old_oof=baseline,all_class_gates=all_class_gates(m,baseline),mean_group_accuracy=mm['mean_source_group_accuracy'],corrected=sum(b!=e and c==e for e,b,c in exact),introduced=sum(b==e and c!=e for e,b,c in exact));candidates.append(item);saved[threshold]=pred
 if candidates!=frozen['candidates'] or max(candidates,key=key)!=frozen['winner']:raise ValueError('Train-only gate selection differs')
 winner=frozen['winner'];threshold=winner['threshold'];oof=saved[threshold]
 if policy['threshold']!=threshold or oof!=[r['prediction'] for r in read(run/'oof_answers.json')]:raise ValueError('Chosen policy differs')
 teacher=load_file(str(ref/'facies_head.safetensors'));state=load_file(str(run/'facies_head.safetensors'));ev=load_file(str(parent/'paired_evaluation.safetensors'));rcfg=read(ref/'run_config.json');vr=load_file(str(Path(rcfg['reference'])/'validation_features.safetensors'))['raw_val'][:,:6144];hr=load_file(str(original/'external_features.safetensors'))['external_holdout'];vt,ht=predict(vr,teacher),predict(hr,teacher);vp=predictions(vt,predict_features(vr,ev['paired_val'],state),threshold);hp=predictions(ht,predict_features(hr,ev['paired_external'],state),threshold);bp=['ABCDEF'[i] for i in vt.argmax(1).tolist()];hb=['ABCDEF'[i] for i in ht.argmax(1).tolist()];evaluation=controls(val,held,vp,bp,hp,hb,winner['all_class_gates'])
 for k,v in evaluation.items():
  if v!=result[k]:raise ValueError('Control metric differs: '+k)
 for name,pred in (('val',vp),('external',hp)):
  if pred!=[r['prediction'] for r in read(run/(name+'_answers.json'))]:raise ValueError('Control prediction differs')
 if mixed_metrics(rows,oof,source)!=result['oof'] or winner['old_oof']!=result['old_oof'] or source.digest(run/'selection_frozen.json')!=result['selection_frozen_sha256']:raise ValueError('OOF/frozen state differs')
 audit=dict(status=result['status'],head_sha256=result['head_sha256'],parent_context_feature_audit_verified=True,all9_train_only_candidates_reproduced=True,grouped_and_control_predictions_reproduced=True,validation_gates=evaluation['validation_gates'],new_well_gates=evaluation['new_well_gates'],selected_threshold=threshold,threshold_zero_leaves_parent_unchanged=threshold==0.,distribution_approved=False,installed_in_application=False,prepared_test_pixels_opened=0,controls_reused_from_prior_development=True,audit_script_sha256=source.digest(__file__))
 source.write_json(run/'audit_result.json',audit);print(json.dumps(audit,indent=2),flush=True)


if __name__=='__main__':main()
