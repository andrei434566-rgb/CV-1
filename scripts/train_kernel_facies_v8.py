"""All-class train-group selection for a nonlinear head on frozen Qwen vision.

No validation-based selection and no language/encoder update. This is a local
research experiment; mixed source distribution permissions remain unresolved.
"""
import argparse
import json
from pathlib import Path
import shutil
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import train_unibo_visual_facies as source
from scripts.facies_head_v2_features import prepare_state,normalize,predict
from scripts.partial_facies_ridge import mixed_metrics
from scripts.kernel_facies_head import kernel,training_bandwidth,partial_dual,grouped_weights,all_class_gates,selection_key,predict_features,validate_state

RECIPE=dict(profile='qwen_post',factors=[.5,2.,8.],alphas=[.01,.1,1.],weightings=['class','class_group'],external_weight=.5,
 folds=3,selection='On same old728 OOF: all six recall AND F1 nondecreasing, macroF1 higher; if none pass, maximum worst-class recall gain then minimum recall, macroF1, mean group accuracy. No val selection.',
 normalization='All train rows in each fold only, population std floor1e-3, L2; RBF bandwidth=median positive squared training distance',
 partial_AB='Only sum A+B is supervised for generic floodplain; no guessed drainage',
 validation='Previous50 loaded once after train-group selection frozen',seed=20261031,
 distribution_approved=False,installed_in_application=False,encoder_or_chat_weights_changed=False)


def code_list(scores):return [source.CODES[i] for i in scores.argmax(1).tolist()]


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--previous',type=Path,default=ROOT.parent/'vlm_core_v1/outputs/core_qwen_core_contrasts_20261031_v7');p.add_argument('--output',type=Path,required=True);a=p.parse_args();prev,out=a.previous.resolve(),a.output.resolve()
 if out.exists() or not out.is_relative_to(ROOT.parent/'vlm_core_v1/outputs'):raise ValueError('Choose a new local output')
 oldcfg=json.loads((prev/'run_config.json').read_text(encoding='utf-8'));ref=Path(oldcfg['reference']);contract=json.loads((prev/'MODEL_CONTRACT.json').read_text(encoding='utf-8'))
 rows=source.read_rows(prev/'train.jsonl');val=source.read_rows(prev/'val.jsonl')
 if len(rows)!=804 or len(val)!=50 or val!=source.read_rows(ref/'val.jsonl'):raise ValueError('Reference manifest differs')
 for f,h in oldcfg['dependencies'].items():
  if source.digest(f)!=h:raise ValueError('Previous dependency differs: '+f)
 for r in rows:
  if source.digest(r['image'])!=r['image_sha256']:raise ValueError('Training image changed')
 out.mkdir(parents=True);source.write_rows(out/'train.jsonl',rows);source.write_rows(out/'val.jsonl',val)
 deps=[Path(__file__),ROOT/'scripts/kernel_facies_head.py',ROOT/'scripts/facies_head_v2_features.py',ROOT/'scripts/partial_facies_ridge.py',prev/'train.jsonl',prev/'val.jsonl',prev/'features.safetensors',prev/'selection_frozen.json',prev/'audit_result.json',ref/'oof_answers.json',ref/'facies_head.safetensors']
 cfg=dict(recipe=RECIPE,previous=str(prev),reference=str(ref),base=oldcfg['base'],train_count=len(rows),val_count=len(val),dependencies={str(f):source.digest(f) for f in deps},
 source_rights=oldcfg['source_rights'],distribution_approved=False,installed_in_application=False,prepared_test_pixels_opened=0)
 source.write_json(out/'run_config.json',cfg);shutil.copy2(__file__,out/'training_script_snapshot.py')
 from scripts.train_qwen_source_pilot import configure_offline_environment,_deny_network
 configure_offline_environment();sys.addaudithook(_deny_network)
 import torch
 from safetensors.torch import load_file,save_file
 torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.manual_seed(RECIPE['seed'])
 raw=load_file(str(prev/'features.safetensors'))['train']
 if raw.shape!=(804,6144):raise ValueError('Feature cache dimensions differ')
 mapping=json.loads((prev/'selection_frozen.json').read_text(encoding='utf-8'))['fold_source_groups']
 folds=[];dims=(6144,2048,114)
 for f in range(3):
  mask=torch.tensor([mapping[r['source_group']]!=f for r in rows]);held=~mask
  subset=[r for r,k in zip(rows,mask.tolist()) if k]
  if {r['source_group'] for r in subset}&{r['source_group'] for r,k in zip(rows,held.tolist()) if k}:raise ValueError('Group leakage')
  s=prepare_state(raw[mask],'qwen_post',*dims);x=normalize(raw[mask],s)[:,:-1];xh=normalize(raw[held],s)[:,:-1];band=training_bandwidth(x)
  folds.append(dict(mask=mask,held=held,rows=subset,x=x,xh=xh,band=band))
 baseline_answers=json.loads((ref/'oof_answers.json').read_text(encoding='utf-8'))
 baseline=mixed_metrics(rows[:728],[r['prediction'] for r in baseline_answers],source)['exact_six_class']
 def log(event,**kw):
  data=dict(event=event,**kw)
  with (out/'progress.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(data)+'\n')
  print(json.dumps(data),flush=True)
 candidates=[];saved_predictions={}
 for factor in RECIPE['factors']:
  kernels=[(kernel(f['x'],f['x'],f['band'],factor),kernel(f['xh'],f['x'],f['band'],factor)) for f in folds]
  for alpha in RECIPE['alphas']:
   for mode in RECIPE['weightings']:
    oof=['']*len(rows)
    for fold,(gram,heldgram) in zip(folds,kernels):
     weights=grouped_weights(fold['rows'],mode,RECIPE['external_weight'])
     dual=partial_dual(gram,[r['allowed_codes'] for r in fold['rows']],weights,alpha)
     for i,c in zip(torch.where(fold['held'])[0].tolist(),code_list(heldgram@dual)):oof[i]=c
    m=mixed_metrics(rows[:728],oof[:728],source)['exact_six_class'];allm=mixed_metrics(rows,oof,source)
    item=dict(factor=factor,alpha=alpha,weighting=mode,old_oof=m,baseline_old_oof=baseline,all_class_gates=all_class_gates(m,baseline),mean_group_accuracy=allm['mean_source_group_accuracy'],all_oof=allm)
    candidates.append(item);saved_predictions[factor,alpha,mode]=oof;log('candidate',factor=factor,alpha=alpha,weighting=mode,old_correct=m['correct'],macro_f1=m['macro_f1'],eligible=all(item['all_class_gates'].values()),recall=m['per_class_recall'])
 winner=max(candidates,key=selection_key);factor,alpha,mode=(winner[k] for k in ('factor','alpha','weighting'));oof=saved_predictions[factor,alpha,mode]
 frozen=dict(stage='before_validation_features_and_predictions',winner=winner,candidates=candidates,fold_source_groups=mapping,training_record_ids=[r['record_id'] for r in rows],
  selection_status='eligible_research_update' if all(winner['all_class_gates'].values()) else 'no_all_class_eligible_candidate_diagnostic_only')
 source.write_json(out/'selection_frozen.json',frozen);source.write_json(out/'oof_answers.json',[dict(record_id=r['record_id'],source_group=r['source_group'],allowed_codes=r['allowed_codes'],prediction=c,fold=mapping[r['source_group']]) for r,c in zip(rows,oof)])
 s=prepare_state(raw,'qwen_post',*dims);x=normalize(raw,s)[:,:-1];band=training_bandwidth(x);gram=kernel(x,x,band,factor);weights=grouped_weights(rows,mode,RECIPE['external_weight'])
 s.update(anchors=x,bandwidth=band,factor=torch.tensor(factor,dtype=torch.float64),dual=partial_dual(gram,[r['allowed_codes'] for r in rows],weights,alpha))
 save_file({k:v.contiguous() for k,v in s.items()},str(out/'facies_head.safetensors'))
 result=dict(head_sha256=source.digest(out/'facies_head.safetensors'),selection_frozen_sha256=source.digest(out/'selection_frozen.json'),selected={k:winner[k] for k in ('factor','alpha','weighting')},old_oof=winner['old_oof'],baseline_old_oof=baseline,oof=winner['all_oof'],train=mixed_metrics(rows,code_list(predict_features(raw,s)),source),distribution_approved=False,installed_in_application=False,encoder_or_chat_weights_changed=False)
 # Validation is deliberately first loaded after frozen selection, never used to pick a kernel.
 refcfg=json.loads((ref/'run_config.json').read_text(encoding='utf-8'));vraw=load_file(str(Path(refcfg['reference'])/'validation_features.safetensors'))['raw_val'][:,:6144]
 expected=[r['conversations'][1]['value'] for r in val];vp=code_list(predict_features(vraw,s));bp=code_list(predict(vraw,load_file(str(ref/'facies_head.safetensors'))))
 vm=source.classification_metrics(expected,vp);bm=source.classification_metrics(expected,bp);gates=all_class_gates(vm,bm);gates['train_eligible']=all(winner['all_class_gates'].values());gates['accuracy']=vm['correct']>=bm['correct']
 result.update(val=vm,baseline_val=bm,validation_gates=gates,paired=dict(fixed=sum(b!=e and c==e for e,b,c in zip(expected,bp,vp)),introduced=sum(b==e and c!=e for e,b,c in zip(expected,bp,vp))))
 result['status']='research_all_class_gates_passed_requires_independent_validation' if all(gates.values()) else 'not_promoted_all_class_gate_failed'
 source.write_json(out/'result.json',result);source.write_json(out/'val_answers.json',[dict(record_id=r['metadata']['record_id'],expected_code=e,prediction=c,baseline_prediction=b) for r,e,c,b in zip(val,expected,vp,bp)])
 contract.update(type='frozen_qwen_rbf_partial_ridge_head',feature_contract_version='qwen3vl_rbf_partial_head_v1',feature_dimension=4096,anchor_count=len(rows),head_sha256=result['head_sha256'],acceptance_status=result['status'],training_recipe=RECIPE,selected_profile=dict(profile='qwen_post',**result['selected']),distribution_approved=False,
  feature_spec=dict(encoder='Original frozen Qwen3-VL-2B',input='RGB core crop thumbnail300',normalization='Train-only Qwen post features, L2; RBF kernel to804 train anchors',output='Six uncalibrated class scores; partial A/B sum supervision'))
 validate_state(s,contract);source.write_json(out/'MODEL_CONTRACT.json',contract);log('complete',selected=result['selected'],val=vm,gates=gates,status=result['status'])


if __name__=='__main__':main()
