"""One fixed bounded neural-fusion recipe, three grouped folds and final fit."""
import argparse
import json
from pathlib import Path
import shutil
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import train_unibo_visual_facies as source
from scripts.residual_facies_head import fit,infer
from scripts.facies_head_v2_features import predict
from scripts.train_core_facies_conservative import make_state
from scripts.partial_facies_ridge import mixed_metrics
from scripts.kernel_facies_head import all_class_gates

RECIPE=dict(view='letterbox',hidden=64,epochs=50,batch_size=64,lr=.001,weight_decay=.02,cap=.15,logit_scale=8.,teacher_KL=1.,seed=20261031,
 fit='Class+source-group weighted exact/partial CE + KL on correctly classified old exact training rows only; teacher labels never replace true source labels',
 selection='One fixed recipe, final epoch only; no tuning against val50/external16; selected DINO view inherited from train-only v12',
 features='Train-only standardized and block-L2 DINO1152 + re-fitted train-only Qwen teacher6 scores; zero-output initial residual, tanh cap0.15',
 pretrained_encoders_changed=False,language_weights_changed=False,distribution_approved=False,installed_in_application=False)


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output.resolve();prev=ROOT.parent/'vlm_core_v1/outputs/core_qwen_dino_facies_20261031_v12'
 if out.exists() or not out.is_relative_to(ROOT.parent/'vlm_core_v1/outputs'):raise ValueError('Choose new local output')
 pcfg=json.loads((prev/'run_config.json').read_text(encoding='utf-8'));ref,original=Path(pcfg['reference']),Path(pcfg['previous']);contract=json.loads((prev/'MODEL_CONTRACT.json').read_text(encoding='utf-8'))
 for f,h in pcfg['dependencies'].items():
  if source.digest(f)!=h:raise ValueError('Previous dependency changed')
 if source.package_versions()!=contract['package_versions']:raise ValueError('Runtime changed')
 rows,val,held=[source.read_rows(prev/f) for f in ('train.jsonl','val.jsonl','external_holdout.jsonl')];out.mkdir(parents=True)
 for name,rr in (('train',rows),('val',val),('external_holdout',held)):source.write_rows(out/(name+'.jsonl'),rr)
 deps=[Path(__file__),ROOT/'scripts/residual_facies_head.py',ROOT/'scripts/kernel_facies_head.py',ROOT/'scripts/vision_facies_adapter.py',ROOT/'scripts/facies_head_v2_features.py',ROOT/'scripts/partial_facies_ridge.py',ROOT/'scripts/train_core_facies_conservative.py',prev/'dino_train.safetensors',prev/'selection_frozen.json',prev/'run_config.json',original/'features.safetensors',ref/'facies_head.safetensors',ref/'oof_answers.json']
 cfg=dict(recipe=RECIPE,previous=str(prev),reference=str(ref),original=str(original),qwen_base=pcfg['qwen_base'],dino=pcfg['dino'],dino_contract=pcfg['dino_contract'],train_count=len(rows),validation_count=len(val),external_count=len(held),dependencies={str(f):source.digest(f) for f in deps},source_rights=pcfg['source_rights'],distribution_approved=False,installed_in_application=False,prepared_test_pixels_opened=0)
 source.write_json(out/'run_config.json',cfg);shutil.copy2(__file__,out/'training_script_snapshot.py')
 from scripts.train_qwen_source_pilot import configure_offline_environment,_deny_network
 configure_offline_environment();sys.addaudithook(_deny_network)
 import torch
 from safetensors.torch import load_file,save_file
 torch.set_num_threads(4);torch.set_num_interop_threads(1)
 if not torch.cuda.is_available() or torch.cuda.mem_get_info()[0]<512*1024**2:raise RuntimeError('Need512MiB free CUDA')
 rgb=load_file(str(original/'features.safetensors'))['train'];df=load_file(str(prev/'dino_train.safetensors'))[RECIPE['view']];mapping=json.loads((prev/'selection_frozen.json').read_text(encoding='utf-8'))['fold_source_groups'];oof=['']*len(rows);receipts=[]
 def codes(x):return ['ABCDEF'[i] for i in x.argmax(1).tolist()]
 for f in range(3):
  ids=[i for i,r in enumerate(rows) if mapping[r['source_group']]!=f];test=[i for i,r in enumerate(rows) if mapping[r['source_group']]==f];old=[i for i in ids if i<728]
  if {rows[i]['source_group'] for i in ids}&{rows[i]['source_group'] for i in test}:raise ValueError('Group leakage')
  teacher,_=make_state(rgb[old],[rows[i] for i in old],(6144,2048,114));ts=predict(rgb[ids],teacher);state,receipt=fit(df[ids],ts,[rows[i] for i in ids],[i<728 for i in ids],RECIPE['seed']+f,RECIPE)
  folder=out/('fold'+str(f));folder.mkdir();save_file(state,str(folder/'residual.safetensors'));receipt.update(tag='fold'+str(f),training_ids=[rows[i]['record_id'] for i in ids],held_ids=[rows[i]['record_id'] for i in test],head_sha256=source.digest(folder/'residual.safetensors'));source.write_json(folder/'receipt.json',receipt);receipts.append(receipt)
  for i,c in zip(test,codes(infer(df[test],predict(rgb[test],teacher),load_file(str(folder/'residual.safetensors')),RECIPE))):oof[i]=c
  print(json.dumps(dict(event='fold_complete',fold=f,loss_first=receipt['epoch_mean_losses'][0],loss_last=receipt['epoch_mean_losses'][-1])),flush=True)
 baseline=mixed_metrics(rows[:728],[r['prediction'] for r in json.loads((ref/'oof_answers.json').read_text(encoding='utf-8'))],source)['exact_six_class'];oldm=mixed_metrics(rows[:728],oof[:728],source)['exact_six_class'];tg=all_class_gates(oldm,baseline)
 frozen=dict(stage='before_control_features_or_predictions',recipe=RECIPE,old_oof=oldm,training_gates=tg,fold_source_groups=mapping,training_ids=[r['record_id'] for r in rows]);source.write_json(out/'selection_frozen.json',frozen);source.write_json(out/'oof_answers.json',[dict(record_id=r['record_id'],prediction=c,allowed_codes=r['allowed_codes'],source_group=r['source_group'],fold=mapping[r['source_group']]) for r,c in zip(rows,oof)])
 teacher=load_file(str(ref/'facies_head.safetensors'));ts=predict(rgb,teacher);state,receipt=fit(df,ts,rows,[i<728 for i in range(len(rows))],RECIPE['seed']+3,RECIPE);folder=out/'final';folder.mkdir();save_file(state,str(folder/'residual.safetensors'));receipt.update(tag='final',training_ids=[r['record_id'] for r in rows],held_ids=[],head_sha256=source.digest(folder/'residual.safetensors'));source.write_json(folder/'receipt.json',receipt);receipts.append(receipt)
 evaluation=load_file(str(prev/'dino_evaluation.safetensors'));rcfg=json.loads((ref/'run_config.json').read_text(encoding='utf-8'));vr=load_file(str(Path(rcfg['reference'])/'validation_features.safetensors'))['raw_val'][:,:6144];hr=load_file(str(original/'external_features.safetensors'))['external_holdout'];vp=codes(infer(evaluation['val__'+RECIPE['view']],predict(vr,teacher),state,RECIPE));hp=codes(infer(evaluation['external__'+RECIPE['view']],predict(hr,teacher),state,RECIPE));bp,hb=codes(predict(vr,teacher)),codes(predict(hr,teacher));expected=[r['conversations'][1]['value'] for r in val];vm,bm=source.classification_metrics(expected,vp),source.classification_metrics(expected,bp);hm,hbm=mixed_metrics(held,hp,source),mixed_metrics(held,hb,source)
 gates=all_class_gates(vm,bm);gates.update(train_eligible=all(tg.values()),accuracy=vm['correct']>=bm['correct'],no_previously_correct_case_lost=not any(b==e and c!=e for e,b,c in zip(expected,bp,vp)))
 wg={}
 for well in sorted({r['well'] for r in held}):
  for c in sorted({r['allowed_codes'][0] for r in held if r['well']==well}):wg[well+'_'+c]=sum(p==c for r,p in zip(held,hp) if r['well']==well and r['allowed_codes']==[c])>=sum(p==c for r,p in zip(held,hb) if r['well']==well and r['allowed_codes']==[c])
 wg['at_least_one_gain']=hm['exact_six_class']['correct']>hbm['exact_six_class']['correct'];gates['new_wells_eligible']=all(wg.values())
 result=dict(status='research_all_gates_passed_requires_independent_wells' if all(gates.values()) else 'not_promoted_all_class_or_new_well_gate_failed',head_sha256=source.digest(folder/'residual.safetensors'),selection_frozen_sha256=source.digest(out/'selection_frozen.json'),old_oof=oldm,baseline_old_oof=baseline,training_gates=tg,oof=mixed_metrics(rows,oof,source),train=mixed_metrics(rows,codes(infer(df,ts,state,RECIPE)),source),val=vm,baseline_val=bm,external_holdout=hm,baseline_external_holdout=hbm,validation_gates=gates,new_well_gates=wg,training_receipts=receipts,distribution_approved=False,installed_in_application=False,pretrained_encoders_changed=False,language_weights_changed=False,controls_reused_from_previous_development=True)
 source.write_json(out/'result.json',result)
 for name,rr,pred,basepred in [('val',val,vp,bp),('external',held,hp,hb)]:source.write_json(out/(name+'_answers.json'),[dict(record_id=r.get('record_id',r.get('metadata',{}).get('record_id')),well=r.get('well'),expected_code=r['allowed_codes'][0] if 'allowed_codes' in r else r['conversations'][1]['value'],prediction=c,baseline_prediction=b) for r,c,b in zip(rr,pred,basepred)])
 contract.update(type='qwen_dinov2_bounded_neural_fusion',feature_contract_version='qwen_dino_residual_facies_v1',head_sha256=result['head_sha256'],acceptance_status=result['status'],training_recipe=RECIPE,distribution_approved=False)
 source.write_json(out/'MODEL_CONTRACT.json',contract);print(json.dumps(dict(status=result['status'],old_oof=oldm['correct'],val=vm['correct'],external=hm['exact_six_class']['correct'],failed_gates=[k for k,v in gates.items() if not v]),indent=2),flush=True)


if __name__=='__main__':main()
