"""Compare frozen DINOv2 texture heads and Qwen/DINO score fusion, train OOF only."""
import argparse
import json
from pathlib import Path
import shutil
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import train_unibo_visual_facies as source
from scripts.dino_facies_head import extract_features,fit_head,candidate_key,pack_state,predict_features
from scripts.gray_facies_head import blend_scores
from scripts.facies_head_v2_features import predict
from scripts.train_core_facies_conservative import make_state
from scripts.partial_facies_ridge import mixed_metrics
from scripts.kernel_facies_head import all_class_gates

RECIPE=dict(views=['standard','letterbox'],alphas=[.1,1.,10.,100.],dino_weights=[.25,.5,1.],folds=3,
 features='DINOv2-small CLS384 + patch mean384/std384; three separately normalized blocks',
 standard='Official HF processor resize-shortest256 / center224',letterbox='Entire RGB crop resized to fit224 keeping aspect ratio, mean-colour padding; patch pooling uses centers inside image only',
 training='Class+source-group balanced partial ridge on train only; external0.5,endface0.25; generic floodplain supervises A+B only',
 rgb='Exact original Qwen v3 teacher; re-fit old728 train subset per group fold',
 selection='24 candidates on old OOF: all-six recall/F1 preservation and macro gain, worst recall/F1 gain, mean source-group accuracy, macroF1, smaller DINO weight, larger alpha, letterbox tie-break',
 controls='Reused val50 and external16 are development controls; first used only after OOF and selected head frozen',
 pretrained_encoders_changed=False,language_weights_changed=False,distribution_approved=False,installed_in_application=False)


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--dino',type=Path,default=ROOT.parent/'vlm_core_v1/model_cache/local_dinov2_small/ed25f3a31f01632728cabb09d1542f84ab7b0056');a=p.parse_args();out,dino=a.output.resolve(),a.dino.resolve();prev=ROOT.parent/'vlm_core_v1/outputs/core_qwen_new_well_facies_20261031_v10'
 if out.exists() or not out.is_relative_to(ROOT.parent/'vlm_core_v1/outputs'):raise ValueError('Choose a new local output')
 pcfg=json.loads((prev/'run_config.json').read_text(encoding='utf-8'));ref=Path(pcfg['reference']);contract=json.loads((prev/'MODEL_CONTRACT.json').read_text(encoding='utf-8'));dc=json.loads((dino/'download_receipt.json').read_text(encoding='utf-8'))
 if dc['model_id']!='facebook/dinov2-small' or dc['revision']!='ed25f3a31f01632728cabb09d1542f84ab7b0056' or dc['license']!='Apache-2.0':raise ValueError('DINO provenance changed')
 for name,h in dc['sha256'].items():
  if source.digest(dino/name)!=h:raise ValueError('DINO source changed')
 for f,h in pcfg['dependencies'].items():
  if source.digest(f)!=h:raise ValueError('Previous dependency changed')
 if source.package_versions()!=contract['package_versions']:raise ValueError('Runtime changed')
 rows,val,held=[source.read_rows(prev/f) for f in ('train.jsonl','val.jsonl','external_holdout.jsonl')]
 if len(rows)!=822 or len(val)!=50 or len(held)!=16 or not (prev/'audit_result.json').is_file():raise ValueError('Audited data required')
 out.mkdir(parents=True)
 for name,rr in (('train',rows),('val',val),('external_holdout',held)):source.write_rows(out/(name+'.jsonl'),rr)
 deps=[Path(__file__),ROOT/'scripts/dino_facies_head.py',ROOT/'scripts/gray_facies_head.py',ROOT/'scripts/kernel_facies_head.py',ROOT/'scripts/partial_facies_ridge.py',ROOT/'scripts/facies_head_v2_features.py',ROOT/'scripts/train_core_facies_conservative.py',prev/'features.safetensors',prev/'external_features.safetensors',prev/'train.jsonl',prev/'val.jsonl',prev/'external_holdout.jsonl',prev/'selection_frozen.json',prev/'audit_result.json',ref/'facies_head.safetensors',ref/'oof_answers.json',dino/'download_receipt.json']
 cfg=dict(recipe=RECIPE,previous=str(prev),reference=str(ref),dino=str(dino),dino_contract=dc,qwen_base=pcfg['base'],train_count=822,validation_count=50,external_count=16,
  dependencies={str(f):source.digest(f) for f in deps},source_rights=pcfg['source_rights'],distribution_approved=False,installed_in_application=False,prepared_test_pixels_opened=0)
 source.write_json(out/'run_config.json',cfg);shutil.copy2(__file__,out/'training_script_snapshot.py')
 from scripts.train_qwen_source_pilot import configure_offline_environment,_deny_network
 configure_offline_environment();sys.addaudithook(_deny_network)
 import torch
 from safetensors.torch import load_file,save_file
 torch.set_num_threads(4);torch.set_num_interop_threads(1)
 if not torch.cuda.is_available() or torch.cuda.mem_get_info()[0]<1024**3:raise RuntimeError('Need1GiB free CUDA')
 def log(event,**data):
  v=dict(event=event,**data)
  with (out/'progress.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(v)+'\n')
  if event!='dino_features' or data['done']%80==0 or data['done']==data['total']:print(json.dumps(v),flush=True)
 features=extract_features(rows,dino,log);save_file(features,str(out/'dino_train.safetensors'));rgb=load_file(str(prev/'features.safetensors'))['train'];mapping=json.loads((prev/'selection_frozen.json').read_text(encoding='utf-8'))['fold_source_groups']
 baseline=mixed_metrics(rows[:728],[r['prediction'] for r in json.loads((ref/'oof_answers.json').read_text(encoding='utf-8'))],source)['exact_six_class'];fold_data=[];teacher_oof=['']*728
 for f in range(3):
  ids=[i for i,r in enumerate(rows) if mapping[r['source_group']]!=f];test=[i for i,r in enumerate(rows) if mapping[r['source_group']]==f];old=[i for i in ids if i<728]
  if {rows[i]['source_group'] for i in ids}&{rows[i]['source_group'] for i in test}:raise ValueError('Group leakage')
  teacher,_=make_state(rgb[old],[rows[i] for i in old],(6144,2048,114));scores=predict(rgb[test],teacher);fold_data.append((ids,test,scores))
  for i,c in zip(test,scores.argmax(1).tolist()):
   if i<728:teacher_oof[i]='ABCDEF'[c]
 if teacher_oof!=[r['prediction'] for r in json.loads((ref/'oof_answers.json').read_text(encoding='utf-8'))]:raise ValueError('RGB teacher OOF differs')
 candidates=[];allpred={}
 for view in RECIPE['views']:
  for alpha in RECIPE['alphas']:
   df=[]
   for ids,test,teacher in fold_data:df.append(predict(features[view][test],fit_head(features[view][ids],[rows[i] for i in ids],alpha)))
   for weight in RECIPE['dino_weights']:
    oof=['']*len(rows)
    for (_,test,teacher),dscore in zip(fold_data,df):
     for i,c in zip(test,blend_scores(teacher,dscore,weight).argmax(1).tolist()):oof[i]='ABCDEF'[c]
    oldm=mixed_metrics(rows[:728],oof[:728],source)['exact_six_class'];allm=mixed_metrics(rows,oof,source)
    item=dict(view=view,alpha=alpha,dino_weight=weight,old_oof=oldm,baseline_old_oof=baseline,all_class_gates=all_class_gates(oldm,baseline),mean_group_accuracy=allm['mean_source_group_accuracy'])
    candidates.append(item);allpred[view,alpha,weight]=oof;log('candidate',view=view,alpha=alpha,dino_weight=weight,old_correct=oldm['correct'],macro_f1=oldm['macro_f1'],eligible=all(item['all_class_gates'].values()))
 winner=max(candidates,key=candidate_key);view=winner['view'];oof=allpred[view,winner['alpha'],winner['dino_weight']];teacher=load_file(str(ref/'facies_head.safetensors'))
 state=pack_state(teacher,fit_head(features[view],rows,winner['alpha']),winner['dino_weight']);save_file({k:v.contiguous() for k,v in state.items()},str(out/'facies_head.safetensors'))
 frozen=dict(stage='before_validation_or_external_DINO_features_or_predictions',winner=winner,candidates=candidates,fold_source_groups=mapping,training_ids=[r['record_id'] for r in rows],head_sha256=source.digest(out/'facies_head.safetensors'))
 source.write_json(out/'selection_frozen.json',frozen);source.write_json(out/'oof_answers.json',[dict(record_id=r['record_id'],prediction=c,allowed_codes=r['allowed_codes'],source_group=r['source_group'],fold=mapping[r['source_group']]) for r,c in zip(rows,oof)])
 vfeatures=extract_features(val,dino);hfeatures=extract_features(held,dino);save_file({**{'val__'+k:v for k,v in vfeatures.items()},**{'external__'+k:v for k,v in hfeatures.items()}},str(out/'dino_evaluation.safetensors'))
 rcfg=json.loads((ref/'run_config.json').read_text(encoding='utf-8'));vr=load_file(str(Path(rcfg['reference'])/'validation_features.safetensors'))['raw_val'][:,:6144];hr=load_file(str(prev/'external_features.safetensors'))['external_holdout']
 def code(s):return ['ABCDEF'[i] for i in s.argmax(1).tolist()]
 vp,bp=code(predict_features(vr,vfeatures[view],state)),code(predict(vr,teacher));hp,hb=code(predict_features(hr,hfeatures[view],state)),code(predict(hr,teacher));expected=[r['conversations'][1]['value'] for r in val];vm,bm=source.classification_metrics(expected,vp),source.classification_metrics(expected,bp);hm,hbm=mixed_metrics(held,hp,source),mixed_metrics(held,hb,source)
 gates=all_class_gates(vm,bm);gates.update(train_eligible=all(winner['all_class_gates'].values()),accuracy=vm['correct']>=bm['correct'],no_previously_correct_case_lost=not any(b==e and c!=e for e,b,c in zip(expected,bp,vp)))
 wg={}
 for well in sorted({r['well'] for r in held}):
  for c in sorted({r['allowed_codes'][0] for r in held if r['well']==well}):wg[well+'_'+c]=sum(p==c for r,p in zip(held,hp) if r['well']==well and r['allowed_codes']==[c])>=sum(p==c for r,p in zip(held,hb) if r['well']==well and r['allowed_codes']==[c])
 wg['at_least_one_gain']=hm['exact_six_class']['correct']>hbm['exact_six_class']['correct'];gates['new_wells_eligible']=all(wg.values())
 result=dict(status='research_all_gates_passed_requires_independent_wells' if all(gates.values()) else 'not_promoted_all_class_or_new_well_gate_failed',selected={k:winner[k] for k in ('view','alpha','dino_weight')},head_sha256=source.digest(out/'facies_head.safetensors'),selection_frozen_sha256=source.digest(out/'selection_frozen.json'),old_oof=winner['old_oof'],baseline_old_oof=baseline,training_gates=winner['all_class_gates'],oof=mixed_metrics(rows,oof,source),train=mixed_metrics(rows,code(predict_features(rgb,features[view],state)),source),val=vm,baseline_val=bm,external_holdout=hm,baseline_external_holdout=hbm,validation_gates=gates,new_well_gates=wg,distribution_approved=False,installed_in_application=False,pretrained_encoders_changed=False,language_weights_changed=False,controls_reused_from_previous_development=True)
 source.write_json(out/'result.json',result)
 for name,rr,pred,basepred in [('val',val,vp,bp),('external',held,hp,hb)]:source.write_json(out/(name+'_answers.json'),[dict(record_id=r.get('record_id',r.get('metadata',{}).get('record_id')),well=r.get('well'),expected_code=r['allowed_codes'][0] if 'allowed_codes' in r else r['conversations'][1]['value'],prediction=c,baseline_prediction=b) for r,c,b in zip(rr,pred,basepred)])
 contract.update(type='qwen_dinov2_small_score_fusion',feature_contract_version='qwen3vl_dinov2_facies_v1',head_sha256=result['head_sha256'],acceptance_status=result['status'],training_recipe=RECIPE,selected_profile=result['selected'],distribution_approved=False,feature_spec='Original Qwen v3 teacher + learned DINOv2-small CLS/patch1152 head, weighted uncalibrated scores',dino_contract=dc,training_source_license=cfg['source_rights'])
 source.write_json(out/'MODEL_CONTRACT.json',contract);log('complete',old_oof=winner['old_oof']['correct'],val=vm['correct'],external=hm['exact_six_class']['correct'],selected=result['selected'],status=result['status'])


if __name__=='__main__':main()
