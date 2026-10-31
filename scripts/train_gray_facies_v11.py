"""Train-only selection of two-view Qwen heads on822 records, no test tuning."""
import argparse
import json
from pathlib import Path
import shutil
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import train_unibo_visual_facies as source
from scripts.gray_facies_head import grayscale_view,fit_gray,blend_scores,candidate_key,pack_state,predict_features
from scripts.facies_head_v2_features import predict
from scripts.train_core_facies_conservative import make_state
from scripts.kernel_facies_head import all_class_gates
from scripts.partial_facies_ridge import mixed_metrics

RECIPE=dict(alphas=[.1,1.,10.],gray_weights=[.1,.25,.5],folds=3,seed=20261031,
 rgb='Original v3 ridge teacher, re-fit only on OLD728 train subset per fold; final exact v3 state',
 gray='Class+source-group balanced partial ridge on all train records; external0.5,endface0.25; normalizer per train fold only',
 view='Pillow grayscale luma -> RGB, original thumbnail300/processor; colour view unchanged',
 selection='9 candidates by old OOF all-six recall/F1 preservation and macro gain; worst recall gain, worst F1 gain, mean group accuracy, macroF1, smaller gray weight, larger alpha. No val/external selection.',
 scope='Two heads on original frozen Qwen features; not language/vision LoRA or general chat',
 encoder_or_language_weights_changed=False,distribution_approved=False,installed_in_application=False)


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output.resolve();prev=ROOT.parent/'vlm_core_v1/outputs/core_qwen_new_well_facies_20261031_v10'
 if out.exists() or not out.is_relative_to(ROOT.parent/'vlm_core_v1/outputs'):raise ValueError('Choose a new local output')
 pcfg=json.loads((prev/'run_config.json').read_text(encoding='utf-8'));base=Path(pcfg['base']);ref=Path(pcfg['reference']);contract=json.loads((prev/'MODEL_CONTRACT.json').read_text(encoding='utf-8'))
 if not (prev/'audit_result.json').is_file():raise ValueError('Previous run unaudited')
 for f,h in pcfg['dependencies'].items():
  if source.digest(f)!=h:raise ValueError('Previous dependency changed')
 from scripts.classify_with_qwen_facies_head import streamed_sha
 if streamed_sha(base/'model.safetensors')!=contract['base_weights_sha256'] or source.package_versions()!=contract['package_versions']:raise ValueError('Base/runtime changed')
 for f,h in contract['preprocessing_files_sha256'].items():
  if source.digest(base/f)!=h:raise ValueError('Processor changed')
 rows,val,held=[source.read_rows(prev/f) for f in ('train.jsonl','val.jsonl','external_holdout.jsonl')]
 if len(rows)!=822 or len(val)!=50 or len(held)!=16:raise ValueError('Data changed')
 out.mkdir(parents=True)
 for name,rr in (('train',rows),('val',val),('external_holdout',held)):source.write_rows(out/(name+'.jsonl'),rr)
 deps=[Path(__file__),ROOT/'scripts/gray_facies_head.py',ROOT/'scripts/vision_facies_adapter.py',ROOT/'scripts/kernel_facies_head.py',ROOT/'scripts/partial_facies_ridge.py',ROOT/'scripts/facies_head_v2_features.py',ROOT/'scripts/train_core_facies_conservative.py',prev/'features.safetensors',prev/'external_features.safetensors',prev/'selection_frozen.json',prev/'audit_result.json',prev/'train.jsonl',prev/'val.jsonl',prev/'external_holdout.jsonl',ref/'facies_head.safetensors',ref/'oof_answers.json']
 cfg=dict(recipe=RECIPE,base=str(base),previous=str(prev),reference=str(ref),train_count=822,validation_count=50,external_count=16,
  dependencies={str(f):source.digest(f) for f in deps},source_rights=pcfg['source_rights'],distribution_approved=False,installed_in_application=False,prepared_test_pixels_opened=0)
 source.write_json(out/'run_config.json',cfg);shutil.copy2(__file__,out/'training_script_snapshot.py')
 from scripts.train_qwen_source_pilot import configure_offline_environment,_deny_network
 configure_offline_environment();sys.addaudithook(_deny_network)
 import torch
 from safetensors.torch import load_file,save_file
 from transformers import AutoProcessor
 from PIL import Image,ImageDraw
 from scripts.vision_facies_adapter import load_original_vision,batch_inputs
 from scripts.train_qwen_visual_facies_head import pool_features
 torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.manual_seed(RECIPE['seed'])
 if not torch.cuda.is_available() or torch.cuda.mem_get_info()[0]<2*1024**3:raise RuntimeError('Need2GiB free CUDA')
 processor=AutoProcessor.from_pretrained(base,local_files_only=True);vision=load_original_vision(base)
 def log(event,**data):
  v=dict(event=event,**data)
  with (out/'progress.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(v)+'\n')
  print(json.dumps(v),flush=True)
 def extract(records):
  result=[]
  for i,r in enumerate(records):
   expected=r.get('image_sha256',r.get('metadata',{}).get('image_sha256'))
   if source.digest(r['image'])!=expected:raise ValueError('Source pixels changed')
   with Image.open(r['image']) as im:inp=batch_inputs(processor,[grayscale_view(im)],'cuda')
   with torch.inference_mode():result.append(pool_features(vision(inp['pixel_values'],grid_thw=inp['grid'],return_dict=True)).cpu())
   if (i+1)%100==0:log('gray_features',done=i+1,total=len(records))
  return torch.stack(result)
 rgb=load_file(str(prev/'features.safetensors'))['train']
 # Control the new loader using train pixels only, before any selection.
 with Image.open(rows[0]['image']) as im:inp=batch_inputs(processor,[im.convert('RGB')],'cuda')
 with torch.inference_mode():check=pool_features(vision(inp['pixel_values'],grid_thw=inp['grid'],return_dict=True)).cpu()
 delta=float((check-rgb[0]).abs().max())
 if delta>1e-5:raise ValueError('Original colour feature contract changed')
 gray=extract(rows);save_file({'gray_train':gray},str(out/'gray_train.safetensors'));log('train_features_complete',original_rgb_delta=delta)
 mapping=json.loads((prev/'selection_frozen.json').read_text(encoding='utf-8'))['fold_source_groups'];baseline=mixed_metrics(rows[:728],[r['prediction'] for r in json.loads((ref/'oof_answers.json').read_text(encoding='utf-8'))],source)['exact_six_class']
 fold_data=[]
 for f in range(3):
  ids=[i for i,r in enumerate(rows) if mapping[r['source_group']]!=f];test=[i for i,r in enumerate(rows) if mapping[r['source_group']]==f];old=[i for i in ids if i<728]
  if {rows[i]['source_group'] for i in ids}&{rows[i]['source_group'] for i in test}:raise ValueError('Group leakage')
  teacher,_=make_state(rgb[old],[rows[i] for i in old],(6144,2048,114));fold_data.append((ids,test,predict(rgb[test],teacher)))
 candidates=[];predictions={}
 for alpha in RECIPE['alphas']:
  grayfold=[]
  for ids,test,teacher in fold_data:
   gs=fit_gray(gray[ids],[rows[i] for i in ids],alpha);grayfold.append(predict(gray[test],gs))
  for weight in RECIPE['gray_weights']:
   oof=['']*len(rows)
   for (_,test,teacher),gscore in zip(fold_data,grayfold):
    for i,c in zip(test,blend_scores(teacher,gscore,weight).argmax(1).tolist()):oof[i]='ABCDEF'[c]
   oldm=mixed_metrics(rows[:728],oof[:728],source)['exact_six_class'];allm=mixed_metrics(rows,oof,source)
   item=dict(alpha=alpha,gray_weight=weight,old_oof=oldm,baseline_old_oof=baseline,all_class_gates=all_class_gates(oldm,baseline),mean_group_accuracy=allm['mean_source_group_accuracy'])
   candidates.append(item);predictions[alpha,weight]=oof;log('candidate',alpha=alpha,gray_weight=weight,old_correct=oldm['correct'],macro_f1=oldm['macro_f1'],eligible=all(item['all_class_gates'].values()))
 winner=max(candidates,key=candidate_key);oof=predictions[winner['alpha'],winner['gray_weight']]
 state=pack_state(load_file(str(ref/'facies_head.safetensors')),fit_gray(gray,rows,winner['alpha']),winner['gray_weight']);save_file({k:v.contiguous() for k,v in state.items()},str(out/'facies_head.safetensors'))
 frozen=dict(stage='before_gray_validation_and_external_features_or_predictions',winner=winner,candidates=candidates,fold_source_groups=mapping,training_ids=[r['record_id'] for r in rows],head_sha256=source.digest(out/'facies_head.safetensors'))
 source.write_json(out/'selection_frozen.json',frozen);source.write_json(out/'oof_answers.json',[dict(record_id=r['record_id'],prediction=c,allowed_codes=r['allowed_codes'],source_group=r['source_group'],fold=mapping[r['source_group']]) for r,c in zip(rows,oof)])
 # These controls have been used in previous development; never presented as a new blind test.
 vg,hg=extract(val),extract(held);save_file({'gray_val':vg,'gray_external':hg},str(out/'gray_evaluation.safetensors'));del vision,processor;torch.cuda.empty_cache()
 rcfg=json.loads((ref/'run_config.json').read_text(encoding='utf-8'));vr=load_file(str(Path(rcfg['reference'])/'validation_features.safetensors'))['raw_val'][:,:6144];hr=load_file(str(prev/'external_features.safetensors'))['external_holdout'];teacher=load_file(str(ref/'facies_head.safetensors'))
 def code(s):return ['ABCDEF'[i] for i in s.argmax(1).tolist()]
 vp,bp=code(predict_features(vr,vg,state)),code(predict(vr,teacher));hp,hb=code(predict_features(hr,hg,state)),code(predict(hr,teacher));expected=[r['conversations'][1]['value'] for r in val];vm,bm=source.classification_metrics(expected,vp),source.classification_metrics(expected,bp);hm,hbm=mixed_metrics(held,hp,source),mixed_metrics(held,hb,source)
 gates=all_class_gates(vm,bm);gates.update(train_eligible=all(winner['all_class_gates'].values()),accuracy=vm['correct']>=bm['correct'],no_previously_correct_case_lost=not any(b==e and c!=e for e,b,c in zip(expected,bp,vp)))
 wellgates={}
 for well in sorted({r['well'] for r in held}):
  for c in sorted({r['allowed_codes'][0] for r in held if r['well']==well}):wellgates[well+'_'+c]=sum(p==c for r,p in zip(held,hp) if r['well']==well and r['allowed_codes']==[c])>=sum(p==c for r,p in zip(held,hb) if r['well']==well and r['allowed_codes']==[c])
 wellgates['at_least_one_gain']=hm['exact_six_class']['correct']>hbm['exact_six_class']['correct'];gates['new_wells_eligible']=all(wellgates.values())
 result=dict(status='research_all_gates_passed_requires_independent_wells' if all(gates.values()) else 'not_promoted_all_class_or_new_well_gate_failed',selected={k:winner[k] for k in ('alpha','gray_weight')},head_sha256=source.digest(out/'facies_head.safetensors'),selection_frozen_sha256=source.digest(out/'selection_frozen.json'),old_oof=winner['old_oof'],baseline_old_oof=baseline,training_gates=winner['all_class_gates'],oof=mixed_metrics(rows,oof,source),train=mixed_metrics(rows,code(predict_features(rgb,gray,state)),source),val=vm,baseline_val=bm,external_holdout=hm,baseline_external_holdout=hbm,validation_gates=gates,new_well_gates=wellgates,distribution_approved=False,installed_in_application=False,encoder_or_language_weights_changed=False,controls_reused_from_previous_development=True)
 source.write_json(out/'result.json',result)
 for name,rr,pred,basepred in [('val',val,vp,bp),('external',held,hp,hb)]:source.write_json(out/(name+'_answers.json'),[dict(record_id=r.get('record_id',r.get('metadata',{}).get('record_id')),well=r.get('well'),expected_code=r['allowed_codes'][0] if 'allowed_codes' in r else r['conversations'][1]['value'],prediction=c,baseline_prediction=b) for r,c,b in zip(rr,pred,basepred)])
 contract.update(type='qwen_rgb_gray_teacher_blend',feature_contract_version='qwen3vl_rgb_gray_facies_v1',head_sha256=result['head_sha256'],acceptance_status=result['status'],training_recipe=RECIPE,selected_profile=result['selected'],distribution_approved=False,feature_spec='RGB original v3 teacher + trained grayscale partial ridge, two4096 post profiles, weighted raw scores',training_source_license=cfg['source_rights'])
 source.write_json(out/'MODEL_CONTRACT.json',contract);log('complete',old_oof=winner['old_oof']['correct'],val=vm['correct'],external=hm['exact_six_class']['correct'],status=result['status'])


if __name__=='__main__':main()
