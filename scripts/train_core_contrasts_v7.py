"""Offline fixed-recipe continuation on the prepared v7 source contrasts.

This is local research only (Rome NC-ND / Salpi unresolved redistribution).
All prior data are retained, no val search and no automatic model promotion.
"""
import argparse
import gc
import json
from pathlib import Path
import shutil
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import train_unibo_visual_facies as source
from scripts.train_core_facies_conservative import make_state,fit_update,codes
from scripts.conservative_facies_update import training_gates,preservation,PROTECTED
from scripts.partial_facies_ridge import mixed_metrics
from scripts.prepare_core_contrasts_20261031_v7 import SOURCES,PHOTOS,PDFS,pixel_digest

RECIPE=dict(profile='qwen_post',alpha=1.,proximity=10.,new_weight=.25,end_face_weight=.25,folds=3,
 selection='One fixed previous-v6 recipe; no search after new source acquisition',
 normalizer='Old728 train subset only per fold; prior re-fit on that subset',
 training_scope='local_research_only',distribution_approved=False,installed_in_application=False,
 validation='Previous50 unchanged and loaded only after train OOF and recipe frozen',encoder_or_chat_weights_changed=False)


def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--data',type=Path,default=ROOT.parent/'vlm_core_v1/data/core_contrasts_20261031_v7_clean_v2')
 p.add_argument('--previous',type=Path,default=ROOT.parent/'vlm_core_v1/outputs/core_qwen_core_facies_20261031_v6_conservative')
 p.add_argument('--output',type=Path,required=True);a=p.parse_args();data,prev,out=(x.resolve() for x in (a.data,a.previous,a.output))
 if out.exists() or not out.is_relative_to(ROOT.parent/'vlm_core_v1/outputs'):raise ValueError('Choose new local experiment output')
 pcfg=json.loads((prev/'run_config.json').read_text(encoding='utf-8'));ref=Path(pcfg['reference']);contract=json.loads((prev/'MODEL_CONTRACT.json').read_text(encoding='utf-8'))
 previous=source.read_rows(prev/'train.jsonl');extra=source.read_rows(data/'all_local.jsonl');val=source.read_rows(prev/'val.jsonl')
 summary=json.loads((data/'summary.json').read_text(encoding='utf-8'))
 if len(previous)!=752 or len(extra)!=52 or len(val)!=50 or summary.get('do_not_train') or summary['sources']!=SOURCES or summary['photo_sha256']!=PHOTOS:raise ValueError('Unreviewed preparation')
 if source.package_versions()!=contract['package_versions']:raise ValueError('Extraction environment differs')
 from PIL import Image
 seen={};pixels={}
 for r in previous+val:
  seen[r.get('image_sha256',r.get('metadata',{}).get('image_sha256'))]=r['image']
  with Image.open(r['image']) as im:pixels[pixel_digest(im.convert('RGB'))]=r['image']
 for r in extra:
  if source.digest(r['image'])!=r['image_sha256'] or r['image_sha256'] in seen:raise ValueError('Changed/duplicate crop')
  with Image.open(r['image']) as im:h=pixel_digest(im.convert('RGB'))
  if h!=r['pixel_sha256'] or h in pixels:raise ValueError('Changed/duplicate decoded pixels')
  if r['source']!=SOURCES[r['source_id']] or r['source_file_sha256']!=PHOTOS[r['source_file']] or r['domain']!='core_longitudinal':raise ValueError('Source/crop provenance differs')
  seen[r['image_sha256']]=r['image'];pixels[h]=r['image']
 base=Path(pcfg['base'])
 from scripts.classify_with_qwen_facies_head import streamed_sha
 if streamed_sha(base/'model.safetensors')!=contract['base_weights_sha256']:raise ValueError('Base differs')
 for name,h in contract['preprocessing_files_sha256'].items():
  if source.digest(base/name)!=h:raise ValueError('Processor differs')
 out.mkdir(parents=True);rows=previous+extra;old=rows[:728];newrows=rows[728:]
 source.write_rows(out/'train.jsonl',rows);source.write_rows(out/'val.jsonl',val)
 deps=[Path(__file__),ROOT/'scripts/prepare_core_contrasts_20261031_v7.py',ROOT/'scripts/train_core_facies_conservative.py',
  ROOT/'scripts/conservative_facies_update.py',ROOT/'scripts/partial_facies_ridge.py',ROOT/'scripts/facies_head_v2_features.py',
  prev/'features.safetensors',prev/'train.jsonl',prev/'val.jsonl',prev/'selection_frozen.json',prev/'audit_result.json',
  data/'all_local.jsonl',data/'summary.json',ref/'facies_head.safetensors',ref/'oof_answers.json']
 cfg=dict(recipe=RECIPE,base=str(base),reference=str(ref),previous=str(prev),data=str(data),train_count=len(rows),new_count=len(extra),
  dependencies={str(x):source.digest(x) for x in deps},source_rights='Po CC BY; Rome CC BY-NC-ND and previous Salpi unresolved. Local only; mixed weights not distributed.',
  distribution_approved=False,installed_in_application=False,prepared_test_pixels_opened=0)
 source.write_json(out/'run_config.json',cfg);shutil.copy2(__file__,out/'training_script_snapshot.py')
 from scripts.train_qwen_source_pilot import configure_offline_environment,_deny_network
 configure_offline_environment();sys.addaudithook(_deny_network)
 import torch
 from safetensors.torch import load_file,save_file
 torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.manual_seed(20261031)
 def log(event,**kw):
  obj=dict(event=event,**kw)
  with (out/'progress.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(obj)+'\n')
  print(json.dumps(obj),flush=True)
 cache=load_file(str(prev/'features.safetensors'))['train']
 if cache.shape!=(752,6144):raise ValueError('Previous feature dimensions differ')
 if not torch.cuda.is_available() or torch.cuda.mem_get_info()[0]<5*1024**3:raise RuntimeError('Need5GiB free CUDA')
 from transformers import AutoProcessor,Qwen3VLForConditionalGeneration
 from scripts.train_qwen_visual_facies_head import pool_features
 processor=AutoProcessor.from_pretrained(base,local_files_only=True)
 model=Qwen3VLForConditionalGeneration.from_pretrained(base,local_files_only=True,dtype=torch.float16,attn_implementation='sdpa',device_map='cuda');model.eval();model.requires_grad_(False);features=[]
 for i,r in enumerate(extra,1):
  with Image.open(r['image']) as im:image=im.convert('RGB');image.thumbnail((300,300))
  inputs=source.prompt_inputs(processor,image,source.PROMPT).to('cuda')
  with torch.inference_mode():
   visual=model.model.visual(inputs['pixel_values'].to(model.dtype),grid_thw=inputs['image_grid_thw'],return_dict=True);features.append(pool_features(visual).cpu())
  del inputs,visual
  if i%8==0 or i==len(extra):log('new_features',row=i,total=len(extra))
 del model,processor;gc.collect();torch.cuda.empty_cache();raw=torch.cat((cache,torch.stack(features)));save_file({'train':raw},str(out/'features.safetensors'))
 oldraw,newraw=raw[:728],raw[728:];dims=(contract['qwen_feature_dimension'],contract['pre_merger_feature_dimension'],contract['texture_dimension'])
 mapping=json.loads((prev/'selection_frozen.json').read_text(encoding='utf-8'))['fold_source_groups']
 for g in sorted({r['source_group'] for r in extra}):
  def cost(f):
   assigned=[r for r in rows if r['source_group'] in mapping and mapping[r['source_group']]==f]
   return sum(r['allowed_codes']==['B'] for r in assigned),len(assigned),f
  mapping[g]=min(range(3),key=cost)
 oof=['']*len(rows);boof=['']*728
 for f in range(3):
  om=torch.tensor([mapping[r['source_group']]!=f for r in old]);nm=torch.tensor([mapping[r['source_group']]!=f for r in newrows]);held=torch.tensor([mapping[r['source_group']]==f for r in rows])
  oo=[r for r,k in zip(old,om.tolist()) if k];nn=[r for r,k in zip(newrows,nm.tolist()) if k]
  if {r['source_group'] for r in oo+nn}&{r['source_group'] for r,k in zip(rows,held.tolist()) if k}:raise ValueError('Group leakage')
  state=fit_update(oldraw[om],oo,newraw[nm],nn,dims,RECIPE['proximity'],RECIPE['new_weight'])
  for i,c in zip(torch.where(held)[0].tolist(),codes(raw[held],state)):oof[i]=c
  bs,_=make_state(oldraw[om],oo,dims)
  for i,c in zip(torch.where(~om)[0].tolist(),codes(oldraw[~om],bs)):boof[i]=c
 if boof!=[r['prediction'] for r in json.loads((ref/'oof_answers.json').read_text(encoding='utf-8'))]:raise ValueError('Baseline OOF differs')
 baseline=mixed_metrics(old,boof,source)['exact_six_class'];updated=mixed_metrics(old,oof[:728],source)['exact_six_class']
 checks=training_gates(updated,baseline)
 frozen=dict(stage='before_validation_features_and_predictions',recipe=RECIPE,baseline_old_oof=baseline,old_oof=updated,training_gates=checks,
  fold_source_groups=mapping,training_record_ids=[r['record_id'] for r in rows],selection='Fixed prior recipe; diagnostic only unless every gate passes')
 source.write_json(out/'selection_frozen.json',frozen)
 source.write_json(out/'oof_answers.json',[dict(record_id=r['record_id'],source_group=r['source_group'],allowed_codes=r['allowed_codes'],prediction=c,fold=mapping[r['source_group']]) for r,c in zip(rows,oof)])
 state=fit_update(oldraw,old,newraw,newrows,dims,RECIPE['proximity'],RECIPE['new_weight']);save_file({k:v.contiguous() for k,v in state.items()},str(out/'facies_head.safetensors'))
 result=dict(head_sha256=source.digest(out/'facies_head.safetensors'),selection_frozen_sha256=source.digest(out/'selection_frozen.json'),
  train=mixed_metrics(rows,codes(raw,state),source),oof=mixed_metrics(rows,oof,source),old_train_oof=updated,baseline_old_train_oof=baseline,training_gates=checks,
  distribution_approved=False,installed_in_application=False,encoder_or_chat_weights_changed=False)
 refcfg=json.loads((ref/'run_config.json').read_text(encoding='utf-8'));vraw=load_file(str(Path(refcfg['reference'])/'validation_features.safetensors'))['raw_val'][:,:dims[0]]
 expected=[r['conversations'][1]['value'] for r in val];vp=codes(vraw,state);bp=codes(vraw,load_file(str(ref/'facies_head.safetensors')))
 vm=source.classification_metrics(expected,vp);bm=source.classification_metrics(expected,bp)
 gates=preservation(vm,bm);gates.update(B_improved=vm['per_class_recall']['B']>bm['per_class_recall']['B'],C_improved=vm['per_class_recall']['C']>bm['per_class_recall']['C'],
  accuracy=vm['correct']>=bm['correct'],macro_f1=vm['macro_f1']+1e-12>=bm['macro_f1'],train_eligible=all(checks.values()),
  no_protected_correct_case_lost=not any(e in PROTECTED and b==e and c!=e for e,b,c in zip(expected,bp,vp)))
 result.update(val=vm,baseline_val=bm,validation_gates=gates,status='research_gates_passed_requires_independent_validation' if all(gates.values()) else 'not_promoted_preservation_or_target_gate_failed')
 source.write_json(out/'result.json',result);source.write_json(out/'val_answers.json',[dict(record_id=r['metadata']['record_id'],expected_code=e,prediction=c,baseline_prediction=b) for r,e,c,b in zip(val,expected,vp,bp)])
 contract.update(type='frozen_qwen_prior_regularized_partial_ridge_head',head_sha256=result['head_sha256'],acceptance_status=result['status'],training_recipe=RECIPE,
  selected_profile=dict(profile='qwen_post',proximity=RECIPE['proximity'],new_weight=RECIPE['new_weight']),training_source_license=cfg['source_rights'],distribution_approved=False)
 source.write_json(out/'MODEL_CONTRACT.json',contract);log('complete',val=vm,training_gates=checks,validation_gates=gates)


if __name__=='__main__':main()
