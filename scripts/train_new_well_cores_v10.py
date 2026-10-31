"""Fixed conservative continuation on new photographed wells; no test tuning."""
import argparse
import gc
import json
from pathlib import Path
import shutil
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import train_unibo_visual_facies as source
from scripts.train_core_facies_conservative import fit_update,codes
from scripts.partial_facies_ridge import mixed_metrics
from scripts.kernel_facies_head import all_class_gates
from scripts.prepare_new_well_cores_v10 import digest,pixel_digest,SOURCES,PDFS

RECIPE=dict(alpha=1.,proximity=100.,new_weight=.25,folds=3,seed=20261031,profile='qwen_post',
 prior='v3 re-fit on old728 TRAIN rows only per fold',normalization='Old728 TRAIN subset only per fold',
 selection='One fixed conservative recipe, no hyperparameter or checkpoint search. Freeze OOF/head before val50 or external-heldout inference.',
 acceptance='All six recall/F1 nondecreasing and macro-F1 higher on old OOF and reused val; no previously correct val case lost; no held-well recall loss and at least one gain.',
 vision_encoder_changed=False,language_weights_changed=False,distribution_approved=False,installed_in_application=False)


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',type=Path,default=ROOT.parent/'vlm_core_v1/data/new_well_cores_20261031_v10_reviewed');p.add_argument('--output',type=Path,required=True);a=p.parse_args();data,out=a.data.resolve(),a.output.resolve()
 prev=ROOT.parent/'vlm_core_v1/outputs/core_qwen_core_contrasts_20261031_v7';ref=ROOT.parent/'vlm_core_v1/outputs/core_qwen_web_partial_facies_20261031_v3'
 if out.exists() or not out.is_relative_to(ROOT.parent/'vlm_core_v1/outputs'):raise ValueError('Choose a new local output')
 summary=json.loads((data/'summary.json').read_text(encoding='utf-8'));previous=source.read_rows(prev/'train.jsonl');extra=source.read_rows(data/'train.jsonl');hold=source.read_rows(data/'external_holdout.jsonl');val=source.read_rows(prev/'val.jsonl')
 if summary.get('do_not_train') or summary['sources']!=SOURCES or summary['pdf_sha256']!=PDFS or len(previous)!=804 or len(extra)!=18 or len(hold)!=16:raise ValueError('Unreviewed preparation')
 if {r['well'] for r in extra}&{r['well'] for r in hold}:raise ValueError('Held-out well in train')
 from PIL import Image
 known=set();pixels=set()
 for r in previous+val:
  known.add(digest(r['image']))
  with Image.open(r['image']) as im:pixels.add(pixel_digest(im.convert('RGB')))
 for r in extra+hold:
  if digest(r['image'])!=r['image_sha256'] or r['image_sha256'] in known:raise ValueError('Changed/duplicate image')
  with Image.open(r['image']) as im:h=pixel_digest(im.convert('RGB'))
  if h!=r['pixel_sha256'] or h in pixels or r['source']!=SOURCES[r['source_id']]:raise ValueError('Changed/duplicate source pixels')
  known.add(r['image_sha256']);pixels.add(h)
 cfgold=json.loads((prev/'run_config.json').read_text(encoding='utf-8'));base=Path(cfgold['base']);contract=json.loads((prev/'MODEL_CONTRACT.json').read_text(encoding='utf-8'))
 for f,h in cfgold['dependencies'].items():
  if digest(f)!=h:raise ValueError('Previous dependency changed')
 from scripts.classify_with_qwen_facies_head import streamed_sha
 if streamed_sha(base/'model.safetensors')!=contract['base_weights_sha256'] or source.package_versions()!=contract['package_versions']:raise ValueError('Exact base/runtime required')
 for f,h in contract['preprocessing_files_sha256'].items():
  if digest(base/f)!=h:raise ValueError('Processor changed')
 out.mkdir(parents=True);rows=previous+extra;source.write_rows(out/'train.jsonl',rows);source.write_rows(out/'val.jsonl',val);source.write_rows(out/'external_holdout.jsonl',hold)
 deps=[Path(__file__),ROOT/'scripts/prepare_new_well_cores_v10.py',ROOT/'scripts/train_core_facies_conservative.py',ROOT/'scripts/conservative_facies_update.py',ROOT/'scripts/vision_facies_adapter.py',ROOT/'scripts/kernel_facies_head.py',
  data/'train.jsonl',data/'external_holdout.jsonl',data/'summary.json',prev/'features.safetensors',prev/'selection_frozen.json',ref/'facies_head.safetensors',ref/'oof_answers.json']
 cfg=dict(recipe=RECIPE,base=str(base),previous=str(prev),reference=str(ref),data=str(data),train_count=len(rows),new_count=len(extra),held_count=len(hold),
  dependencies={str(f):digest(f) for f in deps},source_rights='Mixed sources include previous Salpi/Rome and new GEUS/GSJ with unverified redistribution grant; local research only.',distribution_approved=False,prepared_test_pixels_opened=0)
 source.write_json(out/'run_config.json',cfg);shutil.copy2(__file__,out/'training_script_snapshot.py')
 from scripts.train_qwen_source_pilot import configure_offline_environment,_deny_network
 configure_offline_environment();sys.addaudithook(_deny_network)
 import torch
 from safetensors.torch import load_file,save_file
 from transformers import AutoProcessor
 from scripts.vision_facies_adapter import load_original_vision,batch_inputs
 from scripts.train_qwen_visual_facies_head import pool_features
 torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.manual_seed(RECIPE['seed'])
 if not torch.cuda.is_available() or torch.cuda.mem_get_info()[0]<2*1024**3:raise RuntimeError('Need2GiB free CUDA')
 processor=AutoProcessor.from_pretrained(base,local_files_only=True);vision=load_original_vision(base)
 def features(records):
  result=[]
  for r in records:
   with Image.open(r['image']) as im:inp=batch_inputs(processor,[im.convert('RGB')],'cuda')
   with torch.inference_mode():v=vision(inp['pixel_values'],grid_thw=inp['grid'],return_dict=True);result.append(pool_features(v).cpu())
  return torch.stack(result)
 newraw=features(extra);raw=torch.cat((load_file(str(prev/'features.safetensors'))['train'],newraw));save_file({'train':raw},str(out/'features.safetensors'))
 mapping=json.loads((prev/'selection_frozen.json').read_text(encoding='utf-8'))['fold_source_groups']
 from hashlib import sha256
 for g in sorted({r['source_group'] for r in extra}):mapping[g]=int(sha256(g.encode()).hexdigest()[:8],16)%3
 dims=(6144,2048,114);oof=['']*len(rows)
 for fold in range(3):
  ids=[i for i,r in enumerate(rows) if mapping[r['source_group']]!=fold];held=[i for i,r in enumerate(rows) if mapping[r['source_group']]==fold];old=[i for i in ids if i<728];new=[i for i in ids if i>=728]
  if {rows[i]['source_group'] for i in ids}&{rows[i]['source_group'] for i in held}:raise ValueError('Group leakage')
  state=fit_update(raw[old],[rows[i] for i in old],raw[new],[rows[i] for i in new],dims,RECIPE['proximity'],RECIPE['new_weight'])
  for i,c in zip(held,codes(raw[held],state)):oof[i]=c
 baseline_oof=mixed_metrics(rows[:728],[r['prediction'] for r in json.loads((ref/'oof_answers.json').read_text(encoding='utf-8'))],source)['exact_six_class'];old_oof=mixed_metrics(rows[:728],oof[:728],source)['exact_six_class'];tg=all_class_gates(old_oof,baseline_oof)
 state=fit_update(raw[:728],rows[:728],raw[728:],rows[728:],dims,RECIPE['proximity'],RECIPE['new_weight']);save_file({k:v.contiguous() for k,v in state.items()},str(out/'facies_head.safetensors'))
 frozen=dict(stage='before_val_and_external_predictions',recipe=RECIPE,old_oof=old_oof,baseline_old_oof=baseline_oof,training_gates=tg,fold_source_groups=mapping,head_sha256=digest(out/'facies_head.safetensors'),
  training_ids=[r['record_id'] for r in rows],held_out_wells=sorted({r['well'] for r in hold}),held_record_ids=[r['record_id'] for r in hold])
 source.write_json(out/'selection_frozen.json',frozen);source.write_json(out/'oof_answers.json',[dict(record_id=r['record_id'],prediction=c,allowed_codes=r['allowed_codes'],fold=mapping[r['source_group']],source_group=r['source_group']) for r,c in zip(rows,oof)])
 # No held-well predictions or features existed before the frozen recipe/head.
 hraw=features(hold);save_file({'external_holdout':hraw},str(out/'external_features.safetensors'));del vision,processor;gc.collect();torch.cuda.empty_cache()
 refcfg=json.loads((ref/'run_config.json').read_text(encoding='utf-8'));vraw=load_file(str(Path(refcfg['reference'])/'validation_features.safetensors'))['raw_val'][:,:6144];bs=load_file(str(ref/'facies_head.safetensors'))
 expected=[r['conversations'][1]['value'] for r in val];vp,bp=codes(vraw,state),codes(vraw,bs);vm,bm=source.classification_metrics(expected,vp),source.classification_metrics(expected,bp)
 hp,hb=codes(hraw,state),codes(hraw,bs);hm,hbm=mixed_metrics(hold,hp,source),mixed_metrics(hold,hb,source)
 hg={w:sum(c==r['allowed_codes'][0] for r,c in zip(hold,hp) if r['well']==w)>=sum(c==r['allowed_codes'][0] for r,c in zip(hold,hb) if r['well']==w) for w in sorted({r['well'] for r in hold})}
 for c in ('C','E'):hg[c+'_recall']=hm['exact_six_class']['per_class_recall'][c]>=hbm['exact_six_class']['per_class_recall'][c]
 hg['at_least_one_gain']=hm['exact_six_class']['correct']>hbm['exact_six_class']['correct']
 vg=all_class_gates(vm,bm);vg.update(accuracy=vm['correct']>=bm['correct'],no_previously_correct_case_lost=not any(b==e and c!=e for e,b,c in zip(expected,bp,vp)),train_eligible=all(tg.values()),new_wells_eligible=all(hg.values()))
 result=dict(head_sha256=digest(out/'facies_head.safetensors'),selection_frozen_sha256=digest(out/'selection_frozen.json'),old_oof=old_oof,baseline_old_oof=baseline_oof,oof=mixed_metrics(rows,oof,source),train=mixed_metrics(rows,codes(raw,state),source),
  val=vm,baseline_val=bm,external_holdout=hm,baseline_external_holdout=hbm,training_gates=tg,validation_gates=vg,new_well_gates=hg,
  status='research_gates_passed_requires_more_independent_validation' if all(vg.values()) else 'not_promoted_all_class_or_new_well_gate_failed',distribution_approved=False,installed_in_application=False,vision_encoder_changed=False,language_weights_changed=False)
 source.write_json(out/'result.json',result)
 for name,records,pred,basepred in [('val',val,vp,bp),('external',hold,hp,hb)]:source.write_json(out/(name+'_answers.json'),[dict(record_id=r.get('record_id',r.get('metadata',{}).get('record_id')),well=r.get('well'),expected_code=r['allowed_codes'][0] if 'allowed_codes' in r else r['conversations'][1]['value'],prediction=c,baseline_prediction=b) for r,c,b in zip(records,pred,basepred)])
 contract.update(head_sha256=result['head_sha256'],acceptance_status=result['status'],training_recipe=RECIPE,selected_profile=dict(profile='qwen_post',proximity=100.,new_weight=.25),distribution_approved=False,training_source_license=cfg['source_rights'])
 source.write_json(out/'MODEL_CONTRACT.json',contract);print(json.dumps(dict(status=result['status'],old_oof=old_oof['correct'],val=vm['correct'],new_wells=hm['exact_six_class']['correct'],baseline_new_wells=hbm['exact_six_class']['correct'],gates=vg),indent=2),flush=True)


if __name__=='__main__':main()
