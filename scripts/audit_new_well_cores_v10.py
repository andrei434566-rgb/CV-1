"""Replay v10 source crops, original-Qwen features, grouped fits and held wells."""
import argparse
import json
from pathlib import Path
import sys
import subprocess
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import train_unibo_visual_facies as source
from scripts.prepare_new_well_cores_v10 import digest,pixel_digest,PDFS
from scripts.train_core_facies_conservative import fit_update,codes
from scripts.partial_facies_ridge import mixed_metrics
from scripts.kernel_facies_head import all_class_gates


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--raw',type=Path,required=True)
 p.add_argument('--pdf-python',type=Path,default=Path.home()/'.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe')
 a=p.parse_args();run,rawdir=a.run.resolve(),a.raw.resolve()
 cfg=json.loads((run/'run_config.json').read_text(encoding='utf-8'));result=json.loads((run/'result.json').read_text(encoding='utf-8'));frozen=json.loads((run/'selection_frozen.json').read_text(encoding='utf-8'));recipe=cfg['recipe'];base,prev,ref=(Path(cfg[k]) for k in ('base','previous','reference'))
 for f,h in cfg['dependencies'].items():
  if digest(f)!=h:raise ValueError('Dependency changed: '+f)
 for f,h in PDFS.items():
  if digest(rawdir/f)!=h:raise ValueError('Source PDF changed')
 from scripts.classify_with_qwen_facies_head import streamed_sha
 contract=json.loads((run/'MODEL_CONTRACT.json').read_text(encoding='utf-8'))
 if streamed_sha(base/'model.safetensors')!=contract['base_weights_sha256'] or digest(run/'facies_head.safetensors')!=result['head_sha256'] or digest(run/'selection_frozen.json')!=result['selection_frozen_sha256']:raise ValueError('Base/head/frozen recipe changed')
 rows,val,hold=[source.read_rows(run/f) for f in ('train.jsonl','val.jsonl','external_holdout.jsonl')];extra=rows[804:]
 if rows[:804]!=source.read_rows(prev/'train.jsonl') or rows[:728]!=source.read_rows(ref/'train.jsonl') or val!=source.read_rows(ref/'val.jsonl'):raise ValueError('Old rows changed')
 if [r['record_id'] for r in rows]!=frozen['training_ids'] or [r['record_id'] for r in hold]!=frozen['held_record_ids']:raise ValueError('Record order changed')
 if {r.get('well') for r in rows}&{r['well'] for r in hold}:raise ValueError('Held well in train')
 from PIL import Image
 seen=set();ph=set()
 for r in rows+val+hold:
  h=digest(r['image'])
  if h in seen:raise ValueError('Duplicate file across train/val/held')
  seen.add(h)
  with Image.open(r['image']) as im:v=pixel_digest(im.convert('RGB'))
  if v in ph:raise ValueError('Duplicate pixels across train/val/held')
  ph.add(v)
 for r in extra+hold:
  if digest(rawdir/r['source_file'])!=r['source_file_sha256']:raise ValueError('Canonical source changed')
  with Image.open(rawdir/r['source_file']) as im:crop=im.convert('RGB').crop(r['box'])
  if pixel_digest(crop)!=r['pixel_sha256'] or digest(r['image'])!=r['image_sha256']:raise ValueError('Crop reconstruction differs')
 # Independently verify the same embedded profile applies to the second GEUS page.
 # Keep PDF tooling in its existing separate runtime; do not install packages
 # into the hash-pinned ML environment just to inspect two PDF resources.
 code="import json,sys; from hashlib import sha256; from pypdf import PdfReader; d=PdfReader(sys.argv[1]); print(json.dumps([sha256(d.pages[i]['/Resources']['/ColorSpace']['/DefaultCMYK'].get_object()[1].get_object().get_data()).hexdigest() for i in (11,12)]))"
 profiles=json.loads(subprocess.run([str(a.pdf_python),'-c',code,str(rawdir/'andsbjerg2003.pdf')],check=True,capture_output=True,text=True,encoding='utf-8').stdout)
 if len(set(profiles))!=1:raise ValueError('GEUS pages use different CMYK profiles')
 from scripts.train_qwen_source_pilot import configure_offline_environment,_deny_network
 configure_offline_environment();sys.addaudithook(_deny_network)
 import torch
 from safetensors.torch import load_file
 from transformers import AutoProcessor
 from scripts.vision_facies_adapter import load_original_vision,batch_inputs
 from scripts.train_qwen_visual_facies_head import pool_features
 torch.set_num_threads(4);torch.set_num_interop_threads(1)
 cache=load_file(str(run/'features.safetensors'))['train'];hraw=load_file(str(run/'external_features.safetensors'))['external_holdout']
 if not torch.equal(cache[:804],load_file(str(prev/'features.safetensors'))['train']):raise ValueError('Previous cache changed')
 processor=AutoProcessor.from_pretrained(base,local_files_only=True);vision=load_original_vision(base);fresh=[]
 for r in extra+hold:
  with Image.open(r['image']) as im:inp=batch_inputs(processor,[im.convert('RGB')],'cuda')
  with torch.inference_mode():fresh.append(pool_features(vision(inp['pixel_values'],grid_thw=inp['grid'],return_dict=True)).cpu())
 feature_delta=float((torch.stack(fresh)-torch.cat((cache[804:],hraw))).abs().max())
 if feature_delta>1e-5:raise ValueError('Fresh source features differ')
 del vision,processor;torch.cuda.empty_cache()
 dims=(6144,2048,114);mapping=frozen['fold_source_groups'];oof=['']*len(rows)
 for fold in range(3):
  ids=[i for i,r in enumerate(rows) if mapping[r['source_group']]!=fold];held=[i for i,r in enumerate(rows) if mapping[r['source_group']]==fold];old=[i for i in ids if i<728];new=[i for i in ids if i>=728]
  if {rows[i]['source_group'] for i in ids}&{rows[i]['source_group'] for i in held}:raise ValueError('Group leakage')
  state=fit_update(cache[old],[rows[i] for i in old],cache[new],[rows[i] for i in new],dims,recipe['proximity'],recipe['new_weight'])
  for i,c in zip(held,codes(cache[held],state)):oof[i]=c
 if oof!=[r['prediction'] for r in json.loads((run/'oof_answers.json').read_text(encoding='utf-8'))]:raise ValueError('Grouped OOF differs')
 state=fit_update(cache[:728],rows[:728],cache[728:],rows[728:],dims,recipe['proximity'],recipe['new_weight']);saved=load_file(str(run/'facies_head.safetensors'))
 if any(not torch.equal(state[k],saved[k]) for k in state):raise ValueError('Freshly fitted head differs')
 if mixed_metrics(rows,oof,source)!=result['oof'] or mixed_metrics(rows,codes(cache,state),source)!=result['train']:raise ValueError('OOF/train metrics differ')
 oldm=mixed_metrics(rows[:728],oof[:728],source)['exact_six_class']
 if oldm!=result['old_oof'] or all_class_gates(oldm,result['baseline_old_oof'])!=result['training_gates']:raise ValueError('Training acceptance differs')
 refcfg=json.loads((ref/'run_config.json').read_text(encoding='utf-8'));vraw=load_file(str(Path(refcfg['reference'])/'validation_features.safetensors'))['raw_val'][:,:6144];bs=load_file(str(ref/'facies_head.safetensors'))
 expected=[r['conversations'][1]['value'] for r in val];vp,bp=codes(vraw,state),codes(vraw,bs);vm,bm=source.classification_metrics(expected,vp),source.classification_metrics(expected,bp);hp,hb=codes(hraw,state),codes(hraw,bs);hm,hbm=mixed_metrics(hold,hp,source),mixed_metrics(hold,hb,source)
 if vm!=result['val'] or bm!=result['baseline_val'] or hm!=result['external_holdout'] or hbm!=result['baseline_external_holdout']:raise ValueError('Val/new-well evaluation differs')
 hg={w:sum(c==r['allowed_codes'][0] for r,c in zip(hold,hp) if r['well']==w)>=sum(c==r['allowed_codes'][0] for r,c in zip(hold,hb) if r['well']==w) for w in sorted({r['well'] for r in hold})}
 for c in ('C','E'):hg[c+'_recall']=hm['exact_six_class']['per_class_recall'][c]>=hbm['exact_six_class']['per_class_recall'][c]
 hg['at_least_one_gain']=hm['exact_six_class']['correct']>hbm['exact_six_class']['correct']
 vg=all_class_gates(vm,bm);vg.update(accuracy=vm['correct']>=bm['correct'],no_previously_correct_case_lost=not any(b==e and c!=e for e,b,c in zip(expected,bp,vp)),train_eligible=all(result['training_gates'].values()),new_wells_eligible=all(hg.values()))
 if vg!=result['validation_gates'] or hg!=result['new_well_gates']:raise ValueError('Promotion gates differ')
 audit=dict(status=result['status'],head_sha256=result['head_sha256'],source_pdfs_verified=True,reviewed_crops_reconstructed=len(extra+hold),
  previous804_features_unchanged=True,fresh_qwen_feature_max_delta=feature_delta,embedded_geus_cmyk_profile_verified=profiles[0],
  grouped_fits_and_all_metrics_reproduced=True,train_held_wells_disjoint=True,train_val_held_exact_file_and_pixel_duplicates=0,
  validation_gates=vg,new_well_gates=hg,distribution_approved=False,installed_in_application=False,prepared_test_pixels_opened=0,audit_script_sha256=digest(__file__),
  limits='Held16 are correlated crops from two wells, C/E only. West Lulu-2 shares a paper with training West Lulu-3. Not95% or92-facies evidence.')
 source.write_json(run/'audit_result.json',audit);print(json.dumps(audit,indent=2),flush=True)


if __name__=='__main__':main()
