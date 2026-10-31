"""Reconstruct v7 crop provenance, fixed-recipe equations, OOF and gates."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import train_unibo_visual_facies as source
from scripts.prepare_core_contrasts_20261031_v7 import PHOTOS,PDFS,SOURCES,TARGETS,specifications,pixel_digest
from scripts.train_core_facies_conservative import make_state,fit_update,codes,new_weights
from scripts.conservative_facies_update import training_gates,preservation,PROTECTED
from scripts.partial_facies_ridge import label_system,mixed_metrics
from scripts.facies_head_v2_features import normalize


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--raw',type=Path,default=ROOT/'open_datasets/core_materials_20261031_v7/raw');a=p.parse_args();run=a.run.resolve()
 import torch
 from PIL import Image
 from safetensors.torch import load_file
 torch.set_num_threads(4);torch.set_num_interop_threads(1)
 cfg=json.loads((run/'run_config.json').read_text(encoding='utf-8'));result=json.loads((run/'result.json').read_text(encoding='utf-8'));frozen=json.loads((run/'selection_frozen.json').read_text(encoding='utf-8'));contract=json.loads((run/'MODEL_CONTRACT.json').read_text(encoding='utf-8'))
 for f,h in cfg['dependencies'].items():
  if source.digest(f)!=h:raise ValueError('Dependency differs: '+f)
 if source.digest(run/'facies_head.safetensors')!=result['head_sha256'] or source.digest(run/'selection_frozen.json')!=result['selection_frozen_sha256']:raise ValueError('Weights or recipe changed')
 rows=source.read_rows(run/'train.jsonl');val=source.read_rows(run/'val.jsonl');prev=Path(cfg['previous']);ref=Path(cfg['reference'])
 if rows[:752]!=source.read_rows(prev/'train.jsonl') or val!=source.read_rows(ref/'val.jsonl'):raise ValueError('Prior data/val changed')
 if len(rows)!=804 or [r['record_id'] for r in rows]!=frozen['training_record_ids']:raise ValueError('Feature row order differs')
 for n,h in PDFS.items():
  if source.digest(a.raw/n)!=h:raise ValueError('Reviewed PDF differs')
 images={}
 for n,h in PHOTOS.items():
  if source.digest(a.raw/n)!=h:raise ValueError('Original photo differs')
  with Image.open(a.raw/n) as im:images[n]=im.convert('RGB')
 specs=specifications()
 for r,s in zip(rows[752:],specs):
  target=TARGETS[s['target']];meta=SOURCES[target['source']]
  if r['source']!=meta or r['allowed_codes']!=[target['code']] or r['well']!=target['well'] or r['photographed_interval_m']!=target['interval'] or r['crop_exact_depth_m'] is not None:raise ValueError('Source interpretation changed')
  if r['source_box']!=s['box'] or pixel_digest(images[s['file']].crop(s['box']))!=r['pixel_sha256']:raise ValueError('New crop not reproduced')
  if r['distribution_approved']!=meta['distribution_approved'] or r['geologist_approved']:raise ValueError('Approval scope changed')
 for i,s in enumerate(specs):
  for t in specs[i+1:]:
   if s['file']!=t['file']:continue
   x,y,xx,yy=s['box'];u,v,uu,vv=t['box']
   if max(0,min(xx,uu)-max(x,u))*max(0,min(yy,vv)-max(y,v))>0:raise ValueError('Overlapping source crops')
 seen=set()
 for r in rows+val:
  h=r.get('image_sha256',r.get('metadata',{}).get('image_sha256'))
  if source.digest(r['image'])!=h or h in seen:raise ValueError('Duplicate/changed image')
  if 'outcrop' in r.get('domain',''):raise ValueError('Unexpected outcrop')
  seen.add(h)
 raw=load_file(str(run/'features.safetensors'))['train'];state=load_file(str(run/'facies_head.safetensors'))
 if not torch.equal(raw[:752],load_file(str(prev/'features.safetensors'))['train']):raise ValueError('Previous features differ')
 old,new=rows[:728],rows[728:];oldraw,newraw=raw[:728],raw[728:]
 dims=(contract['qwen_feature_dimension'],contract['pre_merger_feature_dimension'],contract['texture_dimension']);base,w=make_state(oldraw,old,dims)
 for k,v in base.items():
  if k!='coefficients' and not torch.equal(state[k],v):raise ValueError('Old-only normalizer differs')
 recipe=cfg['recipe'];pr,mult=recipe['proximity'],recipe['new_weight'];allowed=[r['allowed_codes'] for r in rows];y,precision,rotation=label_system(allowed)
 x=normalize(raw,state);wp=torch.cat((w,new_weights(old,new,mult)))[:,None]*precision;b=state['coefficients']@rotation;b0=base['coefficients']@rotation
 rhs=x.T@(wp*y)+pr*b0;residual=x.T@(wp*(x@b))+b+pr*(b-b0)-x.T@(wp*y);relative=float(residual.norm()/rhs.norm())
 if relative>1e-10:raise ValueError('Normal equations not satisfied')
 mapping=frozen['fold_source_groups'];oof=['']*len(rows);boof=['']*728
 for fold in range(3):
  om=torch.tensor([mapping[r['source_group']]!=fold for r in old]);nm=torch.tensor([mapping[r['source_group']]!=fold for r in new]);held=torch.tensor([mapping[r['source_group']]==fold for r in rows])
  oo=[r for r,k in zip(old,om.tolist()) if k];nn=[r for r,k in zip(new,nm.tolist()) if k]
  if {r['source_group'] for r in oo+nn}&{r['source_group'] for r,k in zip(rows,held.tolist()) if k}:raise ValueError('Group leakage')
  fs=fit_update(oldraw[om],oo,newraw[nm],nn,dims,pr,mult)
  for i,c in zip(torch.where(held)[0].tolist(),codes(raw[held],fs)):oof[i]=c
  bs,_=make_state(oldraw[om],oo,dims)
  for i,c in zip(torch.where(~om)[0].tolist(),codes(oldraw[~om],bs)):boof[i]=c
 oldm=mixed_metrics(old,oof[:728],source)['exact_six_class'];basem=mixed_metrics(old,boof,source)['exact_six_class']
 if oldm!=result['old_train_oof'] or basem!=result['baseline_old_train_oof'] or training_gates(oldm,basem)!=result['training_gates']:raise ValueError('Old-source comparison differs')
 if mixed_metrics(rows,oof,source)!=result['oof'] or [r['prediction'] for r in json.loads((run/'oof_answers.json').read_text(encoding='utf-8'))]!=oof:raise ValueError('OOF differs')
 rcfg=json.loads((ref/'run_config.json').read_text(encoding='utf-8'));vraw=load_file(str(Path(rcfg['reference'])/'validation_features.safetensors'))['raw_val'][:,:dims[0]]
 expected=[r['conversations'][1]['value'] for r in val];vp=codes(vraw,state);bp=codes(vraw,load_file(str(ref/'facies_head.safetensors')))
 vm=source.classification_metrics(expected,vp);bm=source.classification_metrics(expected,bp)
 gates=preservation(vm,bm);gates.update(B_improved=vm['per_class_recall']['B']>bm['per_class_recall']['B'],C_improved=vm['per_class_recall']['C']>bm['per_class_recall']['C'],accuracy=vm['correct']>=bm['correct'],macro_f1=vm['macro_f1']+1e-12>=bm['macro_f1'],train_eligible=all(result['training_gates'].values()),no_protected_correct_case_lost=not any(e in PROTECTED and b==e and c!=e for e,b,c in zip(expected,bp,vp)))
 if vm!=result['val'] or bm!=result['baseline_val'] or gates!=result['validation_gates']:raise ValueError('Validation or gates differ')
 audit=dict(status=result['status'],head_sha256=result['head_sha256'],source_crops_reproduced=52,original_pdf_and_photo_hashes_verified=True,
  previous_features_unchanged=True,old_only_normalizer_reproduced=True,group_oof_and_validation_reproduced=True,proximal_relative_residual=relative,
  training_gates=result['training_gates'],validation_gates=gates,val_correct=vm['correct'],val_total=vm['total'],
  new_source_group_oof={g:dict(total=sum(r['source_group']==g for r in rows[752:]),correct=sum(r['source_group']==g and c in r['allowed_codes'] for r,c in zip(rows[752:],oof[752:]))) for g in sorted({r['source_group'] for r in rows[752:]})},
  prepared_test_pixels_opened=0,distribution_approved=False,installed_in_application=False,general_chat_weights_changed=False,
  audit_script_sha256=source.digest(__file__),limitation='Repeated development50; source/proxy geographic overlap not ruled out; no independent field accuracy claimed')
 source.write_json(run/'audit_result.json',audit);print(json.dumps(audit,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
