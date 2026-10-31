"""Verify the selected RBF head, RKHS equations, group OOF and fixed gates."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import train_unibo_visual_facies as source
from scripts.facies_head_v2_features import prepare_state,normalize,predict
from scripts.partial_facies_ridge import label_system,mixed_metrics
from scripts.kernel_facies_head import kernel,training_bandwidth,partial_dual,grouped_weights,all_class_gates,selection_key,predict_features,validate_state


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);a=p.parse_args();out=a.run.resolve()
 import torch
 from safetensors.torch import load_file
 torch.set_num_threads(4);torch.set_num_interop_threads(1)
 cfg=json.loads((out/'run_config.json').read_text(encoding='utf-8'));result=json.loads((out/'result.json').read_text(encoding='utf-8'));frozen=json.loads((out/'selection_frozen.json').read_text(encoding='utf-8'));contract=json.loads((out/'MODEL_CONTRACT.json').read_text(encoding='utf-8'))
 for f,h in cfg['dependencies'].items():
  if source.digest(f)!=h:raise ValueError('Dependency changed: '+f)
 if source.digest(out/'facies_head.safetensors')!=result['head_sha256'] or source.digest(out/'selection_frozen.json')!=result['selection_frozen_sha256']:raise ValueError('Weights or selection changed')
 if max(frozen['candidates'],key=selection_key)!=frozen['winner']:raise ValueError('Selection differs')
 prev,ref=Path(cfg['previous']),Path(cfg['reference']);rows=source.read_rows(out/'train.jsonl');val=source.read_rows(out/'val.jsonl')
 if rows!=source.read_rows(prev/'train.jsonl') or val!=source.read_rows(ref/'val.jsonl') or [r['record_id'] for r in rows]!=frozen['training_record_ids']:raise ValueError('Row order/val changed')
 raw=load_file(str(prev/'features.safetensors'))['train'];state=load_file(str(out/'facies_head.safetensors'));validate_state(state,contract)
 rebuilt=prepare_state(raw,'qwen_post',6144,2048,114)
 for k,v in rebuilt.items():
  if not torch.equal(state[k],v):raise ValueError('Train normalizer differs')
 x=normalize(raw,state)[:,:-1]
 if not torch.equal(x,state['anchors']) or not torch.equal(training_bandwidth(x),state['bandwidth']):raise ValueError('Anchors or bandwidth differ')
 winner=frozen['winner'];alpha,factor,mode=(winner[k] for k in ('alpha','factor','weighting'))
 gram=kernel(x,x,state['bandwidth'],factor);y,precision,rotation=label_system([r['allowed_codes'] for r in rows]);w=grouped_weights(rows,mode,cfg['recipe']['external_weight'])[:,None]*precision;dual=state['dual']@rotation
 residual=w*(gram@dual-y)+alpha*dual;relative=float(residual.norm()/(w*y).norm())
 if relative>1e-10:raise ValueError('RBF normal equations not satisfied')
 oof=['']*len(rows);mapping=frozen['fold_source_groups']
 for fold in range(3):
  mask=torch.tensor([mapping[r['source_group']]!=fold for r in rows]);held=~mask;sub=[r for r,k in zip(rows,mask.tolist()) if k]
  if {r['source_group'] for r in sub}&{r['source_group'] for r,k in zip(rows,held.tolist()) if k}:raise ValueError('Group leakage')
  s=prepare_state(raw[mask],'qwen_post',6144,2048,114);xx=normalize(raw[mask],s)[:,:-1];xh=normalize(raw[held],s)[:,:-1];band=training_bandwidth(xx)
  a=partial_dual(kernel(xx,xx,band,factor),[r['allowed_codes'] for r in sub],grouped_weights(sub,mode,cfg['recipe']['external_weight']),alpha)
  codes=[source.CODES[i] for i in (kernel(xh,xx,band,factor)@a).argmax(1).tolist()]
  for i,c in zip(torch.where(held)[0].tolist(),codes):oof[i]=c
 if oof!=[r['prediction'] for r in json.loads((out/'oof_answers.json').read_text(encoding='utf-8'))] or mixed_metrics(rows,oof,source)!=result['oof']:raise ValueError('OOF differs')
 old=mixed_metrics(rows[:728],oof[:728],source)['exact_six_class']
 if old!=result['old_oof'] or all_class_gates(old,result['baseline_old_oof'])!=winner['all_class_gates']:raise ValueError('Old comparison differs')
 rcfg=json.loads((ref/'run_config.json').read_text(encoding='utf-8'));vr=load_file(str(Path(rcfg['reference'])/'validation_features.safetensors'))['raw_val'][:,:6144]
 expected=[r['conversations'][1]['value'] for r in val];vp=[source.CODES[i] for i in predict_features(vr,state).argmax(1).tolist()];bp=[source.CODES[i] for i in predict(vr,load_file(str(ref/'facies_head.safetensors'))).argmax(1).tolist()]
 vm=source.classification_metrics(expected,vp);bm=source.classification_metrics(expected,bp);gates=all_class_gates(vm,bm);gates.update(train_eligible=all(winner['all_class_gates'].values()),accuracy=vm['correct']>=bm['correct'])
 if vm!=result['val'] or bm!=result['baseline_val'] or gates!=result['validation_gates']:raise ValueError('Validation/gates differ')
 audit=dict(status=result['status'],head_sha256=result['head_sha256'],training_rows_unchanged=True,train_only_normalizer_and_bandwidth_reproduced=True,selected_group_oof_reproduced=True,
  rkhs_normal_equation_relative_residual=relative,validation_reproduced=True,validation_gates=gates,distribution_approved=False,installed_in_application=False,prepared_test_pixels_opened=0,audit_script_sha256=source.digest(__file__))
 source.write_json(out/'audit_result.json',audit);print(json.dumps(audit,indent=2))


if __name__=='__main__':main()
