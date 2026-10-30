"""Verify core-only provenance, preservation selection and proximal equations."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import train_unibo_visual_facies as source
from scripts.prepare_core_facies_20261031_v6 import FILES,PHOTOS,SOURCES,specs,pixel_digest
from scripts.prepare_target_facies_20261031 import PHOTO_FILES as PREVIOUS_PHOTOS
from scripts.conservative_facies_update import selection_key,training_gates,preservation,PROTECTED
from scripts.train_core_facies_conservative import make_state,fit_update,new_weights,codes
from scripts.partial_facies_ridge import label_system,mixed_metrics
from scripts.facies_head_v2_features import normalize


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    p.add_argument('--raw',type=Path,default=ROOT/'open_datasets/target_core_facies_20261031_v6/raw')
    p.add_argument('--previous-raw',type=Path,default=ROOT/'open_datasets/target_facies_20261031_v4/raw');a=p.parse_args();out=a.run.resolve()
    import torch
    from safetensors.torch import load_file
    from PIL import Image
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    cfg=json.loads((out/'run_config.json').read_text(encoding='utf-8'));result=json.loads((out/'result.json').read_text(encoding='utf-8'))
    frozen=json.loads((out/'selection_frozen.json').read_text(encoding='utf-8'));contract=json.loads((out/'MODEL_CONTRACT.json').read_text(encoding='utf-8'))
    for f,h in cfg['dependencies'].items():
        if source.digest(f)!=h:raise ValueError('Training dependency changed: '+f)
    if source.digest(out/'facies_head.safetensors')!=result['head_sha256'] or source.digest(out/'selection_frozen.json')!=result['selection_frozen_sha256']:raise ValueError('Saved weights/selection changed')
    if max(frozen['candidates'],key=selection_key)!=frozen['winner']:raise ValueError('Selection rule differs')
    if cfg['distribution_approved'] or contract['distribution_approved']:raise ValueError('Unresolved source rights incorrectly granted')
    rows=source.read_rows(out/'train.jsonl');val=source.read_rows(out/'val.jsonl');old=rows[:728];new=rows[728:]
    ref=Path(cfg['reference']);cache=Path(cfg['core_cache'])
    if old!=source.read_rows(ref/'train.jsonl') or val!=source.read_rows(ref/'val.jsonl'):raise ValueError('Old rows changed')
    if rows[728:739]!=source.read_rows(cache/'train.jsonl')[728:739]:raise ValueError('Previous core rows changed')
    if [r['record_id'] for r in rows]!=frozen['training_record_ids']:raise ValueError('Row order changed')
    for n,(_,h) in FILES.items():
        if source.digest(a.raw/n)!=h:raise ValueError('Publication changed: '+n)
    images={}
    for n,h in PHOTOS.items():
        if source.digest(a.raw/n)!=h:raise ValueError('New original changed')
        with Image.open(a.raw/n) as im:images[n]=im.convert('RGB')
    for r,s in zip(rows[739:],specs()):
        if r['source']!=SOURCES[s['source_id']] or r['allowed_codes']!=[s['code']] or r['source_box']!=s['box']:raise ValueError('Caption/box changed')
        if pixel_digest(images[s['file']].crop(s['box']))!=r['pixel_sha256']:raise ValueError('New crop not reproduced')
    for r in rows[728:739]:
        name=r['source_file']
        if source.digest(a.previous_raw/name)!=PREVIOUS_PHOTOS[name][3]:raise ValueError('Previous original changed')
        with Image.open(a.previous_raw/name) as im:crop=im.convert('RGB').crop(r['source_box'])
        if pixel_digest(crop)!=r['pixel_sha256']:raise ValueError('Previous core crop not reproduced')
    seen=set()
    for r in rows+val:
        if 'outcrop' in r.get('domain',''):raise ValueError('Outcrop in core experiment')
        h=r.get('image_sha256',r.get('metadata',{}).get('image_sha256'))
        if source.digest(r['image'])!=h or h in seen:raise ValueError('Duplicate or changed image')
        seen.add(h)
    raw=load_file(str(out/'features.safetensors'))['train'];state=load_file(str(out/'facies_head.safetensors'))
    if not torch.equal(raw[:728],load_file(str(ref/'features.safetensors'))['train']):raise ValueError('Old features changed')
    if not torch.equal(raw[728:739],load_file(str(cache/'features.safetensors'))['train'][728:739]):raise ValueError('Previous core features changed')
    dims=(contract['qwen_feature_dimension'],contract['pre_merger_feature_dimension'],contract['texture_dimension'])
    base,w=make_state(raw[:728],old,dims)
    for k,v in base.items():
        if k!='coefficients' and not torch.equal(state[k],v):raise ValueError('Normalizer is not old-train-only')
    prior=base['coefficients'];winner=frozen['winner'];pr,mult=winner['proximity'],winner['new_weight']
    x=normalize(raw,state);allowed=[r['allowed_codes'] for r in rows];y,precision,rotation=label_system(allowed)
    weights=torch.cat((w,new_weights(old,new,mult)))[:,None]*precision;b=state['coefficients']@rotation;b0=prior@rotation
    rhs=x.T@(weights*y)+pr*b0;residual=x.T@(weights*(x@b))+b+pr*(b-b0)-x.T@(weights*y)
    relative=float(residual.norm()/rhs.norm())
    if relative>1e-10:raise ValueError('Proximal normal equations not satisfied')
    mapping=frozen['fold_source_groups'];oof=['']*len(rows);base_oof=['']*728
    for f in range(3):
        om=torch.tensor([mapping[r['source_group']]!=f for r in old]);nm=torch.tensor([mapping[r['source_group']]!=f for r in new])
        oo=[r for r,k in zip(old,om.tolist()) if k];nn=[r for r,k in zip(new,nm.tolist()) if k]
        held=torch.tensor([mapping[r['source_group']]==f for r in rows])
        if {r['source_group'] for r in oo+nn}&{r['source_group'] for r,k in zip(rows,held.tolist()) if k}:raise ValueError('Group leakage')
        s=fit_update(raw[:728][om],oo,raw[728:][nm],nn,dims,pr,mult)
        for i,c in zip(torch.where(held)[0].tolist(),codes(raw[held],s)):oof[i]=c
        bs,_=make_state(raw[:728][om],oo,dims)
        for i,c in zip(torch.where(~om)[0].tolist(),codes(raw[:728][~om],bs)):base_oof[i]=c
    bm=mixed_metrics(old,base_oof,source)['exact_six_class'];cm=mixed_metrics(old,oof[:728],source)['exact_six_class']
    if bm!=frozen['baseline_old_train_oof'] or cm!=winner['old_train_oof'] or training_gates(cm,bm)!=winner['training_gates']:raise ValueError('Old-only comparison not reproduced')
    if mixed_metrics(rows,oof,source)!=winner['all_rows_oof']:raise ValueError('All-source OOF differs')
    saved=json.loads((out/'oof_answers.json').read_text(encoding='utf-8'))
    if [r['prediction'] for r in saved]!=oof:raise ValueError('OOF answers differ')
    refcfg=json.loads((ref/'run_config.json').read_text(encoding='utf-8'))
    vraw=load_file(str(Path(refcfg['reference'])/'validation_features.safetensors'))['raw_val'][:,:dims[0]]
    target=[r['conversations'][1]['value'] for r in val];vp=codes(vraw,state);bp=codes(vraw,load_file(str(ref/'facies_head.safetensors')))
    vm=source.classification_metrics(target,vp);vb=source.classification_metrics(target,bp)
    if vm!=result['val'] or vb!=result['baseline_val']:raise ValueError('Val not reproduced')
    checks=preservation(vm,vb);checks.update(B_improved=vm['per_class_recall']['B']>vb['per_class_recall']['B'],C_improved=vm['per_class_recall']['C']>vb['per_class_recall']['C'],accuracy=vm['correct']>=vb['correct'],macro_f1=vm['macro_f1']+1e-12>=vb['macro_f1'])
    lost=sum(e in PROTECTED and b==e and c!=e for e,b,c in zip(target,bp,vp))
    checks.update(no_protected_correct_case_lost=lost==0,train_eligible=all(winner['training_gates'].values()))
    if checks!=result['validation_gates'] or lost!=result['introduced_protected_errors']:raise ValueError('Preservation gates differ')
    new_sources={g:dict(total=sum(r['source_group']==g for r in rows[739:]),correct=sum(r['source_group']==g and c in r['allowed_codes'] for r,c in zip(rows[739:],oof[739:]))) for g in sorted({r['source_group'] for r in rows[739:]})}
    audit=dict(status=result['status'],head_sha256=result['head_sha256'],new_source_oof=new_sources,
        new_and_previous_core_crops_reproduced=True,outcrop_crops=0,old_features_unchanged=True,
        normalization_old_train_only=True,cross_fitted_prior_and_selected_oof_reproduced=True,proximal_relative_residual=relative,
        validation_gates=checks,old_train_oof_gates=winner['training_gates'],val_correct=vm['correct'],val_total=vm['total'],
        distribution_approved=False,installed_in_application=False,encoder_or_chat_weights_changed=False,prepared_test_pixels_opened=0,
        audit_script_sha256=source.digest(__file__),limitations='Repeated50 development crops. Salpi and other Italian source provenance may overlap anonymized regions; no claim of independent wells or final accuracy.')
    source.write_json(out/'audit_result.json',audit);print(json.dumps(audit,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
