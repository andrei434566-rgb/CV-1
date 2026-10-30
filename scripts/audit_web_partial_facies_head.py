"""Audit source crops, partial-label equations and fixed selection after training."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import train_unibo_visual_facies as source
from scripts.facies_head_v2_features import prepare_state,normalize,predict
from scripts.partial_facies_ridge import label_system,row_weights,fit_partial,mixed_metrics
from scripts.prepare_web_facies_20261031 import SOURCE_SHA,SOURCES,specifications,pixel_digest
from scripts.train_web_partial_facies_head import selection_key
from scripts.fetch_web_facies_sources import FILES


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--raw',type=Path,default=ROOT/'open_datasets/facies_web_20261031/raw')
    args=p.parse_args(); run=args.run.resolve()
    import torch
    from PIL import Image
    from safetensors.torch import load_file
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    cfg=json.loads((run/'run_config.json').read_text(encoding='utf-8'))
    result=json.loads((run/'result.json').read_text(encoding='utf-8'))
    frozen=json.loads((run/'selection_frozen.json').read_text(encoding='utf-8'))
    contract=json.loads((run/'MODEL_CONTRACT.json').read_text(encoding='utf-8'))
    for path,expected in cfg['dependencies'].items():
        if source.digest(path)!=expected: raise ValueError('Training dependency changed: '+path)
    if source.digest(run/'facies_head.safetensors')!=result['head_sha256']: raise ValueError('Head changed')
    if source.digest(run/'selection_frozen.json')!=result['selection_frozen_sha256']: raise ValueError('Selection changed')
    if max(frozen['candidates'],key=selection_key)!=frozen['winner']: raise ValueError('Selection rule differs')
    rows=source.read_rows(run/'train.jsonl'); val=source.read_rows(run/'val.jsonl')
    if [r['record_id'] for r in rows]!=frozen['training_record_ids']: raise ValueError('Feature row order changed')
    if val!=source.read_rows(Path(cfg['reference'])/'val.jsonl'): raise ValueError('Previous validation changed')
    source_images={}
    if source.digest(args.raw/'sydney_pdf.pdf')!=FILES['sydney_pdf.pdf'][1]: raise ValueError('Original source PDF changed')
    for filename,expected in SOURCE_SHA.items():
        path=args.raw/filename
        if source.digest(path)!=expected: raise ValueError('Raw figure changed')
        with Image.open(path) as opened: source_images[filename]=opened.convert('RGB')
    specs=specifications(); extra=[r for r in rows if r['external']]
    if len(extra)!=len(specs): raise ValueError('External crop count differs')
    for r,spec in zip(extra,specs):
        if (r['source_box']!=spec['box'] or r['source_file']!=spec['file'] or r['allowed_codes']!=spec['allowed_codes']
            or r['source']!=SOURCES[spec['source']] or r['source_group']!=SOURCES[spec['source']]['source_group']): raise ValueError('Caption localization changed')
        crop=source_images[spec['file']].crop(spec['box'])
        if pixel_digest(crop)!=r['pixel_sha256']: raise ValueError('Original source crop not reproduced')
    all_hashes=set(); val_hashes={r['metadata']['image_sha256'] for r in val}
    for r in rows:
        if source.digest(r['image'])!=r['image_sha256']: raise ValueError('Train image changed')
        if r['image_sha256'] in all_hashes or r['image_sha256'] in val_hashes: raise ValueError('Image duplication/leakage')
        all_hashes.add(r['image_sha256'])
    raw=load_file(str(run/'features.safetensors'))['train']; state=load_file(str(run/'facies_head.safetensors'))
    old_cache=load_file(str(Path(cfg['reference'])/'train_features.safetensors'))['qwen_train']
    if not torch.equal(raw[:len(old_cache)],old_cache): raise ValueError('Previous train features differ')
    spec=('qwen_post',contract['qwen_feature_dimension'],contract['pre_merger_feature_dimension'],contract['texture_dimension'])
    rebuilt=prepare_state(raw,*spec)
    if any(not torch.equal(state[k],v) for k,v in rebuilt.items()): raise ValueError('Train-only normalization differs')
    selected=frozen['winner']; allowed=[r['allowed_codes'] for r in rows]
    w=row_weights(allowed,[r['external'] for r in rows],selected['external_weight'])
    y,precision,rotation=label_system(allowed); x=normalize(raw,state); beta=state['coefficients']@rotation
    wp=w[:,None]*precision; rhs=x.T@(wp*y)
    residual=x.T@(wp*(x@beta))+selected['alpha']*beta-rhs
    relative_residual=float(residual.norm()/rhs.norm())
    if relative_residual>1e-10: raise ValueError('Partial ridge normal equations not satisfied')
    assignments=[frozen['fold_source_groups'][r['source_group']] for r in rows]
    oof=['']*len(rows)
    for fold in range(3):
        mask=torch.tensor([f!=fold for f in assignments]); held=~mask
        sub=[r for r,keep in zip(rows,mask.tolist()) if keep]
        if {r['source_group'] for r in sub}&{r['source_group'] for r,keep in zip(rows,held.tolist()) if keep}: raise ValueError('Group leakage')
        fold_state=prepare_state(raw[mask],*spec)
        a=[r['allowed_codes'] for r in sub]; weights=row_weights(a,[r['external'] for r in sub],selected['external_weight'])
        fold_state['coefficients']=fit_partial(normalize(raw[mask],fold_state),a,weights,selected['alpha'])
        codes=[source.CODES[i] for i in predict(raw[held],fold_state).argmax(1).tolist()]
        for i,c in zip(torch.where(held)[0].tolist(),codes): oof[i]=c
    if mixed_metrics(rows,oof,source)!=result['oof']: raise ValueError('OOF results not reproduced')
    saved=json.loads((run/'oof_answers.json').read_text(encoding='utf-8'))
    if [r['prediction'] for r in saved]!=oof: raise ValueError('OOF answers differ')
    vraw=load_file(str(Path(cfg['reference'])/'validation_features.safetensors'))['raw_val'][:,:contract['qwen_feature_dimension']]
    ref_cfg=json.loads((Path(cfg['reference'])/'run_config.json').read_text(encoding='utf-8'))
    first_val=load_file(str(Path(ref_cfg['reference'])/'features.safetensors'))['val']
    if not torch.equal(vraw,first_val): raise ValueError('Original validation features differ')
    vpred=[source.CODES[i] for i in predict(vraw,state).argmax(1).tolist()]
    actual=source.classification_metrics([r['conversations'][1]['value'] for r in val],vpred)
    if actual!=result['val']: raise ValueError('Validation metrics not reproduced')
    m=actual; rec=m['per_class_recall']; criteria=cfg['criteria']
    gates={'accuracy':m['correct']>=criteria['minimum_correct'],'macro_f1':m['macro_f1']>=criteria['macro_f1_minimum'],
        'all_six_recalled':all(v>0 for v in rec.values()),'prodelta':rec['C']*10>=criteria['prodelta_min_correct'],
        'PDF':rec['B']*8>=criteria['PDF_min_correct'],'FS':rec['E']*12>=criteria['FS_min_correct']}
    if gates!=result['gates']: raise ValueError('Prospective gates differ')
    status='accepted_source_development_research_only' if all(gates.values()) else 'not_promoted_failed_prospective_gates'
    audit={'status':status,'source_crops_reconstructed_exactly':True,'generic_floodplain_not_given_hard_subclass':True,
        'group_oof_and_winner_reproduced':True,'train_only_normalizer_reproduced':True,
        'partial_normal_equation_relative_residual':relative_residual,'head_sha256':result['head_sha256'],
        'audit_script_sha256':source.digest(__file__),
        'source_file_sha256':SOURCE_SHA,'val_correct':actual['correct'],'val_total':actual['total'],'val_macro_f1':actual['macro_f1'],
        'external_source_oof':result['oof']['external_caption_consistency'],'research_gates':gates,
        'installed_in_application':False,'geologist_approved':False,'general_chat_weights_changed':False,'prepared_test_pixels_opened':0,
        'validation_note':'Repeated development50, three correlated S-proxy groups; external OOF is a small correlated diagnostic, not general field accuracy.'}
    source.write_json(run/'audit_result.json',audit); source.write_json(run/'acceptance_review.json',audit)
    contract['feature_spec']={'encoder':'Original frozen Qwen3-VL-2B, no vision or language adapter',
        'input':'One RGB core crop, thumbnail side300 before original processor',
        'pooling':'Pre/post merger token mean and population std;6144 raw features',
        'selected':'Post merger mean/std, columns2048:6144,4096 values',
        'normalization':'Train-only feature mean/std floor1e-3, L2, appended intercept',
        'texture':'Not selected; reserved dimensions retain v2 inference compatibility'}
    source.write_json(run/'MODEL_CONTRACT.json',contract)
    print(json.dumps(audit,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__': main()
