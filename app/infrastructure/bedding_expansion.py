"""Research-only, positive/unlabelled extension on frozen bedding CNN features.

Absence from a publication's caption never becomes a negative geology label.
Explicitly massive source ROIs provide negative lamination labels, never
negative grading labels: a massive bed can still be normally/inversely graded.
These narrow negatives cannot establish discrimination among all bedding types.
"""
from __future__ import annotations
TYPES=('Бугорчатая косая','Волнистая','Градационная','Линзовидная',
       'Линзовидно-волнистая','Разнонаправленная косая','Флазерная')
FEATURE_CONTRACT='whole_roi_rgb_letterbox_224_gray127_imagenet_classifier_input_l2_v1'

def validate_examples(rows):
    """Reject group, well and exact pixel leakage before fitting anything."""
    import hashlib
    from pathlib import Path
    from PIL import Image
    groups={};pixels={};seen_ids=set()
    for row in rows:
        if row.get('id') in seen_ids:raise ValueError('Duplicate source example id')
        seen_ids.add(row.get('id'))
        if (row.get('positive_type') not in TYPES or row.get('split') not in ('train','val','test')
            or row.get('domain')!='photographic_core' or not row.get('source_group')
            or row.get('label_provenance')!='caption_and_spatially_matched_author_label'):
            raise ValueError('A captioned photographic core region and source group are required')
        if row['split']=='train' and row.get('research_training_approved') is not True:
            raise ValueError('Source image not approved for research fitting')
        path=Path(row['image'])
        if hashlib.sha256(path.read_bytes()).hexdigest()!=row['sha256']:raise ValueError('Changed crop')
        with Image.open(path) as im:
            im=im.convert('RGB');key=hashlib.sha256(str(im.size).encode()+im.tobytes()).hexdigest()
        group=row['source_group']
        if group in groups and groups[group]!=row['split']:raise ValueError('Source/core group crosses splits')
        groups[group]=row['split']
        well=(row['source'],row.get('well')) if row.get('well') else None
        if well:
            if well in groups and groups[well]!=row['split']:raise ValueError('Well crosses splits')
            groups[well]=row['split']
        if key in pixels:raise ValueError('Duplicate photographic crop, possibly with conflicting labels')
        pixels[key]=row['positive_type']
    return rows

def known_targets(rows):
    """One known positive, all other structures unknown; not absent."""
    import torch
    target=torch.zeros((len(rows),len(TYPES)));mask=torch.zeros_like(target)
    for i,row in enumerate(rows):
        if row.get('negative_origin')=='author_explicit_massive_structure':
            if (row.get('source')!='rf100' or row.get('label')!=2 or row.get('label_origin')!='author_yolo_bbox'
                or row.get('positive_type')):raise ValueError('Negative label requires the explicit source Massive annotation')
            mask[i]=1
            mask[i,TYPES.index('Градационная')]=0
        else:
            j=TYPES.index(row['positive_type']);target[i,j]=1;mask[i,j]=1
    return target,mask

def positive_support(rows):
    return {name:{s:{'n':sum(r.get('positive_type')==name and r['split']==s for r in rows),
                   'groups':len({r['source_group'] for r in rows if r.get('positive_type')==name and r['split']==s})}
                  for s in ('train','val','test')} for name in TYPES}

def promotion_gate(rows,*,varied_negative_examples=False,validation=None):
    """Small literature examples cannot enable automatic 16-type diagnosis."""
    from app.domain.bedding_types import PROFILES
    support=positive_support(rows);reasons=[]
    for name in (p[0] for p in PROFILES):
        record=support.get(name,{s:{'n':0,'groups':0} for s in ('train','val','test')})
        if record['train']['n']<5 or record['train']['groups']<2:reasons.append(name+': мало независимых учебных примеров')
        if record['val']['n']<3 or record['test']['n']<3:reasons.append(name+': нет достаточной независимой проверки')
    if not varied_negative_examples:reasons.append('Нужны проверенные контрпримеры других типов, не только массивная структура')
    validation=validation or {}
    if validation.get('all_16_independent_test') is not True:
        reasons.append('Не завершена независимая проверка всех 16 типов')
    if validation.get('protected_model_regression_passed') is not True:
        reasons.append('Нужен отчёт об отсутствии ухудшения остальных моделей')
    checked={v.get('type'):v for v in validation.get('per_type',[]) if isinstance(v,dict)}
    for name in (p[0] for p in PROFILES):
        m=checked.get(name,{})
        if not all(isinstance(m.get(k),(int,float)) and m[k]>=.8 for k in ('precision','recall')):
            reasons.append(name+': не подтверждены precision и recall не ниже 0.8')
    return {'eligible_for_automatic_assignment':not reasons,'reasons':reasons,'positive_support':support,
            'all_16_covered':False if len(support)<16 else not reasons}

def masked_balanced_loss(logits,target,mask):
    """Balance known positives/negatives separately; ignore every unknown cell."""
    import torch
    raw=torch.nn.functional.binary_cross_entropy_with_logits(logits,target,reduction='none');parts=[]
    for j in range(len(TYPES)):
        pos=(mask[:,j]>0)&(target[:,j]>0);neg=(mask[:,j]>0)&(target[:,j]==0)
        if pos.any() and neg.any():parts.append((raw[pos,j].mean()+raw[neg,j].mean())/2)
    if not parts:raise ValueError('Each fitted structure needs known positive and explicit negative examples')
    return torch.stack(parts).mean()
