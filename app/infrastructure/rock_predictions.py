"""Pure interpretation of discriminative scores in an explicit native vocabulary."""
from __future__ import annotations
import math
from app.infrastructure.vision_models import LABELS, LABELS_RU
from app.domain.rock_types import NATIVE_NAMES, TARGET_NAMES


def summarize_predictions(probabilities, card):
    classes=tuple(card['classes'])
    names=dict(zip(LABELS,LABELS_RU)) if classes==LABELS else NATIVE_NAMES
    background='garbage' if classes==LABELS else 'background'
    bg=classes.index(background)
    for row in probabilities:
        if len(row)!=len(classes) or any(not math.isfinite(v) or not 0<=v<=1 for v in row) or abs(sum(row)-1)>.001:
            raise ValueError('Некорректный вектор оценок классификатора.')
    mean=[sum(p[c] for p in probabilities)/len(probabilities) if probabilities else 0. for c in range(len(classes))]
    predictions=sorted([{'class':c,'name':names[c],'score':mean[i]} for i,c in enumerate(classes)],key=lambda r:-r['score'])
    if not probabilities:predictions=[]
    rock=[p for p in probabilities if max(range(len(p)),key=lambda i:p[i])!=bg]
    support=len(rock)/max(1,len(probabilities))
    ranked=sorted((i for i in range(len(classes)) if i!=bg),key=lambda c:-mean[c])
    index=ranked[0]; confidence=mean[index];margin=confidence-mean[ranked[1]]
    low=(not rock or support<.5 or mean[bg]>=.5 or confidence<card.get('review_threshold',.8)
         or margin<card.get('margin_threshold',.15))
    target=TARGET_NAMES.get(classes[index]) if classes!=LABELS else names[classes[index]]
    candidate = (target or names[classes[index]].split(' · ', 1)[0]) if probabilities else ''
    if candidate and classes[index] == 'mudrock' and low:
        candidate = 'Глинистая порода'
    coal_unvalidated=classes[index]=='coal' and not card.get('coal_core_validated',False)
    if coal_unvalidated:low=True
    if not rock or support<.5 or mean[bg]>=.5:
        lithology='Литология не установлена';detail='Недостаточно надёжных фрагментов породы.'
    elif low:
        lithology='Литология требует проверки'
        detail='Возможные варианты: '+names[classes[ranked[0]]].lower()+' или '+names[classes[ranked[1]]].lower()+'.'
        if coal_unvalidated:detail+=' Угольный класс не проверен на керне и автоматически не назначается.'
    elif target:
        lithology='Предположительно: '+target;detail=''
    else:
        lithology=names[classes[index]];detail='Эта группа не заменяет отдельный класс из вашего справочника.'
    if not low and classes!=LABELS:
        detail+=' '+{'mudrock':'Исходная группа глинистых пород шире аргиллита; это название требует подтверждения.',
                     'crystalline':'Положение фундамента по фотографии не устанавливается; нужны данные разреза. Эта широкая группа включает также железистые породы BIF.',
                     'carbonaceous_siltstone':'Углистый алевролит не равен углистому аргиллиту.',
                     'breccia_other':'Брекчия не равна гравелиту.',
                     'coal':'Класс обучен на угольных текстурах; перенос на фотографии керна требует проверки.'}.get(classes[index],'')
    return dict(predictions=predictions,lithology=lithology,detail=detail.strip(),confidence=confidence,
                margin=margin,low=low or not target,rock_support=support,native_class=classes[index],
                candidate=candidate)
