"""Prioritised human review, independent of training and model probabilities."""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path

from app.infrastructure.description_review import review_description
from app.infrastructure.description_template import UNKNOWN_FACIES, observation, read_assumptions
from app.infrastructure.expert_corrections import current_description_is_confirmed

ACK_KEY='__review_queue_ack'


def _json(value,default):
    try:
        parsed=json.loads(value)
        return parsed if isinstance(parsed,type(default)) else default
    except (ValueError,TypeError):return default


@dataclass
class ReviewItem:
    record: object
    detection: object | None
    reasons: list[dict]
    fingerprint: str
    acknowledged: bool

    @property
    def key(self):return (self.record.identifier,id(self.detection) if self.detection is not None else None)
    @property
    def severity(self):return max((r['severity'] for r in self.reasons),default=1)
    @property
    def interval(self):
        if self.detection is None:return 'Фото целиком'
        top,base=self.detection.depth_from,self.detection.depth_to
        return f'{top:g}–{base:g} м' if top is not None and base is not None else 'Глубина не задана'


def _quality_reasons(report):
    if not isinstance(report,dict):return []
    result=[]
    for row in report.get('issues',[]):
        if not isinstance(row,dict) or not row.get('message'):continue
        try:severity=max(1,min(3,int(row.get('severity',1))))
        except (ValueError,TypeError):severity=1
        result.append({'code':'photo:'+str(row.get('code','quality')),'message':str(row['message']),'severity':severity})
    return result


def _item(record,detection,reasons):
    if not reasons:return None
    attrs=dict(detection.attributes) if detection else {}
    attrs.pop(ACK_KEY,None)
    payload={'record':record.identifier,'reasons':reasons,
             'photo_sample':record.photo_quality.get('sample_sha256'),
             'photo_file':record.photo_quality.get('file_signature'),
             'label':detection.label if detection else '',
             'geometry_confidence':detection.confidence if detection else None,
             'attributes':attrs,
             'depth':[detection.depth_from,detection.depth_to] if detection else [],
             'polygon':[[p.x(),p.y()] for p in detection.polygon] if detection else []}
    encoded=json.dumps(payload,sort_keys=True,ensure_ascii=False,default=str)
    fingerprint=hashlib.sha256(encoded.encode('utf-8')).hexdigest()
    ack=detection.attributes.get(ACK_KEY) if detection else record.photo_review_ack
    return ReviewItem(record,detection,reasons,fingerprint,ack==fingerprint)


def build_interval_review_queue(records):
    items=[]
    for record in records:
        photo_reasons=_quality_reasons(record.photo_quality)
        if not record.photo_quality:
            photo_reasons.append({'code':'quality_pending','severity':1,'message':'Качество фото ещё не проверено.'})
        if not Path(record.path).is_file():
            photo_reasons.append({'code':'source_missing','severity':3,'message':'Исходный файл фото недоступен.'})
        elif record.photo_quality.get('file_signature'):
            from app.infrastructure.photo_quality import file_signature
            try:
                if file_signature(record.path)!=record.photo_quality['file_signature']:
                    photo_reasons.append({'code':'photo_changed','severity':2,'message':'Файл фото изменился после проверки качества. Обновите проверку.'})
            except OSError:pass
        for issue in record.analysis_issues:
            if isinstance(issue,dict) and issue.get('message'):
                photo_reasons.append({'code':'geometry_issue','severity':2,'message':str(issue['message'])})
        if not record.detections:
            photo_reasons.append({'code':'no_intervals','severity':1,'message':'На фото нет выделенных интервалов.'})
        row=_item(record,None,photo_reasons)
        if row:items.append(row)
        for detection in record.detections:
            attrs=detection.attributes;reasons=[]
            def add(code,message,severity=1):reasons.append({'code':code,'message':message,'severity':severity})
            if attrs.get('__interpretation_status')=='error':add('description_error',attrs.get('Статус интерпретации') or 'Описание не получено.',3)
            if len(detection.polygon)<3:add('invalid_polygon','Контур содержит меньше трёх точек.',3)
            elif not all(math.isfinite(p.x()) and math.isfinite(p.y()) for p in detection.polygon):add('invalid_polygon','Некорректные координаты контура.',3)
            whole=attrs.get('__core_column_description')=='1'
            if whole and not record.core_columns_verified:add('column_bounds','Рамка целого столбика ещё не проверена.')
            elif attrs.get('__geometry_only')=='1' and not detection.training_ready and detection.confidence<.5:
                add('geometry_score','Низкая оценка геометрии YOLO; проверьте границы.',2)
            confirmed=current_description_is_confirmed(detection)
            if not confirmed and attrs.get('__vision_lithology_review_required')=='1':add('lithology_score','Классификатор неуверенно определил породу.',2)
            if str(detection.label).casefold() in UNKNOWN_FACIES:add('facies_missing','Фация не указана.')
            if 'Название фации' in read_assumptions(attrs):add('facies_assumed','Фация указана предположительно.',2)
            missing=[field for field in ('Название породы','Зернистость','Слоистость','Контакт') if not observation(attrs.get(field))[0]]
            if missing:add('missing_fields','Не заполнены признаки: '+', '.join(missing)+'.')
            if not str(attrs.get('Краткое описание') or attrs.get('__vlm_description') or '').strip():add('description_missing','Текст описания отсутствует.',2)
            if detection.depth_from is None or detection.depth_to is None:add('depth_missing','Глубины не заданы; это не блокирует сохранение.')
            elif not all(math.isfinite(float(v)) for v in (detection.depth_from,detection.depth_to)) or detection.depth_to<=detection.depth_from:
                add('depth_invalid','Проверьте порядок и значения глубин.',3)
            reasons.extend({'code':'description:'+h.code,'message':h.message,'severity':2} for h in review_description(attrs))
            reasons.extend(_quality_reasons(_json(attrs.get('__image_quality'),{})))
            row=_item(record,detection,reasons)
            if row:items.append(row)
    return sorted(items,key=lambda item:(-item.severity,-len(item.reasons),item.record.well_name,Path(item.record.path).name,item.interval))


def acknowledge(item):
    """Only review of current warnings; no training or geological confirmation."""
    if item.detection is None:item.record.photo_review_ack=item.fingerprint
    else:item.detection.attributes[ACK_KEY]=item.fingerprint
