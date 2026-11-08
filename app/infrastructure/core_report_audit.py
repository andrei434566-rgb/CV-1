"""Non-mutating report completeness and interval topology checks.

Counts describe recorded data, never model accuracy. Gaps may represent
unloaded core or real sampling breaks; overlaps may be repeated photographs.
"""
from __future__ import annotations
from dataclasses import asdict,dataclass
import hashlib
import json
import math

from app.infrastructure.description_template import REVIEW_FIELDS,UNKNOWN_FACIES,observation,read_assumptions


def finite_number(value):
    if isinstance(value,bool):return None
    try:number=float(value)
    except (TypeError,ValueError,OverflowError):return None
    return number if math.isfinite(number) else None


def named_facies(value):
    text=observation(value)[0].strip().casefold()
    return bool(text and text not in UNKNOWN_FACIES and not text.startswith(('не определ','не установ','не распозн','интервал yolo','новый контур')))


@dataclass(frozen=True)
class ReportIssue:
    code: str
    well: str
    well_index: int
    rows: tuple[int,...]
    severity: int
    message: str
    source_ids: tuple[str,...] = ()

    @property
    def key(self):
        return hashlib.sha256(json.dumps(asdict(self),ensure_ascii=False,sort_keys=True).encode('utf-8')).hexdigest()


@dataclass
class ReportAudit:
    wells: list[dict]
    totals: dict
    issues: list[ReportIssue]

    def to_dict(self):
        return {'schema':'cv1-report-audit-v1','wells':self.wells,'totals':self.totals,
                'issues':[asdict(row) for row in self.issues],
                'model_accuracy_evaluated':False,'automatic_corrections':False}


def audit_report_wells(wells,*,tolerance=.005):
    """Audit prepared report rows, keeping well/depth-reference domains separate."""
    if not math.isfinite(tolerance) or tolerance<0:raise ValueError('Некорректный допуск глубин')
    issues=[];summaries=[]
    totals={'wells':len(wells),'rows':0,'valid_depths':0,'with_text':0,'confirmed_descriptions':0,
            'named_facies':0,'assumed_facies':0,'filled_fields':0,'field_slots':0,'assumed_fields':0,
            'gaps':0,'overlaps':0}
    for wi,well in enumerate(wells):
        name=str(well.get('name') or f'Скважина {wi+1}');layers=list(well.get('layers') or [])
        summary={key:0 for key in totals if key!='wells'}
        unit=str((well.get('depth_settings') or {}).get('unit') or 'м')
        summary.update({'name':name,'rows':len(layers),'depth_reference':str(well.get('depth_reference') or 'MD'),
                        'unit':unit,'mapped_length':0.,'depth_from':None,'depth_to':None,
                        'fields':{field:{'filled':0,'assumed':0,'total':len(layers)} for field in REVIEW_FIELDS}})
        def add(code,rows,message,severity=1):
            ids=tuple(dict.fromkeys(str(source) for index in rows for source in
                (layers[index].get('source_ids') or ([layers[index]['source_id']] if layers[index].get('source_id') else []))))
            issues.append(ReportIssue(code,name,wi,tuple(rows),severity,message,ids))
        if not layers:add('no_rows',(),'Нет строк описания. Можно выгрузить пустой шаблон.')
        if unit.casefold() not in {'м','m','метры','метр','meters','metres','meter','metre'}:
            add('depth_units',tuple(range(len(layers))),f'Единицы глубин проекта: {unit}; типовой Excel подписан в метрах. Проверьте единицы и пересчёт.',2)
        valid=[]
        for index,layer in enumerate(layers):
            attrs=layer.get('attributes') or {};assumed=read_assumptions(attrs)
            top,base=finite_number(layer.get('depth_from')),finite_number(layer.get('depth_to'))
            if top is None or base is None:
                add('depth_missing',(index,),'Обе границы строки должны быть заданы конечными числами для проверки глубин.')
            elif base<=top:add('depth_invalid',(index,),'Подошва должна быть глубже кровли; исходные значения сохранены.',3)
            else:
                summary['valid_depths']+=1;valid.append((top,base,index))
            missing=[]
            for field in REVIEW_FIELDS:
                value,inline=observation(attrs.get(field))
                if value:
                    summary['fields'][field]['filled']+=1;summary['filled_fields']+=1
                    if inline or field in assumed:
                        summary['fields'][field]['assumed']+=1;summary['assumed_fields']+=1
                else:missing.append(field)
            summary['field_slots']+=len(REVIEW_FIELDS)
            if missing:add('properties_missing',(index,),'Не заполнены основные поля: '+', '.join(missing)+'.')
            description=str((layer.get('vlm') or {}).get('description') or attrs.get('__vlm_description') or attrs.get('Краткое описание') or '').strip()
            if description and description!='Признаки участка не заполнены.':
                summary['with_text']+=1
                if layer.get('description_confirmed') is True:summary['confirmed_descriptions']+=1
                else:add('description_unconfirmed',(index,),'Текст не имеет актуального подтверждения специалиста в проекте.',2)
            else:add('description_missing',(index,),'Текст описания отсутствует.',2)
            label=attrs.get('Название фации') or layer.get('label')
            if named_facies(label):
                summary['named_facies']+=1
                if 'Название фации' in assumed or observation(label)[1]:
                    summary['assumed_facies']+=1;add('facies_assumed',(index,),'Название фации указано предположительно.',2)
            else:add('facies_missing',(index,),'Фация не указана.')
            if attrs.get('__interpretation_status')=='error':
                add('description_error',(index,),str(attrs.get('Статус интерпретации') or 'Обработка описания завершилась ошибкой.'),3)
        # Sweep with the furthest covered endpoint: nested intervals must not
        # create fictitious gaps. Sorting here never changes the caller's rows.
        ordered=sorted(valid);frontier=None;owner=None;union=0.;start_union=None
        declared=well.get('core_interval')
        if not isinstance(declared,(tuple,list)) or len(declared)!=2:
            passport=well.get('excel_report_inputs') or {}
            declared=(passport.get('core_top'),passport.get('core_base'))
        declared_top=finite_number(declared[0]) if isinstance(declared,(tuple,list)) and len(declared)==2 else None
        declared_base=finite_number(declared[1]) if isinstance(declared,(tuple,list)) and len(declared)==2 else None
        declared_valid=declared_top is not None and declared_base is not None and declared_base>declared_top
        for top,base,index in ordered:
            if declared_valid and (top<declared_top-tolerance or base>declared_base+tolerance):
                add('outside_core_interval',(index,),f'Строка {top:g}–{base:g} {unit} выходит за установленный интервал {declared_top:g}–{declared_base:g} {unit}.',2)
            if frontier is None:
                frontier=base;owner=index;start_union=top
            elif top>frontier:
                union+=frontier-start_union
                if top-frontier>tolerance:
                    summary['gaps']+=1
                    add('data_gap',(owner,index),f'Между загруженными данными нет описаний {frontier:g}–{top:g} {unit} ({top-frontier:g} {unit}). Возможен перерыв отбора или неполная загрузка.')
                start_union=top;frontier=base;owner=index
            else:
                if top<frontier-tolerance:
                    summary['overlaps']+=1
                    add('data_overlap',(owner,index),f'Строки перекрываются на {top:g}–{min(base,frontier):g} {unit}. Проверьте глубины и возможные повторные снимки.',2)
                if base>frontier:frontier=base;owner=index
        if frontier is not None:
            union+=frontier-start_union
            summary.update(mapped_length=round(union,8),depth_from=ordered[0][0],depth_to=frontier)
            if declared_valid:
                if ordered[0][0]>declared_top+tolerance:
                    summary['gaps']+=1;add('data_gap',(ordered[0][2],),f'В начале установленного интервала нет загруженных описаний {declared_top:g}–{ordered[0][0]:g} {unit}.')
                if frontier<declared_base-tolerance:
                    summary['gaps']+=1;add('data_gap',(owner,),f'В конце установленного интервала нет загруженных описаний {frontier:g}–{declared_base:g} {unit}.')
        for key in totals:
            if key!='wells':totals[key]+=summary[key]
        summaries.append(summary)
    issues.sort(key=lambda row:(-row.severity,row.well_index,row.rows,row.code))
    return ReportAudit(summaries,totals,issues)
