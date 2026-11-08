"""Calibrated 2-D apparent grain intercepts; not a sieve or bulk D50 estimate.

Classification is confined to the user's four Wentworth sand fractions.
No colour/contrast statistic or zoom level becomes a grain measurement.
"""
from __future__ import annotations
import math
from statistics import median

SCHEMA='cv1-grain-measurement-v1'
SAND_BANDS=((.0625,.125,'Тонкозернистый'),(.125,.25,'Мелкозернистый'),
            (.25,.5,'Среднезернистый'),(.5,1.,'Крупнозернистый'))
SOURCE='https://publications.iodp.org/proceedings/331/102/102_t1.htm'
MIN_REFERENCE_PX=20.
MIN_GRAIN_PX=8.
MIN_GRAINS=5
PIXEL_ERROR=2.  # Conservative endpoint allowance, not a statistical confidence interval.


def classify_sand_mm(value):
    if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value<=0:
        raise ValueError('Размер должен быть положительным числом в миллиметрах.')
    return next((name for lower,upper,name in SAND_BANDS if lower<=value<upper),'')


def line_length(line,size):
    if (not isinstance(line,(list,tuple)) or len(line)!=4
        or any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) for v in line)):
        raise ValueError('Нужны две точки измерения на исходном изображении.')
    if (not isinstance(size,(list,tuple)) or len(size)!=2
        or any(isinstance(v,bool) or not isinstance(v,(int,float)) or v<=0 or not math.isfinite(v) for v in size)):
        raise ValueError('Неизвестен размер исходного изображения.')
    x1,y1,x2,y2=line;w,h=size
    if not(0<=x1<=w and 0<=x2<=w and 0<=y1<=h and 0<=y2<=h):
        raise ValueError('Точки вышли за границы исходного фрагмента.')
    length=math.hypot(x2-x1,y2-y1)
    if length<=0:raise ValueError('Совпадающие точки не задают размер.')
    return length


def assess_measurements(reference_line,reference_mm,grain_lines,image_size,*,
                        same_plane=False,grains_visible=False,sand_context=False,image_sha256=''):
    """Return measurements and an explicit refusal when physical evidence is absent."""
    if isinstance(reference_mm,bool) or not isinstance(reference_mm,(int,float)) or not math.isfinite(reference_mm) or reference_mm<=0:
        raise ValueError('Укажите известную длину масштаба в миллиметрах.')
    ref=line_length(reference_line,image_size)
    if ref<MIN_REFERENCE_PX:raise ValueError('Отрезок масштаба слишком короткий: нужно минимум 20 исходных пикселей.')
    if not isinstance(grain_lines,(list,tuple)) or len(grain_lines)>500:
        raise ValueError('Допустимо не более 500 измерений зёрен.')
    mm_per_px=reference_mm/ref;rows=[];resolved=[];seen=set();duplicates=0
    for line in grain_lines:
        length=line_length(line,image_size)
        signature=tuple(sorted([(round(line[0],4),round(line[1],4)),(round(line[2],4),round(line[3],4))]))
        duplicate=signature in seen;duplicates+=int(duplicate);seen.add(signature)
        low=max(0.,length-PIXEL_ERROR)*reference_mm/(ref+PIXEL_ERROR)
        high=(length+PIXEL_ERROR)*reference_mm/(ref-PIXEL_ERROR)
        row={'line':list(line),'pixels':length,'apparent_mm':length*mm_per_px,
             'lower_mm':low,'upper_mm':high,'resolved':length>=MIN_GRAIN_PX and not duplicate,'duplicate':duplicate}
        rows.append(row)
        if row['resolved']:resolved.append(row)
    result={'schema':SCHEMA,'scale':'Wentworth sand fractions; lower inclusive, upper exclusive',
            'source':SOURCE,'image_size':list(image_size),'image_sha256':image_sha256,
            'reference_line':list(reference_line),'reference_mm':reference_mm,'mm_per_pixel':mm_per_px,
            'same_plane_confirmed':same_plane is True,'grains_visible_confirmed':grains_visible is True,
            'sand_context_confirmed':sand_context is True,'grain_lines':rows,'usable_grains':len(resolved),
            'duplicates_detected':duplicates,
            'minimum_resolved_grain_mm':MIN_GRAIN_PX*mm_per_px,
            'label':'','assumed':True,'review_required':True,'status':'incomplete',
            'method':'Median of human-selected apparent 2D grain intercepts, not laboratory D50',
            'limitation':'Оценка видимых двумерных сечений. Не заменяет ситовой анализ, шлиф или объёмное распределение размеров.'}
    if same_plane is not True:
        result['reason']='Подтвердите, что масштаб и поверхность породы находятся в одной плоскости без заметного перспективного искажения.'
    elif grains_visible is not True:
        result['reason']='Подтвердите, что видны границы отдельных зёрен; слойки, трещины, поры и цемент зёрнами не считаются.'
    elif sand_context is not True:
        result['reason']='Четыре диапазона относятся к песчаным обломкам. Для других пород автоматически не применяются.'
    elif duplicates:
        result['reason']='Повторяется один и тот же отрезок. Уберите дубликаты: повторный замер не является новым зерном.'
    elif len(resolved)<MIN_GRAINS:
        result['reason']=f'Нужно минимум {MIN_GRAINS} видимых зёрен длиной от {MIN_GRAIN_PX:g} исходных пикселей; сейчас {len(resolved)}. Желательно измерить 20–30 зёрен.'
    elif len(resolved)!=len(rows):
        result['reason']='Часть выбранных зёрен слишком мала в исходных пикселях. Их исключение исказит зернистость в сторону крупных размеров; нужен снимок с более высоким разрешением.'
    else:
        center=median(r['apparent_mm'] for r in resolved)
        lower=median(r['lower_mm'] for r in resolved);upper=median(r['upper_mm'] for r in resolved)
        result.update(median_apparent_mm=center,lower_mm=lower,upper_mm=upper)
        value=classify_sand_mm(center)
        if not value or lower<.0625 or upper>=1.:
            result.update(status='outside_or_boundary',reason='Размер или его погрешность выходит за четыре выбранных диапазона песчаной фракции. Уточните измерения.')
        elif classify_sand_mm(lower)!=value or classify_sand_mm(upper)!=value:
            result.update(status='boundary',reason='Погрешность пересекает границу зернистости. Автоматическое значение не назначено.')
        elif sum(classify_sand_mm(r['apparent_mm'])==value for r in resolved)/len(resolved)<.7:
            result.update(status='mixed',reason='Измеренные зёрна неоднородны: менее 70% попадает в диапазон медианы. Один тип зернистости автоматически не назначен.')
        else:
            result.update(label=value,status='measured',reason='Предположительная зернистость по медиане видимых размеров. Проверьте представительность выбранных зёрен.')
    return result
