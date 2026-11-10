"""Bounded, deterministic photo preflight and conservative label exclusion.

Warnings are photographic heuristics, not calibrated quality or geological
measurements. Pixels and YOLO contours are never edited or inpainted.
"""
from __future__ import annotations
import copy
import hashlib
import math
from pathlib import Path
import time

SCHEMA = "cv1-photo-quality-v1"
MAX_EDGE = 1400
_cache = {}
MAX_EXCLUSION_STROKES = 128
MAX_EXCLUSION_POINTS = 256


def normalize_exclusion_strokes(strokes):
    result = []
    for stroke in strokes[:MAX_EXCLUSION_STROKES] if isinstance(strokes, list) else []:
        try:
            radius = float(stroke["radius"])
            if not 0 < radius <= .25:
                continue
            points = []
            for x, y in stroke["points"][:MAX_EXCLUSION_POINTS]:
                x, y = float(x), float(y)
                if math.isfinite(x) and math.isfinite(y):
                    points.append([max(0., min(1., x)), max(0., min(1., y))])
            if points:
                result.append({"radius": radius, "points": points})
        except (KeyError, TypeError, ValueError, OverflowError):
            continue
    return result


def manual_exclusion_mask(size, strokes, window=None):
    """Rasterize photo-relative brush strokes onto the photo or a crop."""
    import cv2
    import numpy as np

    width, height = size
    left, top, right, bottom = window or (0, 0, width, height)
    mask = np.zeros((bottom - top, right - left), dtype=np.uint8)
    for stroke in normalize_exclusion_strokes(strokes):
        radius = max(1, round(stroke["radius"] * min(width, height)))
        points = [(round(x * width) - left, round(y * height) - top) for x, y in stroke["points"]]
        cv2.circle(mask, points[0], radius, 1, -1)
        for start, end in zip(points, points[1:]):
            cv2.line(mask, start, end, 1, 2 * radius)
    return mask.astype(bool)


def file_signature(path):
    stat = Path(path).stat()
    return {"bytes":stat.st_size,"mtime_ns":stat.st_mtime_ns}


def _regions_mask(shape, regions):
    import numpy as np
    height,width=shape[:2]; mask=np.zeros((height,width),dtype=bool)
    for box in regions or []:
        try:
            l,t,r,b=(box[k] for k in ('left','top','right','bottom')) if isinstance(box,dict) else box
            l,t,r,b=map(float,(l,t,r,b))
            if not all(math.isfinite(v) for v in (l,t,r,b)):continue
            l,t,r,b=max(0,round(l)),max(0,round(t)),min(width,round(r)),min(height,round(b))
            if r>l and b>t:mask[t:b,l:r]=True
        except (KeyError,TypeError,ValueError):continue
    return mask


def label_boxes(image, regions=None, valid_mask=None):
    """Find only bright compact rectangles containing several dark text marks.

    No OCR/download. Pale bands without letter-like marks stay unmasked. Even
    accepted boxes are called candidates because rock can mimic a label.
    Coordinates are normalised to the supplied, unchanged image.
    """
    import cv2
    import numpy as np
    from PIL import Image
    rgb=np.asarray(image.convert('RGB'));height,width=rgb.shape[:2]
    if min(width,height)<12:return []
    scale=min(1.,MAX_EDGE/max(width,height))
    small=cv2.resize(rgb,(max(1,round(width*scale)),max(1,round(height*scale))),interpolation=cv2.INTER_AREA) if scale<1 else rgb
    sh,sw=small.shape[:2]
    scaled=[]
    for box in regions or []:
        try:
            values=tuple(float(box[k]) for k in ('left','top','right','bottom')) if isinstance(box,dict) else tuple(map(float,box))
            if len(values)==4 and all(math.isfinite(v) for v in values):scaled.append(tuple(v*scale for v in values))
        except (ValueError,TypeError,KeyError):continue
    selection=_regions_mask(small.shape,scaled) if scaled else np.ones((sh,sw),bool)
    if valid_mask is not None and np.asarray(valid_mask).shape==(height,width):
        selection &= cv2.resize(np.asarray(valid_mask,dtype='uint8'),(sw,sh),interpolation=cv2.INTER_NEAREST).astype(bool)
    gray=cv2.cvtColor(small,cv2.COLOR_RGB2GRAY)
    chroma=small.max(axis=2).astype(int)-small.min(axis=2).astype(int)
    bright=((gray>=195)&(chroma<40)&selection).astype('uint8')*255
    closed=cv2.morphologyEx(bright,cv2.MORPH_CLOSE,np.ones((3,3),'uint8'))
    count,components,stats,_=cv2.connectedComponentsWithStats(closed,8)
    result=[]
    for index in range(1,count):
        x,y,w,h,area=map(int,stats[index])
        if w<12 or h<6 or not 1.2<=w/max(h,1)<=7 or area<60 or area>sh*sw*.12:continue
        if area/(w*h)<.75 or x==0 or y==0 or x+w>=sw or y+h>=sh:continue
        patch=gray[y:y+h,x:x+w]
        if (patch>=195).mean()<.55:continue
        dark=(patch<min(145,float(np.percentile(patch,90))-55)).astype('uint8')
        n,_,letters,_=cv2.connectedComponentsWithStats(dark,8)
        marks=[s for s in letters[1:] if 1<=s[4]<=w*h*.16 and s[2]<w*.65 and s[3]<h*.95]
        if len(marks)<3:continue
        pad=max(2,round(h*.5)); l,t,r,b=max(0,x-pad),max(0,y-pad),min(sw,x+w+pad),min(sh,y+h+pad)
        ring=gray[t:b,l:r]
        outer=np.ones(ring.shape,bool);outer[y-t:y-t+h,x-l:x-l+w]=False
        if not outer.any() or float(np.median(patch))-float(np.median(ring[outer]))<18:continue
        if selection[t:b,l:r].mean()<.6:continue
        result.append([l/sw,t/sh,r/sw,b/sh])
        if len(result)>=200:break
    return result


def exclusion_mask(size, boxes):
    import numpy as np
    width,height=size; mask=np.zeros((height,width),bool)
    for l,t,r,b in boxes:
        left,top,right,bottom=max(0,round(l*width)),max(0,round(t*height)),min(width,round(r*width)),min(height,round(b*height))
        mask[top:bottom,left:right]=True
    return mask


def tile_usable_mask(tile_size, window, full_size, boxes):
    """Project label boxes into an existing square tile, without altering pixels."""
    import numpy as np
    width,height=tile_size; left,top,right,bottom=window; fw,fh=full_size
    valid=np.ones((height,width),bool)
    for l,t,r,b in boxes:
        x1,y1,x2,y2=max(0,round(l*fw-left)),max(0,round(t*fh-top)),min(width,round(r*fw-left)),min(height,round(b*fh-top))
        if x2>x1 and y2>y1:valid[y1:y2,x1:x2]=False
    return valid


def audit_image(image, regions=None, *, source_size=None, scope='photo', valid_mask=None):
    import cv2
    import numpy as np
    from PIL import Image
    started=time.perf_counter();image=image.convert('RGB');width,height=image.size
    boxes=label_boxes(image,regions,valid_mask)
    small=image.copy();small.thumbnail((MAX_EDGE,MAX_EDGE))
    rgb=np.asarray(small);sh,sw=rgb.shape[:2]
    scaled=[]
    for box in regions or []:
        try:
            l,t,r,b=map(float,(box[k] for k in ('left','top','right','bottom'))) if isinstance(box,dict) else map(float,box)
            if all(math.isfinite(v) for v in (l,t,r,b)):scaled.append([l*sw/width,t*sh/height,r*sw/width,b*sh/height])
        except (ValueError,TypeError,KeyError):continue
    selected=_regions_mask(rgb.shape,scaled) if scaled else np.ones((sh,sw),bool)
    has_mask=valid_mask is not None and np.asarray(valid_mask).shape==(height,width)
    if has_mask:selected &= cv2.resize(np.asarray(valid_mask,dtype='uint8'),(sw,sh),interpolation=cv2.INTER_NEAREST).astype(bool)
    if not scaled:
        # A white report page is not a core surface. Whole-white pictures still
        # receive an exposure warning through the fallback below.
        foreground=rgb.min(axis=2)<245
        if foreground.sum()>max(64,sh*sw*.01):selected &= foreground
    mask=exclusion_mask((sw,sh),boxes)
    selected &= ~mask
    selected=cv2.erode(selected.astype('uint8'),np.ones((3,3),'uint8')).astype(bool)
    gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
    issues=[];metrics={};source_size=tuple(source_size or (width,height))
    def issue(code,message,severity=1):issues.append({'code':code,'message':message,'severity':severity})
    if selected.sum()<64:
        issue('insufficient_pixels','Слишком мало пригодных пикселей породы для проверки фото.',2)
    else:
        values=gray[selected]
        over=float((values>=250).mean());under=float((values<=8).mean())
        contrast=float(np.percentile(values,95)-np.percentile(values,5))
        lap=cv2.Laplacian(gray,cv2.CV_32F,ksize=3)
        lap_score=float(lap[selected].var())
        metrics={'overexposed_fraction':over,'dark_fraction':under,'contrast_p95_p05':contrast,
                 'detail_score':lap_score,'usable_pixels':int(selected.sum()),'label_excluded_fraction':float(mask.mean())}
        if over>.12:issue('bright_areas','Много предельно светлых участков: проверьте пересвет и блики; светлая порода тоже может давать такой сигнал.',2)
        if under>.18:issue('dark_areas','Много предельно тёмных участков: детали могут теряться в тенях.',2)
        if lap_score<20:issue('few_details','Мало мелких деталей: возможны размытие или однородная порода. Проверьте резкость на увеличении.')
        if contrast<18:issue('low_contrast','Низкий контраст ограничивает оценку текстуры; это также бывает свойством однородной породы.')
    if min(source_size)<160:
        issue('small_image','Небольшое изображение: мелкие признаки могут быть неразличимы.')
    if scaled:
        widths=[min(r-l,b-t)*source_size[0]/sw for l,t,r,b in scaled]
        if widths and min(widths)<64:issue('small_core_width','Наиболее узкий фрагмент керна содержит мало пикселей поперёк. Для мелких признаков нужен более крупный снимок.')
    return {'schema':SCHEMA,'scope':scope,'width':width,'height':height,'source_size':list(source_size),
            'region_count':len(scaled),
            'geometry_mask_applied':bool(has_mask),
            'sample_sha256':hashlib.sha256(rgb.tobytes()).hexdigest(),'metrics':metrics,
            'label_boxes':boxes,'label_method':'bright_rectangles_with_text_marks_heuristic',
            'issues':issues,'heuristic':True,'seconds':round(time.perf_counter()-started,4)}


def audit_file(path, columns=None, display_size=None, manual_exclusions=None):
    from PIL import Image,ImageOps
    path=Path(path)
    try:
        signature=file_signature(path)
        key=(str(path.resolve()),tuple(signature.values()),repr(columns),repr(display_size),repr(manual_exclusions),SCHEMA)
        if key in _cache:return copy.deepcopy(_cache[key])
        with Image.open(path) as source:
            source_size=source.size
            if source.width*source.height>100_000_000:raise ValueError('Фото больше 100 Мп; разделите его на отдельные снимки ящиков.')
            if source.getexif().get(274) in (5,6,7,8):source_size=source_size[::-1]
            source.draft('RGB',(MAX_EDGE,MAX_EDGE))
            image=ImageOps.exif_transpose(source).convert('RGB');image.thumbnail((MAX_EDGE,MAX_EDGE))
        regions=[]
        if columns and display_size and min(display_size)>0:
            for box in columns:
                regions.append([float(box[k])*image.width/display_size[0] if k in ('left','right') else float(box[k])*image.height/display_size[1]
                                for k in ('left','top','right','bottom')])
        valid = ~manual_exclusion_mask(image.size, manual_exclusions) if manual_exclusions else None
        result=audit_image(image,regions,source_size=source_size,valid_mask=valid)
        result['file_signature']=signature
        if file_signature(path)!=signature:raise ValueError('Файл изменился во время проверки; повторите проверку фото.')
        if len(_cache)>128:_cache.clear()
        _cache[key]=copy.deepcopy(result)
        return result
    except (OSError,ValueError,TypeError,KeyError,Image.DecompressionBombError) as exc:
        return {'schema':SCHEMA,'scope':'photo','issues':[{'code':'photo_unreadable','severity':3,'message':str(exc)}],
                'label_boxes':[],'metrics':{},'heuristic':True}
