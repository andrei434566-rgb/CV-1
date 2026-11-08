"""Photographic colour families and actual RGB samples within a core mask.

Not a calibrated rock-colour measurement or lithology classifier. Callers use
original photograph pixels, never overlays or a rendered geological column.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

SCHEMA = 'cv1-core-color-v2'


@dataclass(frozen=True)
class CoreColorHint:
    label: str
    supporting_fraction: float
    usable_pixels: int
    limitation: str = "Цвет по фото ориентировочный: зависит от освещения, баланса белого и состояния керна; чёрно-белое фото не передаёт истинный цвет."
    rgb: tuple[int, int, int] = (128, 128, 128)
    components: tuple[dict, ...] = ()
    mixed: bool = False

    def to_dict(self):
        return {'schema':SCHEMA,'label':self.label,'supporting_fraction':self.supporting_fraction,
                'usable_pixels':self.usable_pixels,'rgb':list(self.rgb),'hex':rgb_hex(self.rgb),
                'components':list(self.components),'mixed':self.mixed,'approximate':True,
                'method':'masked_cielab_robust_pixels','limitation':self.limitation}


def rgb_hex(rgb):
    return '#' + ''.join(f'{int(channel):02x}' for channel in rgb)


def _neutral_name(lightness):
    return ('чёрный' if lightness < 18 else 'тёмно-серый' if lightness < 42 else
            'серый' if lightness < 68 else 'светло-серый' if lightness < 90 else 'белый')


def estimate_core_color(image: Image.Image, mask: Image.Image | np.ndarray | None = None, *, exclude_boxes=()) -> CoreColorHint | None:
    """Source-pixel colour families; fraction is pixel coverage, not correctness.

    Low-chroma warm greys stay grey. Separate colours stay separate rather than
    being averaged into a tone absent from the rock. No white-balance guessing.
    Masks/label rectangles must be in the same coordinate system as the image.
    """
    if not isinstance(image,Image.Image) or min(image.size)<12 or image.width*image.height>100_000_000:return None
    import cv2
    original_size=image.size
    sample=image.copy();sample.thumbnail((768,2048),Image.Resampling.BOX)
    width,height=sample.size
    if min(width,height)<4:return None
    selected=np.ones((height,width),dtype=bool)
    if 'A' in sample.getbands():selected &= np.asarray(sample.getchannel('A'))>0
    if mask is not None:
        try:
            source=np.asarray(mask)
            if source.shape!=(original_size[1],original_size[0]):return None
            valid=np.isfinite(source)&(source>0)
            selected &= np.asarray(Image.fromarray(valid.astype('uint8')*255).resize(sample.size,Image.Resampling.NEAREST))>0
        except (ValueError,TypeError):return None
    yy,xx=np.where(selected)
    if len(xx)<64:return None
    l,r,t,b=int(xx.min()),int(xx.max())+1,int(yy.min()),int(yy.max())+1
    dx,dy=max(1,round((r-l)*.12)),max(1,round((b-t)*.03))
    selected[:,:l+dx]=False;selected[:,r-dx:]=False
    selected[:t+dy,:]=False;selected[b-dy:,:]=False
    for box in exclude_boxes or ():
        try:
            l,t,r,b=map(float,box)
            if not all(np.isfinite(v) for v in (l,t,r,b)):continue
            l,t,r,b=max(0,round(l*width)),max(0,round(t*height)),min(width,round(r*width)),min(height,round(b*height))
            if r>l and b>t:selected[t:b,l:r]=False
        except (ValueError,TypeError):continue
    count_before=int(selected.sum())
    if count_before<64:return None
    rgb=np.asarray(sample.convert('RGB'))
    usable=selected&(rgb.max(axis=2)<249)&(rgb.max(axis=2)>8)
    count=int(usable.sum())
    if count<64 or count/count_before<.30:return None
    lab=cv2.cvtColor(rgb.astype('float32')/255.,cv2.COLOR_RGB2LAB)[usable]
    pixels=rgb[usable];light,a,b=lab.T;chroma=np.hypot(a,b)
    hue=np.degrees(np.arctan2(b,a))%360;neutral=chroma<=14
    names=('серо-зелёный','зелёный','бежевый','коричневый','желтовато-коричневый',
           'желтоватый','оливково-коричневый','розоватый','красно-коричневый',
           'красноватый','серо-голубой','голубой','фиолетово-серый','фиолетовый')
    codes=np.full(count,-1,dtype='int8')
    red=~neutral&((hue<40)|(hue>=350))
    codes[red]=9;codes[red&(light<55)]=8;codes[red&(light>=65)&(chroma<40)]=7
    warm=~neutral&(hue>=40)&(hue<80)
    codes[warm]=3;codes[warm&(light>=65)]=4;codes[warm&(light>=65)&(chroma<40)]=2
    yellow=~neutral&(hue>=80)&(hue<110)
    codes[yellow]=5;codes[yellow&(light<48)]=6;codes[yellow&(light>=65)&(chroma<35)]=2
    green=~neutral&(hue>=110)&(hue<175)
    codes[green]=1;codes[green&(chroma<25)]=0
    blue=~neutral&(hue>=175)&(hue<270)
    codes[blue]=11;codes[blue&(chroma<25)]=10
    purple=~neutral&(hue>=270)&(hue<350)
    codes[purple]=13;codes[purple&(chroma<25)]=12
    components=[]
    def component(name,indices):
        representative=tuple(int(v) for v in np.median(pixels[indices],axis=0))
        components.append({'label':name,'fraction':float(indices.sum()/count),
                           'rgb':list(representative),'hex':rgb_hex(representative)})
    if neutral.any():
        tones=np.digitize(light,[18,42,68,90])
        tone_counts=np.bincount(tones[neutral],minlength=5)
        substantial=np.flatnonzero(tone_counts>=max(1,int(neutral.sum()*.25)))
        # Two clearly separated neutral tones must not turn into an invented
        # mid-grey (for example dark and pale alternating bands).
        if len(substantial)>=2 and substantial[-1]-substantial[0]>=2:
            for tone in range(5):
                indices=neutral&(tones==tone)
                if indices.any():component(_neutral_name(float(np.median(light[indices]))),indices)
        else:
            component(_neutral_name(float(np.median(light[neutral]))),neutral)
    for code,name in enumerate(names):
        indices=codes==code
        if indices.any():component(name,indices)
    components.sort(key=lambda row:(-row['fraction'],row['label']))
    major=components[0];significant=[row for row in components if row['fraction']>=.20]
    mixed=major['fraction']<.70 and len(significant)>=2
    label=(' и '.join(row['label'] for row in significant[:3])+' (неоднородный)') if mixed else major['label']
    return CoreColorHint(label,major['fraction'],count,rgb=tuple(major['rgb']),components=tuple(components[:5]),mixed=mixed)


def estimate_core_color_file(path: str | Path, mask: Image.Image | np.ndarray | None = None, **kwargs) -> CoreColorHint | None:
    """Read a crop without assuming an ASCII filesystem path."""
    try:
        with Image.open(Path(path)) as image:
            return estimate_core_color(ImageOps.exif_transpose(image), mask, **kwargs)
    except (OSError, ValueError, UnidentifiedImageError):
        return None


def color_is_confirmed(attributes):
    if attributes.get('__color_expert_confirmed')=='1':return True
    try:assumptions=json.loads(attributes.get('__description_assumptions','[]'))
    except (TypeError,ValueError):assumptions=[]
    if not isinstance(assumptions,list):assumptions=[]
    return bool(attributes.get('__description_composer') and attributes.get('Цвет') and
                attributes.get('Цвет')==attributes.get('__vision_color') and 'Цвет' not in assumptions)


def apply_photo_color(attributes,measurement,*,force=False):
    """Update only a blank/previous automatic colour, preserving expert edits."""
    current=attributes.get('Цвет');previous=attributes.get('__vision_color')
    if not force and (color_is_confirmed(attributes) or current and current!=previous):return False
    if not isinstance(measurement,dict) or not isinstance(measurement.get('label'),str) or not measurement['label']:return False
    attributes['Цвет']=measurement['label'][:120];attributes['__vision_color']=attributes['Цвет']
    attributes['__color_measurement']=json.dumps(measurement,ensure_ascii=False,allow_nan=False)
    attributes['__color_expert_confirmed']='0'
    try:assumptions=json.loads(attributes.get('__description_assumptions','[]'))
    except (TypeError,ValueError):assumptions=[]
    assumptions=assumptions if isinstance(assumptions,list) else []
    attributes['__description_assumptions']=json.dumps(sorted({v for v in assumptions if isinstance(v,str)}|{'Цвет'}),ensure_ascii=False)
    return True
