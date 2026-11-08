"""Offline discriminative lithology inference and source retrieval.

Only allow-listed CNNs are loaded. No decoder, prompt, token generation, server,
Transformers, LLM or VLM is used. Geology absent from the labels stays unknown.
"""
from __future__ import annotations
import hashlib, json, math, re, time
from pathlib import Path

from app.runtime_paths import application_root, bundled_root, user_data_root
from app.infrastructure.vision_models import ARCHITECTURES, LABELS, LABELS_RU, MEAN, STD
from app.domain.rock_types import ROCK_CLASSES, COAL_CLASSES
from app.domain.lithology_attributes import LITHOLOGY_ATTRIBUTE_OPTIONS
from app.infrastructure.description_template import PHOTO_UNRESOLVED, compose_description

def default_model_folder() -> Path:
    for root in dict.fromkeys([user_data_root(),application_root(),bundled_root()]):
        parent=root/'models/vision';pointer=parent/'current.json'
        if pointer.is_file():
            try:
                name=json.loads(pointer.read_text(encoding='utf-8'))['folder']
                path=(parent/name).resolve()
                if path.is_relative_to(parent.resolve()) and (path/'config.json').is_file():return path
            except (OSError,ValueError,KeyError,TypeError):pass
    return user_data_root()/'models/vision/unavailable'

def read_model_card(folder: str | Path) -> dict:
    folder=Path(folder)
    card=json.loads((folder/'config.json').read_text(encoding='utf-8'))
    classes=tuple(card.get('classes',[]))
    valid_vocabulary=(card.get('schema')=='cv1-discriminative-vision-v1' and classes==LABELS) or (
        card.get('schema')=='cv1-discriminative-vision-v2' and classes in (ROCK_CLASSES,COAL_CLASSES)
        and card.get('background_class')=='background')
    if (not valid_vocabulary or card.get('generative') is not False
            or card.get('architecture') not in ARCHITECTURES):
        raise ValueError('Нужна модель распознавания CV-1 Vision. Генеративные модели отключены.')
    if not (folder/'model.pt').is_file():raise FileNotFoundError('Не найдены веса классификатора model.pt.')
    return card

def _sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(2*1024*1024),b''):h.update(block)
    return h.hexdigest()

def _boxes(image,columns):
    """Use supplied physical columns; infer only visual ROIs when absent."""
    import numpy as np
    width,height=image.size;boxes=[]
    for value in columns or []:
        try:
            b=[float(value[k]) for k in ('left','top','right','bottom')] if isinstance(value,dict) else list(map(float,value))
            if len(b)!=4 or not all(math.isfinite(x) for x in b):continue
            l,t,r,bottom=[int(round(x)) for x in b]
            l,t,r,bottom=max(0,l),max(0,t),min(width,r),min(height,bottom)
            if r-l>=12 and bottom-t>=12:boxes.append((l,t,r,bottom))
        except (ValueError,KeyError,TypeError):continue
    if boxes:return boxes[:50]
    if max(width,height)>min(width,height)*2.5:return [(0,0,width,height)]
    # Photos may include large white pages. This detector never creates facies labels.
    from app.infrastructure.ml.rule_based_facies import RuleBasedFaciesDetector
    small=image.copy();small.thumbnail((1600,1600))
    proposals=RuleBasedFaciesDetector._find_core_columns(np.asarray(small)[:,:,::-1].copy())
    sx,sy=width/small.width,height/small.height
    if proposals:return [(round(l*sx),round(t*sy),round(r*sx),round(b*sy)) for l,t,r,b in proposals[:50]]
    return [(0,0,width,height)]

def _tile_data(image,columns=None,max_tiles=40):
    """Square crops along the actual long core axis; never squash a whole column."""
    import numpy as np
    boxes=_boxes(image,columns);tiles=[];windows=[]
    per=max(1,min(8,max_tiles//max(1,len(boxes))))
    for l,t,r,b in boxes:
        dx=max(1,round((r-l)*.08));dy=max(1,round((b-t)*.025))
        roi=image.crop((l+dx,t+dy,max(l+dx+1,r-dx),max(t+dy+1,b-dy)))
        w,h=roi.size;side=min(w,h)
        if side<12:continue
        count=min(per,max(1,math.ceil(max(w,h)/side)))
        for offset in np.linspace(0,max(w,h)-side,count):
            x,y=(0,round(offset)) if h>=w else (round(offset),0)
            tile=roi.crop((x,y,x+side,y+side))
            a=np.asarray(tile);white=(a.min(axis=2)>240).mean()
            if white<.85:
                tiles.append(tile)
                windows.append((l+dx+x,t+dy+y,l+dx+x+side,t+dy+y+side))
    return tiles[:max_tiles],boxes,windows[:max_tiles]


def image_tiles(image,columns=None,max_tiles=40):
    tiles,boxes,_=_tile_data(image,columns,max_tiles)
    return tiles,boxes

def _band_contrast(tiles,masks=None):
    """Contrast seen in horizontal bands of the usable image windows."""
    import numpy as np
    scores=[]
    from PIL import Image
    for index,tile in enumerate(tiles):
        a=np.asarray(tile.resize((128,128)).convert('L'),dtype=float)
        valid=(a>10)&(a<245)
        if masks is not None:
            usable=np.asarray(Image.fromarray(masks[index].astype('uint8')*255).resize((128,128),Image.Resampling.NEAREST))>0
            valid &= usable
        a,valid=a[:,10:-10],valid[:,10:-10]
        rows=valid.sum(axis=1)>=max(12,a.shape[1]*.25)
        if rows.sum()<16:continue
        row=np.ma.median(np.ma.array(a[rows],mask=~valid[rows]),axis=1).filled(0)
        spread=float(np.percentile(row,90)-np.percentile(row,10))
        scores.append(spread)
    if not scores:
        return None
    return sorted(scores, reverse=True)[math.ceil(len(scores)*.4)-1]

def _band_hint(contrast):
    if contrast is None:
        return ""
    if contrast > 25:
        return "На фото заметны контрастные полосы."
    if contrast > 12:
        return "На фото заметна слабоконтрастная полосчатость."
    return "Выраженная полосчатость на фото не видна."

def visual_band_hint(tiles, masks=None):
    return _band_hint(_band_contrast(tiles, masks))


def photo_description_draft(summary, color, contrast, personal_predictions=(), facies_candidate=None):
    """Fill the form with observable values, model candidates or a photographic status."""
    attributes = dict.fromkeys(LITHOLOGY_ATTRIBUTE_OPTIONS, PHOTO_UNRESOLVED)
    assumptions = set()
    lithology = summary['candidate']
    if lithology:
        attributes['Название породы'] = lithology
        assumptions.add('Название породы')
        fraction = {'sandstone': 'Песчаная фракция', 'siltstone': 'Алевритовая фракция',
                    'carbonaceous_siltstone': 'Алевритовая фракция', 'shale': 'Глинистая фракция',
                    'mudrock': 'Тонкодисперсная фракция'}.get(summary['native_class'])
        if fraction:
            attributes['Зернистость'] = fraction
            assumptions.add('Зернистость')
    if color:
        attributes['Цвет'] = color
        assumptions.add('Цвет')
    if contrast is not None:
        attributes['Текстура'] = 'Полосчатая' if contrast > 12 else 'Визуально однородная'
        assumptions.add('Текстура')
        if contrast > 12:
            attributes['Слоистость'] = 'Субгоризонтальная'
            assumptions.add('Слоистость')
    for prediction in personal_predictions:
        field = prediction['field']
        if field in {'Слоистость', 'Биотурбация', 'Текстура', 'Зернистость'}:
            attributes[field] = prediction['label']
            assumptions.add(field)
    reference = facies_candidate or {}
    facies_name = reference.get('facies_name')
    if not facies_name:
        family = {'sandstone': 'Песчаная', 'siltstone': 'Алевритовая', 'shale': 'Глинистая',
                  'mudrock': 'Глинистая', 'limestone': 'Карбонатная', 'carbonate_other': 'Карбонатная',
                  'coal': 'Угольная'}.get(summary['native_class'])
        facies_name = (
            (family + ' литофация' if family else 'Литофация «' + lithology.lower() + '»')
            if lithology else PHOTO_UNRESOLVED)
    for field, key in (('Фациальная ассоциация', 'facies_association'),
                       ('Обстановка седиментации', 'depositional_environment'),
                       ('Гидродинамический режим', 'hydrodynamic_regime')):
        attributes[field] = reference.get(key) or PHOTO_UNRESOLVED
        if attributes[field] != PHOTO_UNRESOLVED:
            assumptions.add(field)
    if facies_name != PHOTO_UNRESOLVED:
        assumptions.add('Название фации')
    return attributes, assumptions, facies_name

class VisionEngine:
    def __init__(self,device=None):
        self.model=None;self.folder=None;self.card=None;self.device=device
    def _load(self,folder):
        import torch
        from app.infrastructure.vision_models import create_network
        folder=Path(folder).resolve();card=read_model_card(folder)
        signature=(str(folder),card['model_sha256'])
        if self.model is not None and self.folder==signature:return 0.
        start=time.perf_counter()
        if _sha(folder/'model.pt')!=card['model_sha256']:raise ValueError('Контрольная сумма весов не совпадает с карточкой модели.')
        torch.set_num_threads(min(4,__import__('os').cpu_count() or 1))
        self.device=self.device or ('cuda' if torch.cuda.is_available() else 'cpu')
        model=create_network(card['architecture'],classes=len(card['classes']))
        model.load_state_dict(torch.load(folder/'model.pt',map_location='cpu',weights_only=True))
        self.model=model.to(self.device).eval();self.folder=signature;self.card=card
        return time.perf_counter()-start
    def describe(self,request):
        import torch
        import numpy as np
        from PIL import Image,ImageOps
        from torchvision import transforms
        from app.infrastructure.core_color_hint import estimate_core_color
        start=time.perf_counter();folder=request.get('base') or str(default_model_folder())
        loaded=self._load(folder)
        with Image.open(request['image']) as im:
            if im.width*im.height>100_000_000:raise ValueError('Изображение больше 100 Мп. Разделите его на фотографии ящиков.')
            geometry_mask=np.asarray(ImageOps.exif_transpose(im).getchannel('A'))>0 if 'A' in im.getbands() else None
        from app.infrastructure.ml.core_column_service import read_bgr_image
        pixels=read_bgr_image(Path(request['image']))
        image=Image.fromarray(pixels[:,:,::-1])
        if geometry_mask is not None and geometry_mask.shape!=(image.height,image.width):geometry_mask=None
        del pixels
        columns=request.get('columns')
        if request.get('operation') in {'describe','describe_batch','describe_photo'} and not columns:
            # These inputs are already masked interval/column crops from YOLO.
            # Searching for new columns inside a rock texture would change the ROI.
            columns=[[0,0,image.width,image.height]]
        tiles,boxes,windows=_tile_data(image,columns)
        from app.infrastructure.photo_quality import audit_image,tile_usable_mask
        scope='interval' if request.get('operation') in {'describe','describe_batch','describe_photo'} else 'photo'
        quality=audit_image(image,boxes,scope=scope,valid_mask=geometry_mask)
        usable_masks=[tile_usable_mask(tile.size,window,image.size,quality['label_boxes']) for tile,window in zip(tiles,windows)]
        if geometry_mask is not None:
            usable_masks=[mask & geometry_mask[top:bottom,left:right] for mask,(left,top,right,bottom) in zip(usable_masks,windows)]
        from app.infrastructure.photo_quality import _regions_mask
        colour_started=time.perf_counter()
        colour_mask=_regions_mask((image.height,image.width),boxes)
        if geometry_mask is not None:colour_mask &= geometry_mask
        color_hint=estimate_core_color(image,colour_mask,exclude_boxes=quality['label_boxes'])
        color_measurement=color_hint.to_dict() if color_hint else None
        color=color_hint.label if color_hint else None
        colour_seconds=time.perf_counter()-colour_started
        transform=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),transforms.Normalize(MEAN,STD)])
        probabilities=[]
        from app.infrastructure.supervised_learning import classifier_input_layer, head_suggestions
        captured=[]
        # Reuse the exact same forward pass, preserving base-model predictions.
        feature_hook=classifier_input_layer(self.model,self.card['architecture']).register_forward_pre_hook(
            lambda _module, inputs: captured.append(inputs[0].detach().float().cpu()))
        try:
            with torch.inference_mode():
                for i in range(0,len(tiles),16):
                    batch=torch.stack([transform(im) for im in tiles[i:i+16]]).to(self.device)
                    if self.device=='cuda':
                        with torch.autocast('cuda',dtype=torch.float16):logits=self.model(batch)
                    else:logits=self.model(batch)
                    if not torch.isfinite(logits).all():raise ValueError('Классификатор вернул некорректные оценки. Требуется проверка модели.')
                    probabilities.extend(torch.softmax(logits.float()/self.card.get('temperature',1.),dim=1).cpu().tolist())
        finally:
            feature_hook.remove()
        suggestions={};personal_predictions=[]
        if captured:
            embedding=torch.nn.functional.normalize(torch.cat(captured).mean(0),dim=0)
            suggestions,personal_predictions=head_suggestions(embedding,self.card['model_sha256'],user_data_root()/'models/features')
        from app.infrastructure.rock_predictions import summarize_predictions
        summary=summarize_predictions(probabilities,self.card)
        confidence=summary['confidence'];low=summary['low']
        contrast = _band_contrast(tiles, usable_masks)
        facies_candidate = request.get('facies_candidate') or {}
        attributes, assumptions, facies_name = photo_description_draft(
            summary, color, contrast, personal_predictions, facies_candidate)
        draft = compose_description(attributes, facies_name=facies_name, assumptions=assumptions)
        fields={'facies_name':facies_name,'lithology':attributes['Название породы'],'texture':attributes['Текстура'],
                'facies_association':attributes['Фациальная ассоциация'],
                'depositional_environment':attributes['Обстановка седиментации'],
                'description':draft.text,'hydrodynamic_regime':attributes['Гидродинамический режим']}
        answer = draft.text
        if self.card.get('noncommercial_only'):answer+=' Модель для некоммерческих исследований.'
        return {'ok':True,'answer':answer,'fields':fields,'hits':[],'incomplete':False,'engine':'vision',
                'review_required':True,'lithology_review_required':low,'field_confidence':{'lithology':confidence,'facies_name':None},
                'draft_observations':attributes,'draft_assumptions':sorted(assumptions),
                'facies_label':facies_candidate.get('facies_label'), 'facies_index':facies_candidate.get('facies_index'),
                'observations':{'color':color,'color_measurement':color_measurement,
                                'color_is_approximate':True,'visual_hint':_band_hint(contrast)},
                'predictions':summary['predictions'],'column_count':len(boxes),'tiles':len(tiles),
                'window_predictions':[{'box':list(window),'predictions':dict(zip(self.card['classes'],row))}
                                      for window,row in zip(windows,probabilities)],
                'image_quality':quality,
                'attribute_suggestions':suggestions,'personal_predictions':personal_predictions,
                'model_info':{'engine':'vision','generative':False,'architecture':self.card['architecture'],
                              'model_sha256':self.card['model_sha256'],'facies_trained':False,
                              'classes':self.card['classes'],'noncommercial_only':bool(self.card.get('noncommercial_only')),
                              'interbedding_trained':False},
                'timing':{'load_seconds':round(loaded,3),'color_seconds':round(colour_seconds,4),'total_seconds':round(time.perf_counter()-start,3)}}
    def answer(self,request):
        operation=request.get('operation','chat')
        if operation=='prepare_runtime':
            if request.get('base'):read_model_card(request['base'])
            return {'ok':True,'prepared':True,'answer':'','engine':'vision'}
        if operation=='clear_description_cache':return {'ok':True,'cache_cleared':True,'answer':''}
        if operation in {'describe_batch','describe_photo'}:
            items=request.get('items')
            if not isinstance(items,list) or not 1<=len(items)<=64:raise ValueError('Пакет должен содержать 1–64 интервала.')
            ids=[str(r.get('id','')) for r in items]
            if any(not s for s in ids) or len(set(ids))!=len(ids):raise ValueError('Требуются уникальные id интервалов.')
            replies={}
            for key,item in zip(ids,items):
                try:replies[key]=self.describe(dict(request,**{k:v for k,v in item.items() if k!='id'}))
                except Exception as exc:replies[key]={'ok':False,'error':f'{type(exc).__name__}: {exc}'}
            return {'ok':True,'items':replies,'answer':'','engine':'vision','incomplete':False}
        if operation in {'describe','photo_chat'} or (operation=='chat' and request.get('image')):
            return self.describe(request)
        if operation in {'chat','search'}:return reference_answer(request)
        raise ValueError('Неизвестная операция локального распознавания: '+str(operation))

def reference_answer(request):
    from app.infrastructure.local_chat_intents import quick_reply
    from app.infrastructure.local_theory_lookup import lookup_theory_answer
    from app.infrastructure.geology_reference import retrieve_notes
    from app.infrastructure.local_knowledge_search import search_folders
    from app.infrastructure.primary_description_source import source_index,is_reference_question
    question=str(request.get('question') or '')[:5000];show=bool(request.get('show_sources'));hits=[]
    if request.get('primary_reference') and is_reference_question(question):
        index=source_index(request['primary_reference']);terms=set(re.findall(r'[а-яёa-z]{4,}',question.casefold()))
        ranked=sorted(((sum(t[:5] in c['text'].casefold() for t in terms),c) for c in index['chunks']),key=lambda r:-r[0])
        found=[c for score,c in ranked[:2] if score>0]
        if found:
            answer='Из основного справочника специалиста:\n\n'+'\n\n'.join(c['text'][:1800] for c in found)
            if show:answer+='\n\nМатериалы: '+', '.join(c['document'] for c in found)
            return {'ok':True,'answer':answer,'hits':[],'engine':'reference','incomplete':False}
    from app.domain.rock_types import rock_reference_answer
    from app.domain.bedding_types import reference_answer as bedding_reference_answer
    bedding_answer=bedding_reference_answer(question,show_sources=show)
    if bedding_answer:
        return {'ok':True,'answer':bedding_answer,'hits':[],'engine':'reference','incomplete':False}
    diagnostic=any(word in question.casefold() for word in ('отлич','признак','сравни','характерист','углист','битуминозн','окремнен','окремнён'))
    answer=(rock_reference_answer(question,show_sources=show) if diagnostic else '') or quick_reply(question)
    if not answer:
        answer=lookup_theory_answer(question,bundled_root()/'app/resources/theory/theory_cards_v5.json',include_sources=show)
    if not answer:answer=rock_reference_answer(question,show_sources=show)
    if not answer:
        notes=retrieve_notes(question,max_notes=2)
        if notes:
            answer='Из геологического справочника:\n\n'+'\n\n'.join(n['text'] for n in notes)
            if show:answer+='\n\nИсточники: '+', '.join(str(s.get('url') or s.get('title') or '') for n in notes for s in n['source_records'])
    if not answer or request.get('operation')=='search':
        hits=search_folders(question,[str(p) for p in request.get('folders',[])],limit=4)
        if hits:answer='Найдено в локальных материалах:\n\n'+'\n\n'.join(h.get('excerpt','') for h in hits)
    if not answer:answer=('В проверенном справочнике подходящего ответа не найдено. Уточните название породы или признак, '
                          'либо добавьте документ в папки знаний. Помощник работает поиском и готовыми справочными ответами.')
    return {'ok':True,'answer':answer,'hits':hits,'engine':'reference','incomplete':False}
