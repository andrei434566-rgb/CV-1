"""Explicit research-only ROI classification. Never replaces expert attributes."""
from __future__ import annotations
import hashlib, json
from pathlib import Path

CLASSES=('cross_bedding','low_angle','massive','parallel_lamination','mud_drape')
PREPROCESSING='whole_roi_rgb_letterbox_224_gray127_imagenet_v1'

def read_card(folder):
    folder=Path(folder)
    card=json.loads((folder/'config.json').read_text(encoding='utf8'))
    if (card.get('schema')!='cv1-bedding-cnn-v1' or card.get('generative') is not False
        or tuple(card.get('native_classes',[]))!=CLASSES
        or card.get('architecture') not in ('efficientnet_b0','convnext_tiny')
        or card.get('preprocessing')!=PREPROCESSING or card.get('input_size')!=224
        or card.get('deployment')!='research_only' or len(card.get('names_ru',[]))!=len(CLASSES)):
        raise ValueError('Несовместимая модель структур керна')
    return card

def prepare_image(image):
    from PIL import Image,ImageOps
    from torchvision import transforms
    from app.infrastructure.vision_models import MEAN,STD
    if min(image.size)<8:raise ValueError('Фрагмент слишком мал для классификации структуры')
    rgb=ImageOps.pad(image.convert('RGB'),(224,224),method=Image.Resampling.BILINEAR,color=(127,127,127))
    return transforms.Compose([transforms.ToTensor(),transforms.Normalize(MEAN,STD)])(rgb)

class BeddingEngine:
    def __init__(self,folder,device='cpu'):
        import torch
        from app.infrastructure.vision_models import create_network
        self.card=read_card(folder);path=Path(folder)/'model.pt'
        if hashlib.sha256(path.read_bytes()).hexdigest()!=self.card['model_sha256']:
            raise ValueError('Контрольная сумма модели слоистости не совпала')
        torch.set_num_threads(min(4,__import__('os').cpu_count() or 1))
        self.model=create_network(self.card['architecture'],classes=len(CLASSES))
        self.model.load_state_dict(torch.load(path,map_location='cpu',weights_only=True))
        self.device=device;self.model=self.model.to(device).eval()

    def classify_roi(self,image):
        import torch
        with torch.inference_mode():p=self.model(prepare_image(image).unsqueeze(0).to(self.device)).float().softmax(1)[0].cpu()
        if not torch.isfinite(p).all():raise ValueError('Некорректные оценки классификатора структур')
        variants=sorted([{'class':k,'name':n,'score':float(v)} for k,n,v in zip(CLASSES,self.card['names_ru'],p)],key=lambda r:-r['score'])
        return {'engine':'bedding_cnn','generative':False,'review_required':True,'research_only':True,
                'model_sha256':self.card['model_sha256'],'variants':variants,'attribute_suggestions':{},
                'warning':'Оценки среди пяти исходных классов, не вероятность геологического диагноза. Автозаполнение выключено; 16 типов ещё не обучены.'}
