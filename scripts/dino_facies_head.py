"""Small DINOv2 research features, with explicit aspect-preserving token masks."""
import math


def letterbox(image,side=224):
    from PIL import Image
    image=image.convert('RGB');scale=side/max(image.size)
    image=image.resize((max(1,round(image.width*scale)),max(1,round(image.height*scale))),Image.Resampling.BICUBIC)
    left=(side-image.width)//2;top=(side-image.height)//2
    canvas=Image.new('RGB',(side,side),(124,116,104));canvas.paste(image,(left,top))
    return canvas,(left,top,left+image.width,top+image.height)


def tokens_in_box(box,side=224,patch=14):
    import torch
    if side%patch:raise ValueError('Whole patch grid required')
    centers=(torch.arange(side//patch)+.5)*patch;y,x=torch.meshgrid(centers,centers,indexing='ij')
    left,top,right,bottom=box
    valid=(x>=left)&(x<right)&(y>=top)&(y<bottom)
    if int(valid.sum())<2:raise ValueError('Insufficient core tokens')
    return valid.flatten()


def pool_tokens(hidden,valid=None):
    import torch
    if hidden.ndim!=2 or hidden.shape[0]<3:raise ValueError('One image token matrix required')
    patches=hidden[1:].float()
    if valid is not None:
        if len(valid)!=len(patches):raise ValueError('Token mask shape differs')
        patches=patches[valid.to(patches.device)]
    if len(patches)<2:raise ValueError('Insufficient non-padding tokens')
    return torch.cat((hidden[0].float(),patches.mean(0),patches.std(0,unbiased=False)))


def fit_head(raw,rows,alpha):
    import torch
    from scripts.facies_head_v2_features import normalize
    from scripts.partial_facies_ridge import fit_partial
    from scripts.kernel_facies_head import grouped_weights
    if raw.ndim!=2 or raw.shape[1]!=1152:raise ValueError('Expected small DINO CLS/mean/std1152')
    x=raw.double()
    state=dict(columns=torch.arange(1152),block_sizes=torch.tensor([384,384,384]),block_weights=torch.full((3,),1/math.sqrt(3),dtype=torch.float64),
               mean=x.mean(0),scale=x.std(0,unbiased=False).clamp_min(1e-3))
    state['coefficients']=fit_partial(normalize(raw,state),[r['allowed_codes'] for r in rows],grouped_weights(rows,'class_group',.5),alpha)
    return state


def candidate_key(item):
    m,b=item['old_oof'],item['baseline_old_oof']
    return (all(item['all_class_gates'].values()),min(m['per_class_recall'][c]-b['per_class_recall'][c] for c in 'ABCDEF'),
            min(m['per_class_f1'][c]-b['per_class_f1'][c] for c in 'ABCDEF'),item['mean_group_accuracy'],m['macro_f1'],
            -item['dino_weight'],item['alpha'],item['view']=='letterbox')


def pack_state(rgb,dino,weight):
    import torch
    result={'dino_weight':torch.tensor(weight,dtype=torch.float64)}
    result.update({'rgb__'+k:v for k,v in rgb.items()});result.update({'dino__'+k:v for k,v in dino.items()})
    return result


def predict_features(rgb,dino,state):
    from scripts.facies_head_v2_features import predict
    from scripts.gray_facies_head import unpack_state,blend_scores
    return blend_scores(predict(rgb,unpack_state(state,'rgb')),predict(dino,unpack_state(state,'dino')),float(state['dino_weight']))


def extract_features(records,base,log=None):
    """Two fixed image views; labels are not passed to either encoder or pooling."""
    import torch
    from PIL import Image
    from transformers import AutoImageProcessor,Dinov2Model
    from scripts import train_unibo_visual_facies as source
    processor=AutoImageProcessor.from_pretrained(base,local_files_only=True,use_fast=False)
    model=Dinov2Model.from_pretrained(base,local_files_only=True,attn_implementation='sdpa').cuda().eval();model.requires_grad_(False)
    features={'standard':[],'letterbox':[]}
    for offset in range(0,len(records),8):
        images=[]
        for r in records[offset:offset+8]:
            expected=r.get('image_sha256',r.get('metadata',{}).get('image_sha256'))
            if source.digest(r['image'])!=expected:raise ValueError('Source changed')
            with Image.open(r['image']) as im:images.append(im.convert('RGB'))
        with torch.inference_mode():
            inputs=processor(images=images,return_tensors='pt').to('cuda');hidden=model(**inputs).last_hidden_state
            features['standard'].extend(pool_tokens(x).cpu() for x in hidden)
            boxed=[letterbox(im) for im in images]
            inputs=processor(images=[im for im,box in boxed],do_resize=False,do_center_crop=False,return_tensors='pt').to('cuda')
            if tuple(inputs['pixel_values'].shape[-2:])!=(224,224):raise ValueError('Letterbox preprocessing changed')
            hidden=model(**inputs).last_hidden_state
            features['letterbox'].extend(pool_tokens(x,tokens_in_box(box)).cpu() for x,(im,box) in zip(hidden,boxed))
        if log:log('dino_features',done=min(offset+8,len(records)),total=len(records))
    del model;torch.cuda.empty_cache()
    return {k:torch.stack(v) for k,v in features.items()}
