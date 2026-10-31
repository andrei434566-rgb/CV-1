"""Colour/texture two-view research head. Labels never affect image conversion."""


def grayscale_view(image):
    from PIL import ImageOps
    return ImageOps.grayscale(image.convert('RGB')).convert('RGB')


def fit_gray(raw,rows,alpha):
    from scripts.facies_head_v2_features import prepare_state,normalize
    from scripts.partial_facies_ridge import fit_partial
    from scripts.kernel_facies_head import grouped_weights
    state=prepare_state(raw,'qwen_post',6144,2048,114)
    state['coefficients']=fit_partial(normalize(raw,state),[r['allowed_codes'] for r in rows],grouped_weights(rows,'class_group',.5),alpha)
    return state


def blend_scores(rgb_scores,gray_scores,weight):
    if not 0<weight<=1:raise ValueError('Gray weight must be in (0,1]')
    if rgb_scores.shape!=gray_scores.shape or rgb_scores.ndim!=2 or rgb_scores.shape[1]!=6:raise ValueError('Aligned six-class scores required')
    return (1-weight)*rgb_scores+weight*gray_scores


def candidate_key(item):
    m,b=item['old_oof'],item['baseline_old_oof']
    return (all(item['all_class_gates'].values()),
            min(m['per_class_recall'][c]-b['per_class_recall'][c] for c in 'ABCDEF'),
            min(m['per_class_f1'][c]-b['per_class_f1'][c] for c in 'ABCDEF'),
            item['mean_group_accuracy'],m['macro_f1'],-item['gray_weight'],item['alpha'])


def pack_state(rgb,gray,weight):
    import torch
    out={'gray_weight':torch.tensor(weight,dtype=torch.float64)}
    out.update({'rgb__'+k:v for k,v in rgb.items()})
    out.update({'gray__'+k:v for k,v in gray.items()})
    return out


def unpack_state(state,prefix):
    marker=prefix+'__'
    return {k[len(marker):]:v for k,v in state.items() if k.startswith(marker)}


def predict_features(rgb,gray,state):
    from scripts.facies_head_v2_features import predict
    return blend_scores(predict(rgb,unpack_state(state,'rgb')),predict(gray,unpack_state(state,'gray')),float(state['gray_weight']))
