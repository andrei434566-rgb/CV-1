"""RBF ridge head with exact and partial A/B labels, no pseudo-labels."""
import math


def squared_distances(a,b):
    return (a.square().sum(1)[:,None]+b.square().sum(1)[None,:]-2*a@b.T).clamp_min(0)


def training_bandwidth(x):
    import torch
    d=squared_distances(x,x);values=d[torch.triu(torch.ones_like(d,dtype=torch.bool),diagonal=1)]
    values=values[values>1e-12]
    if not len(values):raise ValueError('No distinct training anchors')
    return values.median().clamp_min(1e-6)


def kernel(a,b,bandwidth,factor):
    import torch
    if float(bandwidth)<=0 or factor<=0:raise ValueError('Positive kernel scale required')
    return torch.exp(-factor*squared_distances(a,b)/bandwidth)


def partial_dual(gram,allowed,weights,alpha):
    import torch
    from scripts.partial_facies_ridge import label_system
    if alpha<=0 or gram.shape!=(len(allowed),len(allowed)) or len(weights)!=len(allowed):raise ValueError('Invalid kernel fit')
    if not torch.isfinite(gram).all() or not torch.isfinite(weights).all() or (weights<0).any():raise ValueError('Invalid numerical inputs')
    y,precision,rotation=label_system(allowed);answer=torch.zeros_like(y)
    # Axes with identical precision share the same linear solve.
    groups={}
    for axis in range(6):groups.setdefault(tuple(precision[:,axis].tolist()),[]).append(axis)
    for axes in groups.values():
        w=weights*precision[:,axes[0]];active=w>0;sw=w[active].sqrt()
        system=gram[active][:,active]*sw[:,None]*sw[None,:]+alpha*torch.eye(int(active.sum()),dtype=torch.float64)
        solved=torch.linalg.solve(system,y[active][:,axes]*sw[:,None])*sw[:,None]
        for j,axis in enumerate(axes):answer[active,axis]=solved[:,j]
    return answer@rotation.T


def grouped_weights(rows,mode,external_weight=.5):
    import torch
    from collections import Counter
    exact=[r for r in rows if len(r['allowed_codes'])==1]
    counts=Counter(r['allowed_codes'][0] for r in exact);n=len(exact)
    if set(counts)!=set('ABCDEF') or mode not in ('class','class_group'):raise ValueError('Six classes and valid weighting required')
    pairs=Counter((r['allowed_codes'][0],r['source_group']) for r in exact)
    support={c:len({r['source_group'] for r in exact if r['allowed_codes']==[c]}) for c in 'ABCDEF'}
    values=[]
    for r in rows:
        if len(r['allowed_codes'])==1:
            c=r['allowed_codes'][0]
            w=n/(6*counts[c]) if mode=='class' else n/(6*support[c]*pairs[c,r['source_group']])
        elif r['allowed_codes']==['A','B']:w=.5*(n/(6*counts['A'])+n/(6*counts['B']))
        else:raise ValueError('Unsupported partial target')
        if r['external']:w*=external_weight
        if r.get('domain')=='core_end_face':w*=.25
        values.append(w)
    return torch.tensor(values,dtype=torch.float64)


def all_class_gates(candidate,baseline):
    result={}
    for c in 'ABCDEF':
        result[c+'_recall']=candidate['per_class_recall'][c]+1e-12>=baseline['per_class_recall'][c]
        result[c+'_f1']=candidate['per_class_f1'][c]+1e-12>=baseline['per_class_f1'][c]
    result['macro_f1_improved']=candidate['macro_f1']>baseline['macro_f1']+1e-12
    return result


def selection_key(item):
    m=item['old_oof'];b=item['baseline_old_oof']
    worst_gain=min(m['per_class_recall'][c]-b['per_class_recall'][c] for c in 'ABCDEF')
    return (all(item['all_class_gates'].values()),worst_gain,min(m['per_class_recall'].values()),
        m['macro_f1'],item['mean_group_accuracy'],item['alpha'],-item['factor'],item['weighting']=='class_group')


def validate_state(state,contract):
    import torch
    if contract.get('feature_contract_version')!='qwen3vl_rbf_partial_head_v1':raise ValueError('Unsupported RBF contract')
    d=contract['feature_dimension'];n=contract['anchor_count']
    if state['anchors'].shape!=(n,d) or state['dual'].shape!=(n,6):raise ValueError('Anchor/dual dimensions differ')
    if state['mean'].shape!=(d,) or state['scale'].shape!=(d,) or len(state['columns'])!=d:raise ValueError('Normalizer dimensions differ')
    for name in ('anchors','dual','mean','scale','bandwidth','factor'):
        if not torch.isfinite(state[name]).all():raise ValueError('Nonfinite saved head')
    if (state['scale']<=0).any() or float(state['bandwidth'])<=0 or float(state['factor'])<=0:raise ValueError('Invalid saved head scale')
    if state['columns'].dtype!=torch.int64 or not torch.equal(state['columns'],torch.arange(2048,6144)):raise ValueError('Only original Qwen post profile supported')


def predict_features(raw,state):
    from scripts.facies_head_v2_features import normalize
    x=normalize(raw,state)[:,:-1]
    return kernel(x,state['anchors'],state['bandwidth'],float(state['factor']))@state['dual']
