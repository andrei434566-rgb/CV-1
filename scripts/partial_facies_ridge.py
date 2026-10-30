"""Ridge with exact facies and generic floodplain supervision.

For a generic floodplain target: (s_A+s_B-1)^2 + sum(s_C..s_F^2).
The A-B direction is not supervised by those rows. No drainage is imputed.
"""
import math


def label_system(allowed):
    import torch
    rotation=torch.eye(6,dtype=torch.float64)
    rotation[:2,:2]=torch.tensor([[1,1],[1,-1]],dtype=torch.float64)/math.sqrt(2)
    y=torch.zeros((len(allowed),6),dtype=torch.float64)
    precision=torch.ones_like(y)
    for i,codes in enumerate(allowed):
        if len(codes)==1 and codes[0] in 'ABCDEF': y[i,'ABCDEF'.index(codes[0])]=1
        elif list(codes)==['A','B']:
            y[i,:2]=.5; precision[i,0]=2.; precision[i,1]=0.
        else: raise ValueError('Only exact six-class or generic A/B floodplain labels supported')
    return y@rotation,precision,rotation


def row_weights(allowed,is_external,external_weight):
    import torch
    from collections import Counter
    counts=Counter(x[0] for x in allowed if len(x)==1)
    if set(counts)!=set('ABCDEF'): raise ValueError('All exact classes required')
    n=sum(counts.values()); per_class={c:n/(6*counts[c]) for c in 'ABCDEF'}
    weights=[(per_class[x[0]] if len(x)==1 else (per_class['A']+per_class['B'])/2)
             *(external_weight if ext else 1.) for x,ext in zip(allowed,is_external)]
    if len(weights)!=len(allowed): raise ValueError('Row weights not aligned')
    return torch.tensor(weights,dtype=torch.float64)


def fit_partial(x,allowed,weights,alpha):
    import torch
    if alpha<=0 or len(x)!=len(allowed) or len(weights)!=len(x): raise ValueError('Invalid ridge inputs')
    targets,precision,rotation=label_system(allowed)
    outputs=[]
    for axis in range(6):
        w=weights*precision[:,axis]; active=w>0; sw=w[active].sqrt()
        weighted=x[active]*sw[:,None]
        rhs=targets[active,axis]*sw
        dual=torch.linalg.solve(weighted@weighted.T+alpha*torch.eye(len(weighted),dtype=torch.float64),rhs)
        outputs.append(weighted.T@dual)
    return torch.stack(outputs,1)@rotation.T


def mixed_metrics(rows,predictions,source):
    exact=[(r['allowed_codes'][0],p) for r,p in zip(rows,predictions) if len(r['allowed_codes'])==1]
    metrics=source.classification_metrics([a for a,p in exact],[p for a,p in exact])
    partial=[(r,p) for r,p in zip(rows,predictions) if len(r['allowed_codes'])>1]
    partial_correct=sum(p in r['allowed_codes'] for r,p in partial)
    group_names=sorted({r['source_group'] for r in rows})
    group_accuracy={g:sum(p in r['allowed_codes'] for r,p in zip(rows,predictions) if r['source_group']==g)
        /sum(r['source_group']==g for r in rows) for g in group_names}
    external=[(r,p) for r,p in zip(rows,predictions) if r['external']]
    return {'exact_six_class':metrics,'partial_floodplain':{'total':len(partial),'family_correct':partial_correct,
            'drainage_subclass_accuracy':None},'source_group_accuracy':group_accuracy,
            'mean_source_group_accuracy':sum(group_accuracy.values())/len(group_accuracy),
            'external_caption_consistency':{'total':len(external),'allowed_set_correct':sum(p in r['allowed_codes'] for r,p in external)}}
