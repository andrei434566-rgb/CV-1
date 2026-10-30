"""Label-free local texture features and source-group-aware ridge helpers."""
from collections import Counter
import math
import numpy as np
from PIL import Image

PROFILES=('qwen_all','qwen_pre','qwen_post','qwen_texture')
ALPHAS=(.1,1.,10.,100.)
WEIGHTINGS=('class','class_group')
TEXTURE_VERSION='rgb_local_texture_v1'


def texture_features(image):
    """Relative image texture; no physical grain-size or genetic labels."""
    rgb=np.asarray(image.convert('RGB').resize((128,128),Image.Resampling.BILINEAR),dtype=np.float64)/255
    gray=rgb@np.array([.2126,.7152,.0722]); result=[]
    result.extend(rgb.mean((0,1))); result.extend(rgb.std((0,1)))
    result.extend(np.quantile(rgb,[.1,.5,.9],axis=(0,1)).ravel())
    for c in range(3): result.extend(np.histogram(rgb[:,:,c],bins=8,range=(0,1))[0]/gray.size)
    chroma=rgb/(rgb.sum(2,keepdims=True)+1e-9)
    result.extend(chroma.mean((0,1))); result.extend(chroma.std((0,1)))
    saturation=(rgb.max(2)-rgb.min(2))/(rgb.max(2)+1e-9)
    result.extend((saturation.mean(),saturation.std()))
    result.extend(np.histogram(gray,bins=16,range=(0,1))[0]/gray.size)
    result.extend(np.quantile(gray,[.1,.25,.5,.75,.9]))
    local=gray.reshape(8,16,8,16).transpose(0,2,1,3).std((2,3)).ravel()
    result.extend(np.quantile(local,[.1,.25,.5,.75,.9]))
    for radius in (1,2):
        center=gray[radius:-radius,radius:-radius]
        bits=np.stack([gray[radius+dy:128-radius+dy,radius+dx:128-radius+dx]>=center for dy,dx in (
            (0,radius),(radius,radius),(radius,0),(radius,-radius),(0,-radius),(-radius,-radius),(-radius,0),(-radius,radius))])
        transitions=np.count_nonzero(bits!=np.roll(bits,1,axis=0),axis=0)
        code=np.where(transitions<=2,bits.sum(0),9)
        result.extend(np.bincount(code.ravel(),minlength=10)/code.size)
    levels=np.minimum((gray*16).astype(int),15); ii,jj=np.indices((16,16))
    for distance in (1,2,4):
        pairs=np.concatenate(((levels[:,:-distance]*16+levels[:,distance:]).ravel(),
                              (levels[:-distance,:]*16+levels[distance:,:]).ravel()))
        counts=np.bincount(pairs,minlength=256).reshape(16,16).astype(float)
        p=(counts+counts.T)/(2*counts.sum())
        mx=(p*ii).sum(); my=(p*jj).sum()
        corr=(p*(ii-mx)*(jj-my)).sum()/math.sqrt(max(1e-18,(p*(ii-mx)**2).sum()*(p*(jj-my)**2).sum()))
        result.extend(((p*((ii-jj)/15)**2).sum(),(p/(1+abs(ii-jj))).sum(),(p*p).sum(),
                       -(p[p>0]*np.log(p[p>0])).sum(),corr))
    gy,gx=np.gradient(gray); magnitude=np.hypot(gx,gy)
    tensor=np.array([[np.mean(gx*gx),np.mean(gx*gy)],[np.mean(gx*gy),np.mean(gy*gy)]])
    eigen=np.linalg.eigvalsh(tensor)
    result.extend((magnitude.mean(),magnitude.std(),*np.quantile(magnitude,[.5,.9,.99]),
                   (eigen[1]-eigen[0])/(eigen.sum()+1e-12)))
    vector=np.asarray(result,dtype=np.float32)
    if not np.isfinite(vector).all(): raise ValueError('Nonfinite texture features')
    return vector


def make_folds(labels,groups,nfolds=3):
    """Stratify GROUPS using train labels only; rarest-class groups seeded."""
    y=np.asarray(labels,dtype=int); groups=np.asarray(groups)
    names=sorted(set(groups.tolist())); counts=np.array([[np.sum((groups==g)&(y==c)) for c in range(6)] for g in names])
    support=(counts>0).sum(0)
    if min(support)<nfolds: raise ValueError('Insufficient independent source groups for all-class CV')
    rare=int(np.argmin(support)); best=None
    for attempt in range(32):
        rng=np.random.default_rng(20261030+attempt); order=list(np.where(counts[:,rare]>0)[0]); rng.shuffle(order)
        mapping={}; totals=np.zeros((nfolds,6)); total=counts.sum(0)
        for j,index in enumerate(order):
            fold=j%nfolds; mapping[index]=fold; totals[fold]+=counts[index]
        remaining=[i for i in range(len(names)) if i not in mapping]; rng.shuffle(remaining)
        remaining.sort(key=lambda i:-counts[i].sum())
        for index in remaining:
            options=[]
            for fold in range(nfolds):
                trial=totals.copy(); trial[fold]+=counts[index]
                score=np.std(trial/total,axis=0).mean()+.02*np.std(trial.sum(1)/counts.sum())
                options.append((score,totals[fold].sum(),fold))
            fold=min(options)[2]; mapping[index]=fold; totals[fold]+=counts[index]
        if np.any(totals==0): continue
        score=float(np.std(totals/total,axis=0).mean())
        if best is None or score<best[0]: best=(score,{names[i]:f for i,f in mapping.items()})
    if best is None: raise ValueError('Cannot construct complete source-group folds')
    assignments=np.array([best[1][g] for g in groups])
    return assignments,best[1]


def sample_weights(labels,groups,mode):
    import torch
    values=labels.tolist(); n=len(values)
    if set(values)!=set(range(6)) or len(groups)!=n: raise ValueError('All six classes and aligned groups required')
    counts=Counter(values)
    if mode=='class': weights=[n/(6*counts[c]) for c in values]
    elif mode=='class_group':
        support={c:len({g for cc,g in zip(values,groups) if cc==c}) for c in range(6)}
        pairs=Counter(zip(values,groups)); weights=[n/(6*support[c]*pairs[c,g]) for c,g in zip(values,groups)]
    else: raise ValueError('Unknown weighting')
    return torch.tensor(weights,dtype=torch.float64)


def profile_spec(name,qdim,pre_dim,tdim):
    import torch
    if name=='qwen_all': columns=list(range(qdim)); sizes=[qdim]
    elif name=='qwen_pre': columns=list(range(pre_dim)); sizes=[pre_dim]
    elif name=='qwen_post': columns=list(range(pre_dim,qdim)); sizes=[qdim-pre_dim]
    elif name=='qwen_texture': columns=list(range(qdim+tdim)); sizes=[qdim,tdim]
    else: raise ValueError('Unknown feature profile')
    return torch.tensor(columns),torch.tensor(sizes),torch.full((len(sizes),),1/math.sqrt(len(sizes)),dtype=torch.float64)


def normalize(raw,state):
    import torch
    z=(raw.double().index_select(1,state['columns'])-state['mean'])/state['scale']
    blocks=[]; offset=0
    for size,weight in zip(state['block_sizes'].tolist(),state['block_weights']):
        block=z[:,offset:offset+size]; offset+=size
        blocks.append(block/block.norm(dim=1,keepdim=True).clamp_min(1e-12)*weight)
    if offset!=z.shape[1]: raise ValueError('Feature block dimensions differ')
    values=torch.cat(blocks,1)
    return torch.cat((values,torch.ones((len(values),1),dtype=torch.float64)),1)


def prepare_state(raw,profile,qdim,pre_dim,tdim):
    columns,sizes,weights=profile_spec(profile,qdim,pre_dim,tdim)
    selected=raw.double().index_select(1,columns)
    return {'columns':columns,'block_sizes':sizes,'block_weights':weights,'mean':selected.mean(0),
            'scale':selected.std(0,unbiased=False).clamp_min(1e-3)}


def ridge_coefficients(x,labels,weights,alpha):
    import torch
    if alpha<=0: raise ValueError('Positive regularization required')
    weighted=x*weights.sqrt()[:,None]
    y=torch.nn.functional.one_hot(labels,6).double()*weights.sqrt()[:,None]
    dual=torch.linalg.solve(weighted@weighted.T+alpha*torch.eye(len(x),dtype=torch.float64),y)
    return weighted.T@dual


def predict(raw,state): return normalize(raw,state)@state['coefficients']
