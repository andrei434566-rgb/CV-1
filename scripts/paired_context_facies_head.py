"""Learn from both focal rock texture and adjacent same-photo core context."""
import math


def fit_head(raw,rows,alpha):
    import torch
    from scripts.facies_head_v2_features import normalize
    from scripts.partial_facies_ridge import fit_partial
    from scripts.kernel_facies_head import grouped_weights
    if raw.ndim!=2 or raw.shape[1]!=2304:raise ValueError('Expected focal1152 + context1152')
    x=raw.double();s=dict(columns=torch.arange(2304),block_sizes=torch.tensor([384]*6),block_weights=torch.full((6,),1/math.sqrt(6),dtype=torch.float64),mean=x.mean(0),scale=x.std(0,unbiased=False).clamp_min(1e-3))
    s['coefficients']=fit_partial(normalize(raw,s),[r['allowed_codes'] for r in rows],grouped_weights(rows,'class_group',.5),alpha)
    return s
