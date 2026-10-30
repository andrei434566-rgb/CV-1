"""Ridge continuation with a prior head and explicit per-class gates.

Teacher scores are a regularizer, never source labels. True source labels
are retained, including the unsupervised A-B direction for partial rows.
"""
from scripts.partial_facies_ridge import label_system

PROTECTED=('A','D','E','F')


def proximal_fit(x,allowed,weights,prior,proximity,alpha=1.):
    import torch
    if proximity<0 or alpha<=0:raise ValueError('Invalid penalty')
    y,precision,rotation=label_system(allowed)
    prior_rot=prior@rotation;penalty=alpha+proximity;out=[]
    for axis in range(6):
        w=weights*precision[:,axis];active=w>0;sw=w[active].sqrt()
        xx=x[active]*sw[:,None];center=prior_rot[:,axis]*(proximity/penalty)
        rhs=(y[active,axis]-x[active]@center)*sw
        dual=torch.linalg.solve(xx@xx.T+penalty*torch.eye(len(xx),dtype=torch.float64),rhs)
        out.append(center+xx.T@dual)
    return torch.stack(out,1)@rotation.T


def preservation(candidate,baseline):
    result={}
    for c in PROTECTED:
        result[c+'_recall']=candidate['per_class_recall'][c]+1e-12>=baseline['per_class_recall'][c]
        result[c+'_f1']=candidate['per_class_f1'][c]+1e-12>=baseline['per_class_f1'][c]
    return result


def training_gates(candidate,baseline):
    result=preservation(candidate,baseline)
    result['B_improved']=candidate['per_class_recall']['B']>baseline['per_class_recall']['B']+1e-12
    result['C_improved']=candidate['per_class_recall']['C']>baseline['per_class_recall']['C']+1e-12
    result['macro_f1']=candidate['macro_f1']+1e-12>=baseline['macro_f1']
    return result


def selection_key(item):
    m=item['old_train_oof'];b=item['baseline_old_train_oof']
    recall=min(m['per_class_recall'][c]-b['per_class_recall'][c] for c in PROTECTED)
    f1=min(m['per_class_f1'][c]-b['per_class_f1'][c] for c in PROTECTED)
    gains=[m['per_class_recall'][c]-b['per_class_recall'][c] for c in ('B','C')]
    return (all(item['training_gates'].values()),recall,f1,min(gains),sum(gains),m['macro_f1'],item['proximity'],-item['new_weight'])
