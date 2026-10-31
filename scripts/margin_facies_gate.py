"""Conservative switch based on a raw score gap; not calibrated probability."""


def predictions(teacher,candidate,threshold):
    if teacher.shape!=candidate.shape or teacher.ndim!=2 or teacher.shape[1]!=6 or threshold<0:raise ValueError('Aligned scores and nonnegative threshold required')
    top=candidate.topk(2,dim=1);margin=top.values[:,0]-top.values[:,1]
    old,new=teacher.argmax(1),candidate.argmax(1)
    return ['ABCDEF'[int(c if gap>=threshold else t)] for t,c,gap in zip(old,new,margin)]


def key(item):
    m,b=item['old_oof'],item['baseline_old_oof']
    return (all(item['all_class_gates'].values()),item['corrected']-2*item['introduced'],
            min(m['per_class_f1'][c]-b['per_class_f1'][c] for c in 'ABCDEF'),m['macro_f1'],item['mean_group_accuracy'],item['threshold'])
