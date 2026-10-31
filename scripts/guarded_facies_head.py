"""A class guard preserves all teacher decisions of selected classes, including errors."""
import itertools


def guard_sets():
    return [tuple(c for c,b in zip('ABCDEF',bits) if b) for bits in itertools.product((False,True),repeat=6)]


def guarded_predictions(teacher_scores,candidate_scores,guards):
    if teacher_scores.shape!=candidate_scores.shape or teacher_scores.ndim!=2 or teacher_scores.shape[1]!=6:raise ValueError('Aligned six-class scores required')
    if any(c not in 'ABCDEF' for c in guards) or len(set(guards))!=len(guards):raise ValueError('Distinct valid guard classes required')
    t=teacher_scores.argmax(1);c=candidate_scores.argmax(1)
    return ['ABCDEF'[int(old if 'ABCDEF'[int(old)] in guards else new)] for old,new in zip(t,c)]


def selection_key(item):
    m,b=item['old_oof'],item['baseline_old_oof']
    return (all(item['all_class_gates'].values()),
            min(m['per_class_recall'][c]-b['per_class_recall'][c] for c in 'ABCDEF'),
            min(m['per_class_f1'][c]-b['per_class_f1'][c] for c in 'ABCDEF'),
            m['macro_f1'],item['mean_group_accuracy'],-len(item['guard_classes']),tuple(item['guard_classes']))
