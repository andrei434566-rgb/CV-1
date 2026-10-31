"""Common six-class and per-well preservation checks; absent classes are not invented."""
from scripts.kernel_facies_head import all_class_gates
from scripts.partial_facies_ridge import mixed_metrics
from scripts import train_unibo_visual_facies as source


def controls(val,held,vp,bp,hp,hb,training_gates):
    expected=[r['conversations'][1]['value'] for r in val]
    vm,bm=source.classification_metrics(expected,vp),source.classification_metrics(expected,bp)
    hm,hbm=mixed_metrics(held,hp,source),mixed_metrics(held,hb,source)
    gates=all_class_gates(vm,bm)
    gates.update(train_eligible=all(training_gates.values()),accuracy=vm['correct']>=bm['correct'],no_previously_correct_case_lost=not any(b==e and c!=e for e,b,c in zip(expected,bp,vp)))
    wg={}
    for well in sorted({r['well'] for r in held}):
        for c in sorted({r['allowed_codes'][0] for r in held if r['well']==well}):
            wg[well+'_'+c]=sum(p==c for r,p in zip(held,hp) if r['well']==well and r['allowed_codes']==[c])>=sum(p==c for r,p in zip(held,hb) if r['well']==well and r['allowed_codes']==[c])
    wg['at_least_one_gain']=hm['exact_six_class']['correct']>hbm['exact_six_class']['correct'];gates['new_wells_eligible']=all(wg.values())
    return dict(val=vm,baseline_val=bm,external_holdout=hm,baseline_external_holdout=hbm,validation_gates=gates,new_well_gates=wg)
