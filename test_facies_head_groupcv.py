"""Checks for group isolation, source coverage and texture/ridge math."""
from collections import Counter
import unittest
import numpy as np
import torch
from PIL import Image
from scripts.facies_head_v2_features import (
    texture_features,make_folds,sample_weights,prepare_state,normalize,ridge_coefficients,predict,
)
from scripts.train_facies_head_groupcv import cross_validate
from scripts.train_qwen_visual_facies_head import fit_head,predict_head
from scripts.expand_unibo_facies_train import overlap_min,windows
from scripts.prepare_unibo_vlm_facies import box_iou


def cohort():
    labels=[]; groups=[]
    for g in range(9):
        for c in range(6):
            if c==2 and g>=3: continue
            for _ in range(2): labels.append(c); groups.append('S'+str(g))
    return torch.tensor(labels),groups


class GroupCVTests(unittest.TestCase):
    def test_rare_source_groups_are_held_out_whole_with_every_class_present(self):
        labels,groups=cohort(); assignment,mapping=make_folds(labels.numpy(),groups)
        other,again=make_folds(labels.numpy(),groups)
        self.assertEqual(mapping,again); self.assertTrue(np.array_equal(assignment,other))
        for fold in range(3):
            held={g for g,f in zip(groups,assignment) if f==fold}; train={g for g,f in zip(groups,assignment) if f!=fold}
            self.assertFalse(held&train)
            self.assertEqual(set(labels[assignment==fold].tolist()),set(range(6)))
            self.assertEqual(set(labels[assignment!=fold].tolist()),set(range(6)))
        with self.assertRaises(ValueError): make_folds(np.arange(6),['one']*6)

    def test_group_weights_equalize_class_mass_and_group_mass_within_class(self):
        labels,groups=cohort(); labels=torch.cat((labels,labels[:4])); groups=groups+groups[:4]
        weights=sample_weights(labels,groups,'class_group')
        for c in range(6):
            self.assertAlmostEqual(float(weights[labels==c].sum()),len(labels)/6)
            amounts=[float(weights[torch.tensor([(int(cc)==c and gg==g) for cc,gg in zip(labels,groups)])].sum()) for g in set(groups) if any(int(cc)==c and gg==g for cc,gg in zip(labels,groups))]
            self.assertLess(max(amounts)-min(amounts),1e-10)

    def test_v1_baseline_is_reproduced_by_full_qwen_profile(self):
        torch.manual_seed(2); raw=torch.randn(36,11); labels=torch.arange(6).repeat(6)
        old=fit_head(raw[:,:8],labels)
        state=prepare_state(raw,'qwen_all',8,4,3)
        state['coefficients']=ridge_coefficients(normalize(raw,state),labels,sample_weights(labels,['S']*36,'class'),1.)
        self.assertTrue(torch.allclose(predict(raw,state),predict_head(raw[:,:8],old),atol=1e-10))

    def test_fusion_balances_blocks_without_dimension_dominance(self):
        raw=torch.randn(18,12); state=prepare_state(raw,'qwen_texture',8,4,4)
        x=normalize(raw,state)
        self.assertTrue(torch.allclose(x[:,:8].norm(dim=1),torch.full((18,),2**-.5,dtype=torch.float64)))
        self.assertTrue(torch.allclose(x[:,8:12].norm(dim=1),torch.full((18,),2**-.5,dtype=torch.float64)))
        self.assertTrue(torch.equal(x[:,-1],torch.ones(18,dtype=torch.float64)))

    def test_texture_is_finite_fixed_size_and_rotation_invariant(self):
        rng=np.random.default_rng(1); image=Image.fromarray(rng.integers(0,256,(128,128,3),dtype=np.uint8))
        feature=texture_features(image)
        for color in ('black','white','gray'):
            other=texture_features(Image.new('RGB',(80,120),color))
            self.assertEqual(feature.shape,other.shape); self.assertTrue(np.isfinite(other).all())
        self.assertTrue(np.allclose(feature,texture_features(image.transpose(Image.Transpose.ROTATE_90)),atol=1e-6))

    def test_extended_windows_preserve_source_purity_and_overlap_prevents_recycling(self):
        mask=np.ones((160,260),dtype=np.uint8); mask[:8,:]=0; mask[:3,:]=255
        for c,box,q in windows(mask):
            x0,y0,x1,y1=box; crop=mask[y0:y1,x0:x1]
            self.assertEqual(c,1); self.assertGreaterEqual(np.mean(crop==1),.98)
            self.assertLessEqual(np.mean(crop==255),.005)
        a=(0,0,256,256); b=(192,0,320,64)
        self.assertLess(box_iou(a,b),.1); self.assertGreater(overlap_min(a,b),.2)

    def test_group_cv_predicts_only_unseen_groups_and_complete_oof_rows(self):
        labels,groups=cohort(); torch.manual_seed(3)
        raw=torch.cat((torch.nn.functional.one_hot(labels,6).float()+.01*torch.randn(len(labels),6),torch.randn(len(labels),3)*.01),1)
        winner,results,mapping,oof=cross_validate(raw,labels,groups,6,3,3,lambda *a,**k:None)
        self.assertEqual(len(results),32); self.assertEqual(len(oof),len(labels))
        self.assertEqual(winner['metrics']['correct'],len(labels)); self.assertTrue(torch.equal(oof,labels))
        self.assertEqual(len(set(mapping.values())),3)


if __name__=='__main__': unittest.main()
