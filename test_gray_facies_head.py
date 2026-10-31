import unittest
import torch
from PIL import Image
from scripts.gray_facies_head import grayscale_view,blend_scores,pack_state,predict_features,candidate_key


class GrayFaciesTests(unittest.TestCase):
    def test_conversion_has_fixed_luma_and_preserves_original(self):
        image=Image.new('RGB',(3,1));image.putdata([(255,0,0),(0,255,0),(0,0,255)])
        before=image.tobytes();gray=grayscale_view(image)
        self.assertEqual(list(gray.getdata()),[(76,76,76),(150,150,150),(29,29,29)])
        self.assertEqual(image.tobytes(),before)
        self.assertEqual(gray.size,image.size)

    def test_two_view_packed_inference_matches_independent_scores(self):
        def state(columns,coeff):
            return dict(columns=torch.tensor(columns),block_sizes=torch.tensor([2]),block_weights=torch.ones(1,dtype=torch.float64),
                        mean=torch.zeros(2,dtype=torch.float64),scale=torch.ones(2,dtype=torch.float64),coefficients=coeff.double())
        rgb=state([0,1],torch.arange(18).reshape(3,6));gray=state([1,2],torch.arange(18,36).reshape(3,6))
        x=torch.tensor([[3.,4.,8.]],dtype=torch.float64);y=torch.tensor([[9.,5.,12.]],dtype=torch.float64)
        rgb_score=torch.tensor([[.6,.8,1.]],dtype=torch.float64)@rgb['coefficients']
        gray_score=torch.tensor([[5/13,12/13,1.]],dtype=torch.float64)@gray['coefficients']
        torch.testing.assert_close(predict_features(x,y,pack_state(rgb,gray,.25)),.75*rgb_score+.25*gray_score)
        with self.assertRaises(ValueError):blend_scores(rgb_score,gray_score[:,:5],.25)
        with self.assertRaises(ValueError):blend_scores(rgb_score,gray_score,0)

    def test_macro_gain_cannot_override_one_class_regression(self):
        base=dict(per_class_recall=dict.fromkeys('ABCDEF',.5),per_class_f1=dict.fromkeys('ABCDEF',.5),macro_f1=.5)
        good=dict(old_oof=base,baseline_old_oof=base,all_class_gates=dict(all_preserved=True),mean_group_accuracy=.5,gray_weight=.1,alpha=1.)
        bad=dict(good,old_oof=dict(base,per_class_recall=dict(base['per_class_recall'],C=.4),macro_f1=.9),all_class_gates=dict(all_preserved=False),mean_group_accuracy=.9)
        self.assertGreater(candidate_key(good),candidate_key(bad))
        # Development-control scores are irrelevant to train-only selection.
        a=dict(good,validation_accuracy=0.);b=dict(good,validation_accuracy=1.)
        self.assertEqual(candidate_key(a),candidate_key(b))


if __name__=='__main__':unittest.main()
