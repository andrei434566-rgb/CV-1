"""CPU checks for the frozen-encoder head, including independent ridge parity."""
from types import SimpleNamespace
import unittest
import torch
from scripts.train_qwen_visual_facies_head import pool_features,fit_head,predict_head,normalized,RECIPE


class VisualHeadTests(unittest.TestCase):
    def test_pooled_features_include_population_std_without_label_information(self):
        a=torch.tensor([[1.,2.],[3.,4.]])
        b=torch.tensor([[2.,4.,6.],[4.,8.,12.]])
        pooled=pool_features(SimpleNamespace(last_hidden_state=a,pooler_output=b))
        self.assertTrue(torch.allclose(pooled,torch.tensor([2.,3.,1.,1.,3.,6.,9.,1.,2.,3.])))
        reverse=pool_features(SimpleNamespace(last_hidden_state=a.flip(0),pooler_output=b.flip(0)))
        self.assertTrue(torch.equal(pooled,reverse))

    def test_weighted_dual_solution_matches_independent_primal_equations(self):
        torch.manual_seed(4)
        features=torch.randn(18,9)
        labels=torch.tensor([0]*6+[1]*4+[2]*3+[3]*2+[4]+[5]*2)
        state=fit_head(features,labels)
        x=normalized(features,state['mean'],state['scale'])
        counts=torch.bincount(labels,minlength=6).double()
        weights=len(labels)/(6*counts[labels])
        y=torch.nn.functional.one_hot(labels,6).double()
        lhs=x.T@(weights[:,None]*x)+RECIPE['ridge_lambda']*torch.eye(x.shape[1],dtype=torch.float64)
        rhs=x.T@(weights[:,None]*y)
        expected=torch.linalg.solve(lhs,rhs)
        self.assertTrue(torch.allclose(state['coefficients'],expected,atol=1e-10,rtol=1e-9))

    def test_head_can_learn_distinct_visual_feature_clusters(self):
        features=torch.eye(6).repeat_interleave(6,0)
        labels=torch.arange(6).repeat_interleave(6)
        state=fit_head(features,labels)
        self.assertTrue(torch.equal(predict_head(features,state).argmax(-1),labels))

    def test_validation_outlier_does_not_refit_or_mutate_normalizer(self):
        features=torch.eye(6).repeat(2,1); labels=torch.arange(6).repeat(2)
        state=fit_head(features,labels); saved={k:v.clone() for k,v in state.items()}
        scores=predict_head(torch.full((2,6),1e9),state)
        self.assertTrue(torch.isfinite(scores).all())
        self.assertTrue(all(torch.equal(v,saved[k]) for k,v in state.items()))

    def test_constant_features_remain_finite_and_missing_class_is_refused(self):
        features=torch.ones(12,8); labels=torch.arange(6).repeat(2)
        self.assertTrue(torch.isfinite(predict_head(features,fit_head(features,labels))).all())
        with self.assertRaises(ValueError): fit_head(features,torch.zeros(12,dtype=torch.long))
        features[0,0]=float('nan')
        with self.assertRaises(ValueError): fit_head(features,labels)


if __name__=='__main__': unittest.main()
