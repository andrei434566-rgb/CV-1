"""Check genuinely partial supervision and its independent normal equations."""
import math
import unittest
import torch
from scripts.partial_facies_ridge import label_system,row_weights,fit_partial
from scripts.facies_head_v2_features import ridge_coefficients
from scripts.prepare_web_facies_20261031 import specifications


class PartialRidgeTests(unittest.TestCase):
    def test_exact_labels_reproduce_original_ridge(self):
        torch.manual_seed(8); x=torch.randn(18,9,dtype=torch.float64)
        labels=torch.arange(6).repeat(3); allowed=[[c] for c in 'ABCDEF'*3]
        weights=torch.ones(18,dtype=torch.float64)
        self.assertTrue(torch.allclose(fit_partial(x,allowed,weights,1.),ridge_coefficients(x,labels,weights,1.),atol=1e-11))

    def test_partial_loss_does_not_teach_an_A_or_B_subclass(self):
        target,precision,rotation=label_system([['A','B']])
        for a,b in ((1.,0.),(0.,1.),(.25,.75)):
            scores=torch.tensor([[a,b,0,0,0,0]],dtype=torch.float64)
            loss=(((scores@rotation-target)**2)*precision).sum()
            self.assertLess(float(loss),1e-25)
        wrong=torch.tensor([[0.,0,1,0,0,0]],dtype=torch.float64)
        self.assertAlmostEqual(float((((wrong@rotation-target)**2)*precision).sum()),2.)
        with self.assertRaises(ValueError): label_system([['C','D']])

    def test_mixed_solution_matches_independent_primal_equations(self):
        torch.manual_seed(9); x=torch.randn(20,7,dtype=torch.float64)
        allowed=[[c] for c in 'ABCDEF'*3]+[['A','B']]*2
        weights=row_weights(allowed,[False]*18+[True]*2,4.)
        result=fit_partial(x,allowed,weights,10.)
        y,precision,rotation=label_system(allowed)
        expected=[]
        for j in range(6):
            w=weights*precision[:,j]
            lhs=x.T@(w[:,None]*x)+10*torch.eye(7,dtype=torch.float64)
            expected.append(torch.linalg.solve(lhs,x.T@(w*y[:,j])))
        self.assertTrue(torch.allclose(result,torch.stack(expected,1)@rotation.T,atol=1e-11))

    def test_curated_boxes_exclude_known_overlay_regions_and_do_not_overlap(self):
        rows=specifications(); self.assertEqual(len(rows),24)
        self.assertEqual(sum(r['allowed_codes']==['A','B'] for r in rows),6)
        for r in rows:
            x0,y0,x1,y1=r['box']
            if r['panel']=='8C': self.assertGreaterEqual(x0,990); self.assertLessEqual(x1,1248); self.assertGreaterEqual(y0,60)
            if r['panel']=='3-column8': self.assertLessEqual(x1,2415)
        for i,a in enumerate(rows):
            for b in rows[i+1:]:
                if a['file']!=b['file']: continue
                x0,y0,x1,y1=a['box']; u0,v0,u1,v1=b['box']
                intersection=max(0,min(x1,u1)-max(x0,u0))*max(0,min(y1,v1)-max(y0,v0))
                self.assertEqual(intersection,0)


if __name__=='__main__': unittest.main()
