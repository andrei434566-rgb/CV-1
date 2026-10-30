import unittest
import torch

from scripts.conservative_facies_update import proximal_fit,preservation,training_gates
from scripts.partial_facies_ridge import fit_partial


class ConservativeUpdateTests(unittest.TestCase):
    def test_mixed_prior_solution_matches_coupled_primal_system(self):
        g=torch.Generator().manual_seed(73)
        x=torch.randn(15,5,generator=g,dtype=torch.float64)
        allowed=[[c] for c in 'ABCDEFABCDEF']+[['A','B']]*3
        weights=torch.linspace(.5,2.,15,dtype=torch.float64)
        prior=torch.randn(5,6,generator=g,dtype=torch.float64);alpha=.7;proximity=3.
        beta=proximal_fit(x,allowed,weights,prior,proximity,alpha)
        matrix=(alpha+proximity)*torch.eye(30,dtype=torch.float64)
        rhs=proximity*prior.T.flatten()
        for xx,codes,w in zip(x,allowed,weights):
            if len(codes)==1:
                l=torch.eye(6,dtype=torch.float64);target=l['ABCDEF'.index(codes[0])]
            else:
                l=torch.zeros(5,6,dtype=torch.float64);l[0,:2]=1.;l[1:,2:]=torch.eye(4,dtype=torch.float64)
                target=torch.tensor([1.,0,0,0,0],dtype=torch.float64)
            design=torch.kron(l,xx[None,:])
            matrix+=w*(design.T@design);rhs+=w*(design.T@target)
        expected=torch.linalg.solve(matrix,rhs).reshape(6,5).T
        torch.testing.assert_close(beta,expected,rtol=1e-10,atol=1e-10)

    def test_no_new_data_reproduces_prior_ridge(self):
        g=torch.Generator().manual_seed(74);x=torch.randn(18,7,generator=g,dtype=torch.float64)
        a=[[c] for c in 'ABCDEF'*3];w=torch.ones(18,dtype=torch.float64)
        prior=fit_partial(x,a,w,1.)
        torch.testing.assert_close(proximal_fit(x,a,w,prior,100.),prior,rtol=1e-10,atol=1e-10)

    def test_f1_regression_rejected_even_when_recall_is_preserved(self):
        base=dict(per_class_recall=dict.fromkeys('ABCDEF',.5),per_class_f1=dict.fromkeys('ABCDEF',.5),macro_f1=.5)
        candidate=dict(per_class_recall=dict.fromkeys('ABCDEF',.5),per_class_f1=dict.fromkeys('ABCDEF',.5),macro_f1=.6)
        candidate['per_class_recall']['B']=.6;candidate['per_class_recall']['C']=.6
        candidate['per_class_f1']['F']=.4
        self.assertFalse(preservation(candidate,base)['F_f1'])
        self.assertFalse(all(training_gates(candidate,base).values()))


if __name__=='__main__':unittest.main()
