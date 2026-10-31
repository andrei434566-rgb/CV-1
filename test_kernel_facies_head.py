import unittest
import torch
from scripts.kernel_facies_head import partial_dual,kernel,training_bandwidth,grouped_weights,all_class_gates


class KernelFaciesTests(unittest.TestCase):
    def test_partial_kernel_solution_matches_independent_coupled_primal(self):
        g=torch.Generator().manual_seed(82);phi=torch.randn(10,4,generator=g,dtype=torch.float64)
        allowed=[[c] for c in 'ABCDEFAB']+[['A','B'],['A','B']];w=torch.linspace(.2,2,10,dtype=torch.float64);alpha=.3
        gram=phi@phi.T;dual=partial_dual(gram,allowed,w,alpha)
        matrix=alpha*torch.eye(24,dtype=torch.float64);rhs=torch.zeros(24,dtype=torch.float64)
        for x,a,ww in zip(phi,allowed,w):
            if len(a)==1:l=torch.eye(6,dtype=torch.float64);target=l['ABCDEF'.index(a[0])]
            else:
                l=torch.zeros(5,6,dtype=torch.float64);l[0,:2]=1;l[1:,2:]=torch.eye(4,dtype=torch.float64);target=torch.tensor([1.,0,0,0,0],dtype=torch.float64)
            design=torch.kron(l,x[None,:]);matrix+=ww*design.T@design;rhs+=ww*design.T@target
        beta=torch.linalg.solve(matrix,rhs).reshape(6,4).T
        torch.testing.assert_close(gram@dual,phi@beta,atol=1e-10,rtol=1e-10)

    def test_rbf_bandwidth_and_psd(self):
        x=torch.tensor([[0.,1],[1.,0],[2.,1]],dtype=torch.float64);b=training_bandwidth(x);k=kernel(x,x,b,2.)
        torch.testing.assert_close(k,k.T);torch.testing.assert_close(k.diag(),torch.ones(3,dtype=torch.float64))
        self.assertGreaterEqual(float(torch.linalg.eigvalsh(k).min()),-1e-12)
        with self.assertRaises(ValueError):training_bandwidth(torch.zeros(2,2))

    def test_grouped_weight_is_not_proportional_to_crop_count(self):
        rows=[]
        for c in 'ABCDEF':
            for group,n in [('small',1),('large',4)]:
                rows.extend(dict(allowed_codes=[c],source_group=group,external=False) for _ in range(n))
        w=grouped_weights(rows,'class_group')
        for c in 'ABCDEF':
            a=sum(float(ww) for r,ww in zip(rows,w) if r['allowed_codes']==[c] and r['source_group']=='small')
            b=sum(float(ww) for r,ww in zip(rows,w) if r['allowed_codes']==[c] and r['source_group']=='large')
            self.assertAlmostEqual(a,b)

    def test_gate_rejects_one_class_regression_despite_macro_gain(self):
        base=dict(per_class_recall=dict.fromkeys('ABCDEF',.5),per_class_f1=dict.fromkeys('ABCDEF',.5),macro_f1=.5)
        cand=dict(per_class_recall=dict.fromkeys('ABCDEF',.6),per_class_f1=dict.fromkeys('ABCDEF',.6),macro_f1=.6)
        cand['per_class_recall']['C']=.4
        self.assertFalse(all(all_class_gates(cand,base).values()))


if __name__=='__main__':unittest.main()
