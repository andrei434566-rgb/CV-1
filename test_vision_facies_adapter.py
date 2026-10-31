import copy
from types import SimpleNamespace
import unittest
import torch
from torch import nn
from scripts.vision_facies_adapter import add_lora,label_loss,post_features,adapter_state,restore_adapter


class TinyVision(nn.Module):
    def __init__(self):
        super().__init__();self.blocks=nn.ModuleList()
        for _ in range(2):
            block=nn.Module();block.attn=nn.Module();block.attn.qkv=nn.Linear(8,24);block.attn.proj=nn.Linear(8,8);self.blocks.append(block)
    def forward(self,x):
        for b in self.blocks:x=b.attn.qkv(x)[:,:8]+b.attn.proj(x)
        return x


class VisionAdapterTests(unittest.TestCase):
    def test_family_loss_does_not_assign_drainage_subclass(self):
        x=torch.tensor([[.2,1.3,-.4,.1,.3,.5]],requires_grad=True)
        a=label_loss(x,[['A','B']],torch.ones(1));b=label_loss(x[:,[1,0,2,3,4,5]],[['A','B']],torch.ones(1))
        torch.testing.assert_close(a,b)
        expected=-torch.log(torch.softmax(x,1)[0,:2].sum());torch.testing.assert_close(a,expected)

    def test_exact_loss_matches_weighted_cross_entropy(self):
        x=torch.randn(3,6);w=torch.tensor([.4,1.,2.]);labels=[['A'],['D'],['F']]
        expected=(nn.functional.cross_entropy(x,torch.tensor([0,3,5]),reduction='none')*w).mean()
        torch.testing.assert_close(label_loss(x,labels,w),expected)

    def test_zero_lora_original_invariance_real_updates_and_reload(self):
        torch.manual_seed(92);model=TinyVision();original=copy.deepcopy(model);model.requires_grad_(False)
        x=torch.randn(4,8);before=original(x).detach();names=add_lora(model,rank=2,alpha=4,last_blocks=1)
        self.assertEqual(names,['blocks.1.attn.qkv','blocks.1.attn.proj'])
        torch.testing.assert_close(model(x),before,rtol=0,atol=0)
        parameters=[p for p in model.parameters() if p.requires_grad];opt=torch.optim.SGD(parameters,lr=.1)
        model(x).square().sum().backward();self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in parameters));opt.step()
        self.assertTrue(all(p.grad is None for n,p in model.named_parameters() if not p.requires_grad))
        delta=torch.zeros(9,6);norm=dict(mean=torch.zeros(8),scale=torch.ones(8));state=adapter_state(model,delta,norm,delta)
        fresh=copy.deepcopy(original);fresh.requires_grad_(False);add_lora(fresh,rank=2,alpha=4,last_blocks=1);restore_adapter(fresh,state)
        torch.testing.assert_close(model(x),fresh(x),rtol=0,atol=0)
        broken=dict(state);broken.pop(next(k for k in state if k.endswith('lora_b')))
        with self.assertRaises(ValueError):restore_adapter(fresh,broken)

    def test_variable_image_token_pooling_does_not_mix_images(self):
        values=torch.tensor([[2.,4],[10.,20],[14.,28]],requires_grad=True)
        class Fake:
            def __call__(self,*a,**kw):return SimpleNamespace(pooler_output=values)
        result=post_features(Fake(),dict(pixel_values=torch.empty(0),grid=torch.tensor([[1,2,2],[1,2,4]])))
        torch.testing.assert_close(result,torch.tensor([[2.,4.,0.,0.],[12.,24.,2.,4.]]))
        result[0].sum().backward();self.assertTrue(torch.equal(values.grad[1:],torch.zeros(2,2)))


if __name__=='__main__':unittest.main()
