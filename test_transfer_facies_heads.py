import unittest
import torch
from PIL import Image
from scripts.dino_facies_head import letterbox,tokens_in_box,pool_tokens
from scripts.guarded_facies_head import guarded_predictions,guard_sets
from scripts.residual_facies_head import make_network,score
from scripts.prepare_paired_context_v16 import context_box
from scripts.margin_facies_gate import predictions as margin_predictions


class TransferHeadTests(unittest.TestCase):
    def test_context_expands_only_along_declared_core_axis(self):
        self.assertEqual(context_box([100,100,200,200],(1000,400),axis='x'),[0,100,300,200])
        self.assertEqual(context_box([100,100,200,200],(1000,400),axis='y'),[100,0,200,300])
        self.assertEqual(context_box([20,200,80,300],(100,400),y_limits=(150,350)),[20,150,80,350])

    def test_margin_gate_retains_teacher_only_for_small_raw_gap(self):
        teacher=torch.tensor([[1.,0,0,0,0,0],[1.,0,0,0,0,0]])
        candidate=torch.tensor([[.5,.51,0,0,0,0],[.1,.9,0,0,0,0]])
        self.assertEqual(margin_predictions(teacher,candidate,.05),['A','B'])
        self.assertEqual(margin_predictions(teacher,candidate,0),['B','B'])

    def test_letterbox_preserves_aspect_and_mask_excludes_padding(self):
        image=Image.new('RGB',(50,100),(25,40,90));canvas,box=letterbox(image)
        self.assertEqual(canvas.size,(224,224));self.assertEqual(box,(56,0,168,224))
        mask=tokens_in_box(box);self.assertEqual(int(mask.sum()),128)
        hidden=torch.zeros(257,3);hidden[0]=7;hidden[1:]=100;hidden[1:][mask]=torch.tensor([2.,4.,6.])
        torch.testing.assert_close(pool_tokens(hidden,mask),torch.tensor([7.,7.,7.,2.,4.,6.,0.,0.,0.]))
        self.assertTrue(torch.equal(hidden[1:][~mask],torch.full((128,3),100.)))

    def test_guard_keeps_selected_teacher_class_but_does_not_use_truth(self):
        teacher=torch.eye(6);candidate=teacher.roll(1,1)
        pred=guarded_predictions(teacher,candidate,['C','D'])
        self.assertEqual(pred,['B','C','C','D','F','A'])
        self.assertEqual(guarded_predictions(teacher,candidate,list('ABCDEF')),list('ABCDEF'))
        self.assertEqual(len(set(guard_sets())),64)
        with self.assertRaises(ValueError):guarded_predictions(teacher,candidate,['C','C'])

    def test_residual_starts_as_teacher_then_learns_with_bounded_correction(self):
        torch.manual_seed(3);net=make_network();features=torch.randn(4,1152);teacher=torch.randn(4,6)
        torch.testing.assert_close(score(features,teacher,net),teacher,rtol=0,atol=0)
        opt=torch.optim.AdamW(net.parameters(),lr=.02);loss=torch.nn.functional.cross_entropy(score(features,teacher,net)*8,torch.tensor([0,1,2,3]));loss.backward()
        self.assertGreater(float(net[-1].weight.grad.norm()),0);opt.step();net.eval()
        value=score(features,teacher,net);self.assertGreater(float((value-teacher).abs().max()),0)
        self.assertLessEqual(float((value-teacher).abs().max()),.150001)
        fresh=make_network();fresh.load_state_dict(net.state_dict());fresh.eval();torch.testing.assert_close(score(features,teacher,fresh),value,rtol=0,atol=0)


if __name__=='__main__':unittest.main()
