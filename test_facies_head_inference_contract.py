"""Portable research CLI retains v1 and checks the new frozen contract."""
import unittest
import torch
from PIL import Image
from scripts.classify_with_qwen_facies_head import V1,V2,validate_head_state,score_image_features
from scripts.train_qwen_visual_facies_head import fit_head,predict_head
from scripts.facies_head_v2_features import texture_features,prepare_state,normalize,sample_weights,ridge_coefficients,predict


class InferenceContractTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.image=Image.new('RGB',(120,80),'gray')
        self.q=torch.randn(18,8); self.labels=torch.arange(6).repeat(3)

    def test_v1_predictions_remain_identical(self):
        state=fit_head(self.q,self.labels)
        contract={'feature_contract_version':V1,'feature_dimension':8}
        validate_head_state(state,contract)
        self.assertTrue(torch.equal(score_image_features(self.q,self.image,state,contract),predict_head(self.q,state)))

    def test_v2_post_profile_uses_same_selected_features_as_training(self):
        texture=torch.from_numpy(texture_features(self.image))[None,:].repeat(18,1)
        raw=torch.cat((self.q,texture),1)
        state=prepare_state(raw,'qwen_post',8,4,texture.shape[1])
        state['coefficients']=ridge_coefficients(normalize(raw,state),self.labels,sample_weights(self.labels,['S']*18,'class'),1.)
        contract={'feature_contract_version':V2,'feature_dimension':4,'qwen_feature_dimension':8,
            'pre_merger_feature_dimension':4,'texture_dimension':texture.shape[1],'texture_version':'rgb_local_texture_v1',
            'selected_profile':{'profile':'qwen_post'}}
        validate_head_state(state,contract)
        result=score_image_features(self.q[:1],self.image,state,contract)
        self.assertTrue(torch.equal(result,predict(raw[:1],state)))
        # Texture is not part of this winning profile.
        self.assertTrue(torch.equal(result,score_image_features(self.q[:1],Image.new('RGB',(20,20),'red'),state,contract)))
        state['columns']=state['columns'].flip(0)
        with self.assertRaisesRegex(ValueError,'profile'): validate_head_state(state,contract)

    def test_invalid_normalizer_or_weights_are_rejected(self):
        state=fit_head(self.q,self.labels)
        contract={'feature_contract_version':V1,'feature_dimension':8}
        state['scale'][0]=0
        with self.assertRaises(ValueError): validate_head_state(state,contract)
        state['scale'][0]=1; state['coefficients'][0,0]=float('nan')
        with self.assertRaises(ValueError): validate_head_state(state,contract)


if __name__=='__main__': unittest.main()
