import json,unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from app.domain.bedding_types import PROFILES,reference_answer
from app.domain.lithology_attributes import LITHOLOGY_ATTRIBUTE_OPTIONS
from app.infrastructure.bedding_classifier import read_card,CLASSES,PREPROCESSING

class BeddingTests(unittest.TestCase):
    def test_vocabulary_matches_all_16_existing_fields(self):
        self.assertEqual([p[0] for p in PROFILES],LITHOLOGY_ATTRIBUTE_OPTIONS['Слоистость'])

    def test_compound_names_do_not_add_broader_wrong_definition(self):
        a=reference_answer('Что такое таблитчатая крупная косая слоистость?')
        self.assertTrue(a.startswith('Таблитчатая крупная косая\n'))
        self.assertNotIn('\n\nКрупная косая\n',a)
        a=reference_answer('Расскажи о линзовидно-волнистой слоистости')
        self.assertTrue(a.startswith('Линзовидно-волнистая\n'))
        self.assertNotIn('\n\nЛинзовидная\n',a)

    def test_comparison_uses_two_explicit_definitions(self):
        a=reference_answer('Отличия линзовидной и флазерной слоистости')
        self.assertIn('Линзовидная\n',a);self.assertIn('Флазерная\n',a)
        self.assertNotIn('https://',a)
        self.assertIn('https://',reference_answer('что такое флазерная слоистость',True))

    def test_no_generation_or_structure_from_generic_rock_question(self):
        self.assertEqual(reference_answer('Что такое песчаник'), '')
        a=reference_answer('Признаки градационной слоистости')
        self.assertIn('размера зёрен',a);self.assertIn('Градиент цвета сам по себе не подтверждает',a)

    def test_offline_reference_route_has_no_guessed_fields(self):
        from app.infrastructure.local_vision import reference_answer as route
        result=route({'question':'Что такое флазерная слоистость?'})
        self.assertEqual(result['engine'],'reference');self.assertIn('Флазерная',result['answer'])
        self.assertNotIn('https://',result['answer'])

    def test_wrong_model_or_16_fake_class_card_rejected(self):
        c={'schema':'cv1-bedding-cnn-v1','generative':False,'native_classes':list(CLASSES),
           'architecture':'efficientnet_b0','preprocessing':PREPROCESSING,'input_size':224,
           'deployment':'research_only','names_ru':['a']*5}
        with TemporaryDirectory() as d:
            p=Path(d)/'config.json'
            p.write_text(json.dumps(c),encoding='utf8');self.assertEqual(read_card(d)['native_classes'],list(CLASSES))
            for key,value in [('generative',True),('native_classes',[p[0] for p in PROFILES]),
                              ('deployment','automatic'),('preprocessing','squash_column')]:
                p.write_text(json.dumps(dict(c,**{key:value})),encoding='utf8')
                with self.assertRaises(ValueError):read_card(d)

    def test_whole_roi_preprocessing_preserves_aspect(self):
        from PIL import Image
        from app.infrastructure.bedding_classifier import prepare_image
        from app.infrastructure.vision_models import MEAN,STD
        import torch
        x=prepare_image(Image.new('RGB',(30,120),(255,0,0)))
        self.assertEqual(tuple(x.shape),(3,224,224))
        # Letterbox gray edges and red centre, not a square-squashed core.
        denorm=x*torch.tensor(STD)[:,None,None]+torch.tensor(MEAN)[:,None,None]
        self.assertAlmostEqual(float(denorm[0,110,0]),127/255,places=4)
        self.assertAlmostEqual(float(denorm[0,110,110]),1,places=4)
        with self.assertRaises(ValueError):prepare_image(Image.new('RGB',(3,200)))

if __name__=='__main__':unittest.main()
