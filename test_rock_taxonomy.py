"""Rock vocabulary, abstention and legacy compatibility at the inference boundary."""
import json,tempfile,unittest
from pathlib import Path
from app.domain.rock_types import ROCK_CLASSES,COAL_CLASSES,ROCK_PROFILES,rock_profile,training_status,rock_reference_answer
from app.domain.lithology_attributes import LITHOLOGY_ATTRIBUTE_OPTIONS
from app.infrastructure.rock_predictions import summarize_predictions
from app.infrastructure.local_vision import read_model_card


def card(classes=ROCK_CLASSES):
    return {'schema':'cv1-discriminative-vision-v2','generative':False,'classes':list(classes),
            'architecture':'efficientnet_b0','background_class':'background','review_threshold':.85,'margin_threshold':.2}


class RockTaxonomyTests(unittest.TestCase):
    def confident(self,code,classes=ROCK_CLASSES):
        p=[.01/(len(classes)-1)]*len(classes);p[classes.index(code)]=.99
        return summarize_predictions([p],card(classes))

    def test_all_ten_user_names_and_argillite_aliases_resolve(self):
        self.assertEqual(len(ROCK_PROFILES),10)
        for name in LITHOLOGY_ATTRIBUTE_OPTIONS['Название породы']:self.assertIsNotNone(rock_profile(name))
        self.assertEqual(rock_profile('Углистый аргиллит')['name'],'Углистый аргилит')

    def test_union_category_does_not_require_or_invent_both_properties(self):
        from app.domain.rock_types import rock_training_label,SPECIAL_ARGILLITE_NAME
        name,properties=rock_training_label(SPECIAL_ARGILLITE_NAME)
        self.assertEqual(name,SPECIAL_ARGILLITE_NAME)
        self.assertIsNone(properties['bituminous']);self.assertIsNone(properties['silicified'])
        for label in ('Битуминозный известняк','Окремнённый песчаник','Аргиллит с прослоем битуминозного известняка','Неокремнённый аргиллит'):
            self.assertEqual(rock_training_label(label),(label,None))

    def test_native_class_is_not_relabelled_to_another_rock(self):
        r=self.confident('carbonaceous_siltstone')
        self.assertIn('Углистый алевролит',r['lithology']);self.assertTrue(r['low'])
        r=self.confident('breccia_other');self.assertIn('Брекчия',r['lithology']);self.assertNotEqual(r['lithology'],'Предположительно: Гравелит')
        r=self.confident('carbonate_other');self.assertNotIn('Аргиллит',r['lithology'])

    def test_crystalline_group_does_not_assert_basement_position(self):
        r=self.confident('crystalline');self.assertTrue(r['low'])
        self.assertNotEqual(r['lithology'],'Предположительно: Породы фундамента');self.assertIn('Положение фундамента',r['detail'])

    def test_coal_only_appears_when_coal_class_really_exists(self):
        r=self.confident('coal',COAL_CLASSES);self.assertEqual(r['lithology'],'Литология требует проверки')
        self.assertIn('автоматически не назначается',r['detail']);self.assertTrue(r['low'])
        p=rock_profile('Уголь');self.assertIn('нет',training_status(p,card()))
        self.assertIn('отключено',training_status(p,card(COAL_CLASSES)))

    def test_missing_target_classes_do_not_get_zero_percent_fake_probabilities(self):
        r=self.confident('sandstone')
        self.assertEqual({p['class'] for p in r['predictions']},set(ROCK_CLASSES))
        self.assertNotIn('coal',[p['class'] for p in r['predictions']])
        for p in ROCK_PROFILES:
            if p['native'] is None:self.assertIn('нет',training_status(p,card()))

    def test_disagreement_does_not_manufacture_interbedding(self):
        a=[.001]*8;b=[.001]*8;a[0]=.993;b[2]=.993
        r=summarize_predictions([a,b],card())
        self.assertTrue(r['low']);self.assertNotIn('Переслаивание',r['lithology'])

    def test_background_majority_cannot_be_discarded_to_create_rock_confidence(self):
        bg=[.001]*8;bg[7]=.993;sand=[.001]*8;sand[0]=.993
        r=summarize_predictions([bg,bg,bg,sand],card())
        self.assertEqual(r['lithology'],'Литология не установлена');self.assertTrue(r['low'])

    def test_absent_or_invalid_pixels_do_not_create_lithology(self):
        self.assertEqual(summarize_predictions([],card())['lithology'],'Литология не установлена')
        self.assertEqual(summarize_predictions([],card())['predictions'],[])
        for p in [[float('nan')]*8,[-.1]+[.2]*7,[.1]*8,[1.,0.]]:
            with self.assertRaises(ValueError):summarize_predictions([p],card())

    def test_old_four_class_model_still_has_its_original_names(self):
        c={'classes':['sandstone','limestone','shale','garbage']}
        r=summarize_predictions([[.01,.97,.01,.01]],c)
        self.assertEqual(r['lithology'],'Предположительно: Известняк');self.assertEqual(len(r['predictions']),4)

    def test_specific_organic_rock_definition_is_not_replaced_by_generic_argillite(self):
        answer=rock_reference_answer('Что такое углистый аргиллит?')
        self.assertTrue(answer.startswith('Углистый аргилит\n'));self.assertEqual(answer.count('Глинистая матрица'),1)
        self.assertNotIn('https:',answer)
        answer=rock_reference_answer('Чем отличаются песчаник и алевролит?',True)
        self.assertIn('Песчаник\n',answer);self.assertIn('Алевролит\n',answer);self.assertIn('https:',answer)
        self.assertEqual(rock_reference_answer('Опиши фото песчаника'), '')

    def test_only_declared_ordered_vocabulary_is_accepted(self):
        with tempfile.TemporaryDirectory() as d:
            folder=Path(d);(folder/'model.pt').write_bytes(b'test fixture')
            for classes in (ROCK_CLASSES,COAL_CLASSES):
                (folder/'config.json').write_text(json.dumps(card(classes)))
                self.assertEqual(len(read_model_card(folder)['classes']),len(classes))
            invalid=card();invalid['classes'][4]='carbonaceous_argillite'
            (folder/'config.json').write_text(json.dumps(invalid))
            with self.assertRaises(ValueError):read_model_card(folder)

    def test_runtime_loads_nine_outputs_and_does_not_treat_index_three_as_background(self):
        import hashlib,torch
        from torch import nn
        from PIL import Image
        from unittest.mock import patch
        from app.infrastructure.local_vision import VisionEngine
        class Tiny(nn.Module):
            def __init__(self,classes):
                super().__init__();self.classifier=nn.Sequential(nn.Dropout(.0),nn.Linear(3,classes))
            def forward(self,x):return self.classifier(x.mean(dim=(2,3)))
        with tempfile.TemporaryDirectory() as d:
            folder=Path(d);model=Tiny(9)
            with torch.no_grad():model.classifier[1].weight.zero_();model.classifier[1].bias.zero_();model.classifier[1].bias[3]=10.
            torch.save(model.state_dict(),folder/'model.pt')
            c=card(COAL_CLASSES);c['model_sha256']=hashlib.sha256((folder/'model.pt').read_bytes()).hexdigest()
            (folder/'config.json').write_text(json.dumps(c))
            Image.new('RGB',(100,240),(140,140,140)).save(folder/'core.png')
            with patch('app.infrastructure.vision_models.create_network',side_effect=lambda arch,classes:Tiny(classes)):
                result=VisionEngine(device='cpu').describe({'base':str(folder),'image':str(folder/'core.png'),'operation':'describe'})
            self.assertEqual(len(result['predictions']),9)
            self.assertNotEqual(result['fields']['lithology'],'Литология не установлена')
            self.assertIn('метаморфические',result['fields']['lithology'])
            self.assertEqual(result['predictions'][0]['class'],'crystalline')
            self.assertEqual(result['model_info']['classes'],list(COAL_CLASSES))
            self.assertIn('серый',result['observations']['color'])

    def test_optional_model_switch_is_explicit_and_rejects_paths_outside_model_folder(self):
        from unittest.mock import patch
        from PySide6.QtWidgets import QApplication
        from app.ui.dialogs.local_vlm_settings_dialog import LocalVlmSettingsDialog
        app=QApplication.instance() or QApplication([])
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);parent=root/'models/vision';folder=parent/'extended';folder.mkdir(parents=True)
            c=card();c.update(experimental=True,noncommercial_only=True)
            (folder/'config.json').write_text(json.dumps(c));(folder/'model.pt').write_bytes(b'fixture')
            (parent/'candidates.json').write_text(json.dumps({'models':[{'folder':'extended','name':'Расширенный'},
                                                                       {'folder':'../../outside','name':'Bad path'}]}))
            with patch('app.runtime_paths.user_data_root',return_value=root),patch('app.runtime_paths.bundled_root',return_value=root):
                dialog=LocalVlmSettingsDialog({'base':'untouched'})
            try:
                self.assertEqual(dialog.base.text(),'untouched')
                self.assertEqual(dialog.model_choices.count(),2)
                dialog.model_choices.activated.emit(1)
                self.assertEqual(Path(dialog.base.text()),folder.resolve())
                self.assertIn('некоммерческие',dialog.scope.text())
            finally:dialog.close()


if __name__=='__main__':unittest.main()
