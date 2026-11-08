"""Data isolation, explicit supervision and head-training mechanics, not geological accuracy."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from app.infrastructure import supervised_learning as learning


class SupervisedLearningTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory(); self.addCleanup(self.temp.cleanup); self.root = Path(self.temp.name)

    def rows(self):
        rows = []; random = np.random.default_rng(42)
        for label in ('Класс А', 'Класс Б'):
            for well, count in (('Учебная', 5), ('Контрольная', 3)):
                for _ in range(count):
                    path = self.root / f'{len(rows)}.png'
                    Image.fromarray(random.integers(20, 220, (48, 64, 3), dtype=np.uint8)).save(path)
                    rows.append({'id': path.stem, 'image': str(path), 'image_sha256': learning.digest(path),
                                 'description': 'Проверенное описание', 'label': label, 'well': well,
                                 'roi': [0., 0., 1., 1.], 'reviewed': True})
        return rows

    def test_prose_does_not_become_a_label_and_assumptions_stay_unapproved(self):
        self.assertEqual(learning.suggested_label('Видны песчаник и алевролит.', 'Название породы'), '')
        self.assertEqual(learning.suggested_label('Порода: предположительно песчаник', 'Название породы'), '')
        self.assertEqual(learning.suggested_label('Порода: Песчаник\nПорода: Алевролит', 'Название породы'), '')
        self.assertEqual(learning.suggested_label('Фация: Плохо дренируемая пойма.', 'Название фации'), 'Плохо дренируемая пойма')

    def test_only_explicitly_reviewed_regions_are_used(self):
        rows = self.rows(); extra = dict(rows[0], label='', roi=None, reviewed=False); rows.append(extra)
        approved, classes, counts = learning.validate_rows(rows, 'Название фации', 'Контрольная')
        self.assertEqual(len(approved), 16); self.assertEqual(len(classes), 2)
        self.assertEqual(counts['val'], {'Класс А': 3, 'Класс Б': 3})
        self.assertFalse({r['well'] for r in approved if r['split']=='train'} & {r['well'] for r in approved if r['split']=='val'})

    def test_contacts_and_cementation_need_exact_classes_and_field_specific_review(self):
        from app.domain.lithology_attributes import LITHOLOGY_ATTRIBUTE_OPTIONS
        self.assertEqual(learning.suggested_label('Контакт: отчётливый.', 'Контакт'), 'Отчетливый')
        self.assertEqual(learning.suggested_label('Контакт: 2 мм', 'Контакт'), '')
        self.assertEqual(learning.suggested_label('Степень цементации: возможно крепкий', 'Степень цементации'), '')
        for field, basis in learning.REVIEW_BASIS.items():
            with self.subTest(field=field):
                labels = LITHOLOGY_ATTRIBUTE_OPTIONS[field]
                rows = [dict(id=str(i), image_sha256=str(i), reviewed=True, label=labels[i//8],
                             roi=[0., 0., 1., 1.], well='Контрольная' if i%8 >= 5 else 'Учебная')
                        for i in range(24)]
                with self.assertRaisesRegex(ValueError, 'подтвердите основание'):
                    learning.validate_rows(rows, field, 'Контрольная')
                for row in rows:
                    row['review_basis'] = basis
                with self.assertRaisesRegex(ValueError, labels[-1]):
                    learning.validate_rows(rows[:-8], field, 'Контрольная')
                _, classes, counts = learning.validate_rows(rows, field, 'Контрольная')
                self.assertCountEqual(classes, labels)
                self.assertEqual(counts['val'], dict.fromkeys(labels, 3))

    def test_whole_region_model_cannot_be_used_with_an_interval_embedding(self):
        import torch
        folder = self.root/'contact'; folder.mkdir()
        weights = folder/'head.pt'; weights.write_bytes(b'example')
        # Even an incorrectly edited deployment flag must not activate a contact head.
        learning.write_json(folder/'config.json', {'schema': 'cv1-feature-head-v1', 'generative': False,
            'field': 'Контакт', 'deployment': 'suggestions_only', 'head_sha256': learning.digest(weights)})
        with self.assertRaisesRegex(ValueError, 'целого фрагмента'):
            learning.activate_head(folder, self.root)
        learning.write_json(self.root/'active.json', {'Контакт': 'contact'})
        self.assertEqual(learning.head_suggestions(torch.ones(2), 'a'*64, self.root), ({}, []))

    def test_special_argillite_variants_share_one_class_but_keep_separate_properties(self):
        from app.domain.rock_types import SPECIAL_ARGILLITE_NAME
        rows=self.rows()
        for i,row in enumerate(rows[:8]):row['label']='Битуминозный аргиллит' if i%2==0 else 'Окремнённый аргиллит'
        approved,classes,counts=learning.validate_rows(rows,'Название породы','Контрольная')
        self.assertEqual(set(classes),{SPECIAL_ARGILLITE_NAME,'Класс Б'})
        self.assertEqual(counts['train'][SPECIAL_ARGILLITE_NAME],5);self.assertEqual(counts['val'][SPECIAL_ARGILLITE_NAME],3)
        self.assertIs(approved[0]['rock_properties']['bituminous'],True)
        self.assertIsNone(approved[0]['rock_properties']['silicified'])
        self.assertIs(approved[1]['rock_properties']['silicified'],True)
        self.assertIsNone(approved[1]['rock_properties']['bituminous'])
        exported=learning.export_dataset(rows,'Название породы','Контрольная',self.root/'or_class')
        sample=json.loads(exported.read_text(encoding='utf8'))['samples'][1]
        self.assertEqual(sample['rock_source_labels'],['Окремнённый аргиллит']);self.assertTrue(sample['rock_properties']['silicified'])
        repeated,_,_=learning.validate_rows(approved,'Название породы','Контрольная')
        self.assertIs(repeated[1]['rock_properties']['silicified'],True)
        self.assertIsNone(repeated[1]['rock_properties']['bituminous'])

    def test_same_argillite_region_can_confirm_both_properties_without_double_counting(self):
        rows=self.rows()
        for row in rows[:8]:row['label']='Битуминозный аргиллит'
        rows.append(dict(rows[0],label='Окремнённый аргиллит'))
        approved,_,_=learning.validate_rows(rows,'Название породы','Контрольная')
        self.assertEqual(len(approved),16)
        self.assertTrue(approved[0]['rock_properties']['bituminous']);self.assertTrue(approved[0]['rock_properties']['silicified'])

    def test_missing_roi_and_single_well_and_single_class_fail_with_actionable_error(self):
        for change in ('roi', 'well', 'label'):
            with self.subTest(change=change):
                rows = self.rows()
                for row in rows:
                    row[change] = None if change == 'roi' else ('Контрольная' if change == 'well' else 'Один класс')
                with self.assertRaises(ValueError): learning.validate_rows(rows, 'Слоистость', 'Контрольная')

    def test_source_copies_cannot_cross_validation_well(self):
        rows = self.rows(); rows[5]['image_sha256'] = rows[0]['image_sha256']
        with self.assertRaisesRegex(ValueError, 'Одинаковое фото'):
            learning.validate_rows(rows, 'Слоистость', 'Контрольная')

    def test_same_region_conflicting_labels_fail(self):
        rows = self.rows(); rows.append(dict(rows[0], label='Другой'))
        with self.assertRaisesRegex(ValueError, 'противоречивые'):
            learning.validate_rows(rows, 'Слоистость', 'Контрольная')

    def test_changed_original_cannot_be_exported_as_confirmed(self):
        rows = self.rows(); Image.new('RGB', (64,48), 'red').save(rows[0]['image'])
        with self.assertRaisesRegex(ValueError, 'изменилось'):
            learning.export_dataset(rows, 'Слоистость', 'Контрольная', self.root/'dataset')

    def test_portable_export_keeps_description_and_rejects_pixel_duplicates(self):
        rows = self.rows()
        manifest = learning.export_dataset(rows, 'Слоистость', 'Контрольная', self.root/'dataset')
        payload = json.loads(manifest.read_text(encoding='utf-8'))
        self.assertFalse(Path(payload['samples'][0]['image']).is_absolute())
        self.assertEqual(payload['samples'][0]['description'], rows[0]['description'])
        Image.open(rows[0]['image']).save(rows[5]['image'])
        rows[5]['image_sha256'] = learning.digest(rows[5]['image'])
        # Different source-container hashes cannot hide identical training pixels.
        from PIL.PngImagePlugin import PngInfo
        meta = PngInfo(); meta.add_text('copy', 'yes')
        Image.open(rows[0]['image']).save(rows[5]['image'], pnginfo=meta)
        rows[5]['image_sha256'] = learning.digest(rows[5]['image'])
        with self.assertRaisesRegex(ValueError, 'одинаковые пиксели'):
            learning.export_dataset(rows, 'Слоистость', 'Контрольная', self.root/'duplicate')

    def test_metrics_do_not_substitute_missing_class_scores(self):
        result = learning.classification_metrics([0,0,1], [0,0,0], ['А', 'Б'])
        self.assertEqual(result['per_class'][1]['recall'], 0)
        self.assertLess(result['macro_f1'], .5)

    def test_head_training_and_activation_do_not_modify_base_and_require_matching_sha(self):
        import torch
        rows = self.rows()
        dataset = learning.export_dataset(rows, 'Название фации', 'Контрольная', self.root/'dataset')
        marker = self.root/'base_marker'; marker.write_text('unchanged')
        class Engine:
            def __init__(self): self.model = object(); self.device = 'cpu'; self.card = {'architecture':'convnext_tiny','model_sha256':'a'*64}
            def _load(self, folder): pass
        # Synthetic features verify optimization and serialization only.
        vectors = [torch.tensor([1.,0.]) if row['label']=='Класс А' else torch.tensor([0.,1.]) for row in rows]
        with patch('app.infrastructure.local_vision.VisionEngine', Engine), patch.object(learning, 'extract_embedding', side_effect=vectors):
            result = learning.train_feature_head(dataset, self.root, self.root/'models'/'one', steps=180)
        self.assertEqual(marker.read_text(), 'unchanged')
        self.assertFalse(result['card']['backbone_modified'])
        self.assertFalse((self.root/'models'/'active.json').exists())
        learning.activate_head(result['folder'], self.root/'models')
        suggestions, details = learning.head_suggestions(torch.tensor([1.,0.]), 'a'*64, self.root/'models')
        self.assertEqual(suggestions['Название фации'], 'Класс А')
        self.assertFalse(details[0]['confirmed'])
        self.assertEqual(learning.head_suggestions(torch.tensor([1.,0.]), 'b'*64, self.root/'models'), ({}, []))
        learning.deactivate_head('Название фации', self.root/'models')
        self.assertTrue((self.root/'models'/'one'/'head.pt').is_file())
        self.assertEqual(learning.head_suggestions(torch.tensor([1.,0.]), 'a'*64, self.root/'models'), ({}, []))
        learning.activate_head(result['folder'], self.root/'models')
        (self.root/'models'/'one'/'head.pt').write_bytes(b'changed')
        self.assertEqual(learning.head_suggestions(torch.tensor([1.,0.]), 'a'*64, self.root/'models'), ({}, []))

    def test_cancel_does_not_publish_weights(self):
        dataset = learning.export_dataset(self.rows(), 'Слоистость', 'Контрольная', self.root/'dataset')
        with patch('app.infrastructure.local_vision.VisionEngine'):
            with self.assertRaises(InterruptedError):
                learning.train_feature_head(dataset, self.root, self.root/'cancelled', cancelled=lambda: True)
        self.assertFalse((self.root/'cancelled').exists())


if __name__ == '__main__': unittest.main()
