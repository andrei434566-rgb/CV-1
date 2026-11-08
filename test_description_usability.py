"""Regressions from real Windows-dark-theme screenshots and local reference routing."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication, QScrollArea
from PySide6.QtCore import QSettings

from app.infrastructure.local_vision import reference_answer
from app.ui.dialogs.facies_dialog import FaciesDialog


class DescriptionUsabilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_basic_sandstone_definition_wins_over_dirty_document_excerpt(self):
        for text in ('что такое песчаник', 'Песчаник', 'Расскажи о песчанике', 'Что такое алевролит?'):
            with self.subTest(text=text), patch('app.infrastructure.local_knowledge_search.search_folders') as search:
                reply = reference_answer({'question':text,'folders':['ignored']})
                search.assert_not_called()
                self.assertIn('осадочная порода', reply['answer'])
                self.assertNotIn('http', reply['answer'])

    def test_sources_still_available_when_requested(self):
        reply = reference_answer({'question':'Что такое песчаник?','show_sources':True})
        self.assertIn('https://www.usgs.gov/', reply['answer'])

    def test_dark_system_palette_does_not_turn_card_children_dark_and_depths_are_lossless(self):
        old = self.app.palette(); dark = QPalette(old)
        dark.setColor(QPalette.ColorRole.Window, QColor('#202020'))
        dark.setColor(QPalette.ColorRole.WindowText, QColor('#efefef'))
        self.app.setPalette(dark)
        self.addCleanup(self.app.setPalette, old)
        with TemporaryDirectory() as folder:
            settings = QSettings(str(Path(folder)/'test.ini'), QSettings.Format.IniFormat)
            with patch('app.ui.dialogs.facies_dialog.QSettings', return_value=settings):
                card = FaciesDialog('Интервал YOLO', .4, depth_from=3002., depth_to=3002.97987305)
                self.addCleanup(card.close); card.show(); self.app.processEvents()
                panel = card.findChild(QScrollArea).widget()
                self.assertGreater(panel.palette().color(QPalette.ColorRole.Window).lightness(), 200)
                self.assertLess(panel.palette().color(QPalette.ColorRole.WindowText).lightness(), 130)
                self.assertEqual(card.depth_to.text(), '3002.98')
                self.assertEqual(card.selected_depth_range(), (3002.,3002.97987305))
                self.assertNotIn('97987305', card.vlm_description.toPlainText())
                card.depth_to.setText('3003,125')
                self.assertEqual(card.selected_depth_range(), (3002.,3003.125))

    def test_submillimetre_interval_does_not_appear_reversed_or_zero(self):
        from app.infrastructure.description_template import compose_description
        self.assertIn('3002–3002,0001', compose_description({}, 3002., 3002.0001).text)

    def test_new_facies_suggestion_requires_explicit_apply_and_stays_assumed(self):
        card = FaciesDialog('Интервал YOLO', .4); self.addCleanup(card.close)
        self.assertFalse(card.selected_facies())
        card._apply_suggestions({'Название фации':'Экспертный класс из Word'})
        self.assertEqual(card.selected_facies(), 'Экспертный класс из Word')
        self.assertTrue(card.facies_assumed.isChecked())
        self.assertFalse(card.training_confirmed.isChecked())

    def test_training_labels_and_multiple_regions_survive_reopening_and_moving_dataset(self):
        import json
        import shutil
        from PIL import Image
        from app.infrastructure.supervised_learning import digest
        from app.ui.dialogs.supervised_learning_dialog import SupervisedLearningDialog
        with TemporaryDirectory() as folder:
            root = Path(folder)/'before'; root.mkdir()
            image = root/'0001.png'; Image.new('RGB',(64,128),'gray').save(image)
            item = {'image':image.name,'conversations':[{'from':'gpt','value':'Порода: Песчаник'}],
                    'metadata':{'label_provenance':'expert_word_photo_number_reviewed','training_approved':True,
                                'photo_id':'0001','image_sha256':digest(image)}}
            manifest = root/'word_photo_approved.jsonl'; manifest.write_text(json.dumps(item,ensure_ascii=False),encoding='utf-8')
            dialog = SupervisedLearningDialog(manifest, root/'base')
            self.assertEqual(dialog.rows[0]['label'],'Песчаник')
            self.assertFalse(dialog.rows[0]['reviewed'])
            dialog.table.item(0,2).setText('Скважина-1')
            dialog._add_region()
            dialog.rows[1].update(roi=[0,0,.5,.5],reviewed=True)
            dialog._save(); dialog.close()
            moved = Path(folder)/'after'; shutil.copytree(root,moved)
            new = SupervisedLearningDialog(moved/manifest.name,moved/'base')
            self.addCleanup(new.close)
            self.assertEqual(len(new.rows),2)
            self.assertTrue(new.rows[1]['reviewed'])
            self.assertTrue(Path(new.rows[1]['image']).is_relative_to(moved.resolve()))
            new.table.item(1,1).setText('Алевролит')
            self.assertFalse(new.rows[1]['reviewed'])
            new.field.setCurrentText('Слоистость')
            self.assertEqual(new.rows[0]['well'],'Скважина-1')
            self.assertFalse(new.rows[0]['reviewed'])


if __name__ == '__main__': unittest.main()
