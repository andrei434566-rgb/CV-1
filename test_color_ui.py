"""Photographic colour auto-fill and image/score presentation without a generative model."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from PIL import Image
from PySide6.QtGui import QColor,QPixmap
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication,QLabel,QFrame
from app.infrastructure.core_color_hint import estimate_core_color
from app.ui.dialogs.facies_dialog import FaciesDialog
from app.ui.widgets.lithology_predictions import LithologyPredictionGallery

APP=QApplication.instance() or QApplication([])


class ColorUiTests(unittest.TestCase):
    def card(self,attrs=None):
        temp=TemporaryDirectory();self.addCleanup(temp.cleanup)
        settings=QSettings(str(Path(temp.name)/'settings.ini'),QSettings.IniFormat)
        pixmap=QPixmap(100,240);pixmap.fill(QColor(145,145,145))
        with patch('app.ui.dialogs.facies_dialog.QSettings',return_value=settings):
            card=FaciesDialog('Интервал YOLO',.8,attrs or {},preview_pixmap=pixmap)
        self.addCleanup(card.close)
        return card

    def test_gallery_shows_rock_thumbnails_and_scores_without_background(self):
        for background in ("garbage", "background"):
            with self.subTest(background=background):
                rows = [
                    {"class": code, "score": score}
                    for code, score in zip((background, "shale", "limestone", "sandstone"), (.73, .22, .03, .02))
                ]
                gallery = LithologyPredictionGallery(rows)
                self.addCleanup(gallery.close)
                cards = gallery.findChildren(QFrame, "lithologyPredictionCard")
                self.assertEqual([card.property("classCode") for card in cards], ["shale", "limestone", "sandstone"])
                for card, score in zip(cards, ("22%", "3%", "2%")):
                    self.assertFalse(card.findChild(QLabel, "lithologyPredictionImage").pixmap().isNull())
                    self.assertEqual(card.findChild(QLabel, "lithologyPredictionScore").text(), score)

    def test_invalid_or_duplicate_probability_rows_do_not_create_fake_scores(self):
        gallery=LithologyPredictionGallery([{'class':'shale','score':float('nan')},{'class':'unknown','score':.9},
                                           {'class':'shale','score':.7},{'class':'shale','score':.2}])
        self.addCleanup(gallery.close)
        self.assertEqual(len(gallery.findChildren(QFrame,'lithologyPredictionCard')),1)

    def test_opening_empty_card_requests_colour_and_fills_template_without_loading_cnn(self):
        card=self.card();calls=[]
        with patch('app.infrastructure.attribute_tasks.submit',side_effect=lambda op,payload,cb:calls.append((op,payload,cb))):
            card.show();APP.processEvents()
        self.assertEqual(calls[0][0],'color')
        result=estimate_core_color(Image.new('RGB',(100,240),(145,145,145))).to_dict()
        calls[0][2]({'ok':True,'result':{'signature':calls[0][1]['signature'],'measurement':result}})
        self.assertEqual(card.attribute_combos['Цвет'].currentData(),'серый')
        self.assertIn('серый',card.vlm_description.toPlainText())
        attrs=card.selected_attributes()
        self.assertEqual(json.loads(attrs['__color_measurement'])['rgb'],[145,145,145])
        self.assertEqual(attrs['__color_expert_confirmed'],'0')
        card.assumption_checks['Цвет'].setChecked(False)
        self.assertEqual(card.selected_attributes()['__color_expert_confirmed'],'1')

    def test_colour_changed_while_job_runs_is_not_overwritten(self):
        card=self.card();calls=[]
        with patch('app.infrastructure.attribute_tasks.submit',side_effect=lambda op,payload,cb:calls.append((payload,cb))):
            card.show();APP.processEvents()
        card.attribute_combos['Цвет'].setCurrentIndex(card.attribute_combos['Цвет'].findData('Коричневый'))
        result=estimate_core_color(Image.new('RGB',(100,240),(145,145,145))).to_dict()
        calls[0][1]({'ok':True,'result':{'signature':calls[0][0]['signature'],'measurement':result}})
        self.assertEqual(card.attribute_combos['Цвет'].currentData(),'Коричневый')

    def test_confirmed_colour_is_kept_and_does_not_start_background_remeasurement(self):
        card=self.card({'Цвет':'Серый','__vision_color':'Серый','__color_expert_confirmed':'1'})
        with patch('app.infrastructure.attribute_tasks.submit') as submit:
            card.show();APP.processEvents();submit.assert_not_called()


if __name__=='__main__':unittest.main()
