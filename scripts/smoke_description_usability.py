"""Render isolated cards under a dark system palette; all labels/data are temporary QA fixtures."""
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
from PySide6.QtCore import QSettings
from PySide6.QtGui import QColor, QFont, QFontDatabase, QPalette, QPixmap
from PySide6.QtWidgets import QApplication
from PIL import Image
from app.infrastructure.supervised_learning import digest
from app.ui.dialogs.facies_dialog import FaciesDialog
from app.ui.dialogs.supervised_learning_dialog import SupervisedLearningDialog


def main():
    app = QApplication([])
    font_id = QFontDatabase.addApplicationFont('C:/Windows/Fonts/segoeui.ttf')
    families = QFontDatabase.applicationFontFamilies(font_id)
    if families: app.setFont(QFont(families[0], 10))
    dark = app.palette(); dark.setColor(QPalette.ColorRole.Window, QColor('#202020'))
    dark.setColor(QPalette.ColorRole.Base, QColor('#282828')); dark.setColor(QPalette.ColorRole.WindowText, QColor('white'))
    app.setPalette(dark)
    output = ROOT/'research_reports/description_usability_20261104'; output.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix='cv1_usability_') as folder:
        folder = Path(folder)
        photo = QPixmap(str(ROOT/'models/vision/pipeline_worker_benchmark_20261104/tape_01.jpg'))
        if not photo.isNull(): photo = photo.copy(0, 0, photo.width(), min(photo.height(), photo.width()*4))
        settings = QSettings(str(folder/'private.ini'), QSettings.Format.IniFormat)
        with patch('app.ui.dialogs.facies_dialog.QSettings', return_value=settings):
            card = FaciesDialog('Интервал YOLO', .4, {'__geometry_only':'1', '__description_engine':'vision',
                               '__vision_visual_hint':'На фото заметны контрастные полосы; характер слоистости требует проверки.',
                               'Цвет':'тёмно-серый'}, 3002., 3002.97987305, preview_pixmap=photo)
            card.show(); app.processEvents(); card.grab().save(str(output/'card_dark_palette.png'))
            card.close()
        rows = []
        for i, color in enumerate(('gray', 'silver')):
            image = folder/f'{i+1:04}.png'; Image.new('RGB', (100, 300), color).save(image)
            rows.append({'image':image.name,'conversations':[{'from':'gpt','value':f'Фото {i+1:04}\nДемонстрация интерфейса, не учебные данные.\nПорода: Пример {i+1}'}],
                         'metadata':{'label_provenance':'expert_word_photo_number_reviewed','training_approved':True,
                                     'photo_id':str(i+1), 'image_sha256':digest(image)}})
        manifest = folder/'word_photo_approved.jsonl'
        manifest.write_text('\n'.join(json.dumps(row,ensure_ascii=False) for row in rows), encoding='utf-8')
        trainer = SupervisedLearningDialog(manifest, folder/'unavailable')
        trainer.setWindowTitle('Обучение на ваших фото · демонстрация интерфейса')
        trainer.show(); app.processEvents(); trainer.grab().save(str(output/'training.png')); trainer.close()
    print(json.dumps({'dark_palette':'rendered','real_training_run':False,'output':str(output)},ensure_ascii=False))


if __name__ == '__main__': main()
