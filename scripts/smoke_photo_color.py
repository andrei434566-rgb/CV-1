"""Real masked-core colour checks and private offscreen UI preview; no training or model publication."""
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import time
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
os.environ['QT_QPA_PLATFORM']='offscreen'


def main():
    from PIL import Image
    from PySide6.QtCore import QSettings,QPoint
    from PySide6.QtGui import QFontDatabase,QFont,QPixmap
    from PySide6.QtWidgets import QApplication
    from app.infrastructure.core_color_hint import estimate_core_color
    from app.infrastructure.photo_quality import label_boxes
    from app.infrastructure.attribute_tasks import wait_for_tasks
    from app.ui.dialogs.facies_dialog import FaciesDialog
    from app.ui.widgets.workspace_canvas import FaciesConfidencePopup
    from app.domain.models import FaciesDetection
    app=QApplication.instance() or QApplication([])
    font=QFontDatabase.addApplicationFont('C:/Windows/Fonts/segoeui.ttf')
    families=QFontDatabase.applicationFontFamilies(font)
    if families:app.setFont(QFont(families[0],10))
    output=ROOT/'research_reports/color_20261105';output.mkdir(parents=True,exist_ok=True)
    photos=sorted((ROOT/'models/vision/pipeline_worker_benchmark_20261104').glob('tape*.jpg'))[:3]
    if not photos:raise ValueError('No existing core crops on C for the smoke check')
    rows=[]
    for photo in photos:
        started=time.perf_counter()
        with Image.open(photo) as image:
            boxes=label_boxes(image)
            hint=estimate_core_color(image,exclude_boxes=boxes)
        rows.append({'image':photo.name,'measurement':hint.to_dict() if hint else None,
                     'seconds':round(time.perf_counter()-started,4)})
    with TemporaryDirectory(prefix='cv1_color_qa_') as folder:
        settings=QSettings(str(Path(folder)/'private.ini'),QSettings.IniFormat)
        with patch('app.ui.dialogs.facies_dialog.QSettings',return_value=settings):
            card=FaciesDialog('Интервал YOLO',.8,{},3002.,3003.,preview_pixmap=QPixmap(str(photos[0])))
            card.show();app.processEvents();wait_for_tasks();app.processEvents()
            from PySide6.QtWidgets import QScrollArea
            scroll=card.findChild(QScrollArea);scroll.ensureWidgetVisible(card.color_button);app.processEvents()
            card.grab().save(str(output/'color_card.png'))
            attrs=card.selected_attributes();card.close()
        fixture=[{'class':name,'score':score} for name,score in zip(('garbage','shale','limestone','sandstone'),(.73,.22,.03,.02))]
        detection=FaciesDetection('Интервал YOLO',.8,[],attributes={'__description_engine':'vision',
            '__vision_predictions':json.dumps(fixture),'Цвет':attrs.get('Цвет',''),'__vlm_description':'Демонстрация интерфейса. Проценты в этом снимке — пример оформления.'})
        popup=FaciesConfidencePopup();popup.show_for(QPoint(20,20),detection,{})
        app.processEvents();popup.grab().save(str(output/'predictions_gallery.png'));popup.hide()
    report={'real_core_crops':rows,'model_weights_changed':False,'geological_accuracy_evaluated':False,
            'gallery_percentages_are_ui_fixture':True,'card_colour':attrs.get('Цвет')}
    (output/'color_smoke.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False))


if __name__=='__main__':main()
