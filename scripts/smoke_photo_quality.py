"""Actual photo preflight plus offscreen queue/label overlays, private settings."""
import json
import os
from pathlib import Path
import sys
import tempfile
from time import perf_counter

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
os.environ['QT_QPA_PLATFORM']='offscreen'
from PySide6 import QtCore


def main():
    native=QtCore.QSettings
    settings=native(native.NativeFormat,native.UserScope,'Kern Analyzer','KernAnalyzer')
    before={key:settings.value(key) for key in settings.allKeys()}
    output=ROOT/'research_reports/photo_quality_20261104';output.mkdir(parents=True,exist_ok=True)
    previous=json.loads((ROOT/'models/vision/pipeline_worker_benchmark_20261104/report.json').read_text(encoding='utf-8'))
    sources=[Path(row['photo']) for row in previous['results']]
    with tempfile.TemporaryDirectory(prefix='quality_smoke_') as folder:
        class Settings(native):
            def __init__(self,*args,**kwargs):
                if args==('Kern Analyzer','KernAnalyzer'):super().__init__(str(Path(folder)/'settings.ini'),native.IniFormat)
                else:super().__init__(*args,**kwargs)
        QtCore.QSettings=Settings
        from PIL import Image,ImageOps
        from PySide6.QtCore import QPointF
        from PySide6.QtGui import QFont,QFontDatabase
        from PySide6.QtWidgets import QApplication
        from app.domain.models import FaciesDetection,PhotoRecord
        from app.infrastructure.image_loading import load_working_pixmap
        from app.infrastructure.local_vision import _boxes
        from app.infrastructure.photo_quality import audit_file
        from app.ui.dialogs.photo_quality_dialog import PhotoQualityDialog
        from app.ui.windows.main_window import MainWindow
        app=QApplication.instance() or QApplication([])
        font=Path('C:/Windows/Fonts/segoeui.ttf')
        if font.is_file():
            font_id=QFontDatabase.addApplicationFont(str(font));families=QFontDatabase.applicationFontFamilies(font_id)
            if families:app.setFont(QFont(families[0],10))
        window=MainWindow();reports=[];records=[]
        for index,path in enumerate(sources):
            started=perf_counter()
            pixmap=load_working_pixmap(path)
            with Image.open(path) as source:
                image=ImageOps.exif_transpose(source).convert('RGB');image.thumbnail((1400,1400))
            boxes=_boxes(image,None)
            columns=[dict(zip(('left','top','right','bottom'),[l*pixmap.width()/image.width,t*pixmap.height()/image.height,r*pixmap.width()/image.width,b*pixmap.height()/image.height])) for l,t,r,b in boxes]
            report=audit_file(path,columns,(pixmap.width(),pixmap.height()))
            record=PhotoRecord(f'qa-{index}',str(path),pixmap,well_name='Р-31',core_columns=columns,photo_quality=report)
            if columns:
                box=columns[0]
                record.detections=[FaciesDetection('Не определена',.4,[QPointF(box['left'],box['top']),QPointF(box['right'],box['top']),QPointF(box['right'],box['bottom']),QPointF(box['left'],box['bottom'])],
                    {'__geometry_only':'1','__vision_lithology_review_required':'1','Краткое описание':'Демонстрация очереди; определения породы не выполнялись.'})]
            records.append(record)
            reports.append({'photo':str(path),'columns':len(columns),'seconds_including_column_proposal':perf_counter()-started,'quality':report})
            if index==0:
                dialog=PhotoQualityDialog(pixmap,report,window);dialog.show();app.processEvents()
                assert dialog.grab().save(str(output/'label_exclusions.png'));dialog.close()
        window._records=records
        window.show_interval_review_queue();app.processEvents()
        assert window._review_queue_dialog.table.rowCount()>0
        assert window._review_queue_dialog.grab().save(str(output/'review_queue.png'))
        result={'photos':len(reports),'total_preflight_seconds':sum(row['seconds_including_column_proposal'] for row in reports),
                'label_candidates':sum(len(row['quality']['label_boxes']) for row in reports),
                'gui':'passed','settings_unchanged':before=={key:settings.value(key) for key in settings.allKeys()},
                'classification_accuracy_evaluated':False,'demonstration_geometry_not_training_data':True,
                'results':reports}
        assert result['settings_unchanged']
        (output/'report.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps({k:v for k,v in result.items() if k!='results'},ensure_ascii=False))
        window._review_queue_dialog.close();window.close();app.processEvents();app.quit()
        QtCore.QSettings=native


if __name__=='__main__':main()
