"""Demonstration-only report summary with private settings and temporary photos."""
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));os.environ['QT_QPA_PLATFORM']='offscreen'
from PySide6 import QtCore


def main():
    native=QtCore.QSettings
    original=native(native.NativeFormat,native.UserScope,'Kern Analyzer','KernAnalyzer')
    before={key:original.value(key) for key in original.allKeys()}
    output=ROOT/'research_reports/report_readiness_20261104';output.mkdir(parents=True,exist_ok=True)
    with TemporaryDirectory(prefix='cv1_readiness_demo_') as folder:
        folder=Path(folder)
        class Settings(native):
            def __init__(self,*args,**kwargs):
                if args==('Kern Analyzer','KernAnalyzer'):super().__init__(str(folder/'private.ini'),native.IniFormat)
                else:super().__init__(*args,**kwargs)
        QtCore.QSettings=Settings
        from PySide6.QtCore import QPointF
        from PySide6.QtGui import QFont,QFontDatabase,QPixmap
        from PySide6.QtWidgets import QApplication
        from app.domain.models import FaciesDetection,PhotoRecord
        from app.infrastructure.expert_corrections import expert_memory_digest
        from app.infrastructure.core_report_export import export_core_description_report
        from app.ui.windows.main_window import MainWindow
        app=QApplication.instance() or QApplication([])
        font=Path('C:/Windows/Fonts/segoeui.ttf')
        if font.is_file():
            index=QFontDatabase.addApplicationFont(str(font));families=QFontDatabase.applicationFontFamilies(index)
            if families:app.setFont(QFont(families[0],10))
        window=MainWindow();records=[]
        rows=[('Демо А',100,101),('Демо А',102,103),('Демо А',102.5,104),('Демо Б',200,201),('Демо Б',201,202)]
        for index,(well,top,base) in enumerate(rows):
            photo=QPixmap(120,240);photo.fill();path=folder/f'пример_{index+1}.png';photo.save(str(path))
            attrs={'__geometry_only':'1','Название породы':'Песчаник','Цвет':'Серый','Зернистость':'Мелкозернистый',
                   'Слоистость':'Волнистая','Контакт':'Резкий','Краткое описание':'Демонстрационный текст для проверки интерфейса.',
                   '__vlm_description':'Демонстрационный текст для проверки интерфейса.'}
            if index==1:attrs.pop('Контакт')
            if index==2:attrs['__description_assumptions']=json.dumps(['Зернистость'],ensure_ascii=False)
            detection=FaciesDetection('Демо-фация',.8,[QPointF(0,0),QPointF(100,0),QPointF(100,220)],attrs,top,base)
            if index in (0,3):
                attrs['__expert_memory_confirmed']='1';attrs['__expert_memory_sha256']=expert_memory_digest(detection)
            records.append(PhotoRecord(f'demo-{index}',str(path),photo,[detection],well_name=well,analysis_mode='yolo_vlm'))
        window._records=records;window._well_intervals={'Демо А':(100,105),'Демо Б':(200,202)}
        window.show_project_readiness();app.processEvents();dialog=window._readiness_dialog
        assert dialog.grab().save(str(output/'summary.png'))
        dialog.tabs.setCurrentIndex(1);app.processEvents();assert dialog.grab().save(str(output/'issues.png'))
        audit,_,_=window._project_readiness_snapshot()
        assert audit.totals['rows']==5 and audit.totals['gaps']==2 and audit.totals['overlaps']==1
        wells=window._core_description_report_wells(refresh_view=False)
        export_core_description_report(output/'DEMO_report.xlsx','Демонстрация интерфейса',wells,'xlsx')
        result={'demonstration_only':True,'training_examples_written':False,'gui':'passed',
                'native_settings_unchanged':before=={key:original.value(key) for key in original.allKeys()},
                'audit':audit.to_dict()}
        assert result['native_settings_unchanged']
        (output/'demo.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps({key:value for key,value in result.items() if key!='audit'},ensure_ascii=False))
        dialog.close();window.close();app.processEvents();app.quit();QtCore.QSettings=native


if __name__=='__main__':main()
