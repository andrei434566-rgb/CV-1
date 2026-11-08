"""Offscreen GUI check with private settings; never edits existing projects."""
from pathlib import Path
import json,os,sys,tempfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));os.environ['QT_QPA_PLATFORM']='offscreen'
from PySide6 import QtCore
NativeSettings=QtCore.QSettings
original=NativeSettings(NativeSettings.NativeFormat,NativeSettings.UserScope,'Kern Analyzer','KernAnalyzer')
before={k:original.value(k) for k in original.allKeys()}

def main():
    output=ROOT/'research_reports/vision_20261104';output.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='vision_gui_',dir=ROOT/'models/vision') as folder:
        ini=str(Path(folder)/'private.ini')
        class Settings(NativeSettings):
            def __init__(self,*args,**kwargs):
                if len(args)==2 and args==('Kern Analyzer','KernAnalyzer'):
                    super().__init__(ini,NativeSettings.IniFormat)
                else:super().__init__(*args,**kwargs)
        QtCore.QSettings=Settings
        from PySide6.QtWidgets import QApplication,QLabel
        from PySide6.QtCore import QPoint
        from app.ui.windows.main_window import MainWindow
        from app.ui.dialogs.local_vlm_settings_dialog import LocalVlmSettingsDialog
        from app.ui.dialogs.training_center_dialog import TrainingCenterDialog
        from app.ui.widgets.workspace_canvas import FaciesConfidencePopup
        from app.domain.models import FaciesDetection
        app=QApplication.instance() or QApplication([])
        window=MainWindow();config=window._local_vlm_config()
        assert config['model_mode']=='vision' and config['adapter']==''
        settings=LocalVlmSettingsDialog(config,window)
        assert settings.values()['model_mode']=='vision'
        training=TrainingCenterDialog(window)
        assert training.train_pairs.isHidden() and training.train_theory.isHidden()
        assert training.word_pairs.train_button.isHidden()
        detection=FaciesDetection('Не распознана',.84,[],{
            '__description_engine':'vision','Название породы':'Литология требует проверки',
            '__vision_predictions':json.dumps([{'name':'Известняк','score':.45},{'name':'Глинистая порода','score':.4}]),
            'Краткое описание':'Черновик: нужна проверка геологом.'})
        popup=FaciesConfidencePopup();popup.show_for(QPoint(30,30),detection,{})
        labels=[label.text() for label in popup.findChildren(QLabel)]
        assert any('Известняк: 45%' in s for s in labels),labels
        assert any('геометрии YOLO: 84%' in s for s in labels),labels
        assert not any(s=='Варианты модели' for s in labels),labels
        popup.hide();training.close();settings.close();window.close();app.processEvents();app.quit()
        assert {k:original.value(k) for k in original.allKeys()}==before,'Main settings were modified'
        result={'gui':'passed','engine':config['model_mode'],'active_model':config['base'],
                'generative_training_controls_hidden':True,'geometry_and_lithology_scores_separate':True,
                'existing_settings_unchanged':True,'existing_projects_edited':False}
        (output/'gui_smoke.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps(result,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
