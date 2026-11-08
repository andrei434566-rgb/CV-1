"""Private offscreen UI QA; probabilities in gallery screenshot are explicit fixtures."""
import json,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));os.environ['QT_QPA_PLATFORM']='offscreen'
from PySide6.QtGui import QFont,QFontDatabase
from PySide6.QtWidgets import QApplication
from app.domain.rock_types import COAL_CLASSES
from app.ui.dialogs.rock_types_dialog import RockTypesDialog
from app.ui.widgets.lithology_predictions import LithologyPredictionGallery

def main():
    app=QApplication.instance() or QApplication([])
    font=QFontDatabase.addApplicationFont('C:/Windows/Fonts/segoeui.ttf');families=QFontDatabase.applicationFontFamilies(font)
    if families:app.setFont(QFont(families[0],10))
    out=ROOT/'research_reports/rock_taxonomy_20261105';out.mkdir(parents=True,exist_ok=True)
    card={'classes':list(COAL_CLASSES),'noncommercial_only':True}
    dialog=RockTypesDialog(card=card,selected='Углистый аргилит');dialog.show();app.processEvents()
    dialog.grab().save(str(out/'rock_guide_ui.png'));dialog.close()
    values=[.05,.75,.07,.03,.02,.01,.01,.01,.05]
    gallery=LithologyPredictionGallery([{'class':c,'score':v} for c,v in zip(COAL_CLASSES,values)])
    gallery.resize(450,470);gallery.show();app.processEvents()
    gallery.grab().save(str(out/'rock_gallery_ui_fixture.png'));gallery.close()
    from app.ui.dialogs.local_vlm_settings_dialog import LocalVlmSettingsDialog
    settings=LocalVlmSettingsDialog({'base':str(ROOT/'models/vision/rock_taxonomy_core_20261106'),'python':sys.executable})
    settings.show();app.processEvents();settings.grab().save(str(out/'model_choices_ui.png'));settings.close()
    print('Rendered rock guide and illustrative gallery fixture. No user settings or projects accessed.')

if __name__=='__main__':main()
