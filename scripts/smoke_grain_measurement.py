"""Offscreen calibration UI fixture and real photo resolution illustration; no training."""
import os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));os.environ['QT_QPA_PLATFORM']='offscreen'
from PySide6.QtGui import QPixmap,QColor,QPainter,QPen,QFont,QFontDatabase
from PySide6.QtWidgets import QApplication
from app.ui.dialogs.grain_measurement_dialog import GrainMeasurementDialog

def main():
    app=QApplication.instance() or QApplication([])
    f=QFontDatabase.addApplicationFont('C:/Windows/Fonts/segoeui.ttf');families=QFontDatabase.applicationFontFamilies(f)
    if families:app.setFont(QFont(families[0],10))
    pix=QPixmap(1200,500);pix.fill(QColor('#afafa9'));p=QPainter(pix)
    p.setPen(QPen(QColor('#56564f'),2));p.setBrush(QColor('#d3d0c0'))
    for y in range(80,400,65):
        for x in range(40,1150,70):p.drawEllipse(x,y,40,35)
    p.end()
    d=GrainMeasurementDialog(pix);d.reference_mm.setValue(5.);d.add_line([20,30,1020,30]);d.view.set_mode('grain')
    for i in range(8):d.add_line([40+i*70,95,80+i*70,95])
    for c in (d.same_plane,d.visible,d.sand):c.setChecked(True)
    d.show();app.processEvents();out=ROOT/'research_reports/rock_grain_20261106';out.mkdir(exist_ok=True,parents=True)
    d.grab().save(str(out/'grain_measurement_ui_fixture.png'));d.close()
    print('UI fixture: synthetic circles are for interface verification only, not training or accuracy evidence.')

if __name__=='__main__':main()
