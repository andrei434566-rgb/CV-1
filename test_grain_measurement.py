"""Physical-scale grain measurement, refusal cases and UI persistence."""
import json,math,unittest
from tempfile import TemporaryDirectory
from pathlib import Path
from unittest.mock import patch
from app.infrastructure.grain_measurement import classify_sand_mm,assess_measurements


class GrainMeasurementTests(unittest.TestCase):
    def measurement(self,length=40.,**extra):
        args=dict(same_plane=True,grains_visible=True,sand_context=True)
        args.update(extra)
        return assess_measurements([0,0,1000,0],5.,[[10,20+i*20,10+length,20+i*20] for i in range(5)],(1200,400),**args)
    def test_exact_wentworth_boundaries_and_no_forced_gravel_or_clay_class(self):
        for mm,label in ((.0625,'Тонкозернистый'),(.125,'Мелкозернистый'),(.25,'Среднезернистый'),(.5,'Крупнозернистый')):
            self.assertEqual(classify_sand_mm(mm),label)
        for mm in (.001,.02,1.,3.):self.assertEqual(classify_sand_mm(mm),'')
        for mm in (0,-1,float('nan'),True):
            with self.assertRaises(ValueError):classify_sand_mm(mm)
    def test_known_mm_scale_measures_pixels_and_preserves_uncertainty(self):
        r=self.measurement();self.assertAlmostEqual(r['median_apparent_mm'],.2)
        self.assertEqual(r['label'],'Мелкозернистый');self.assertTrue(r['assumed'])
        self.assertLess(r['lower_mm'],.2);self.assertGreater(r['upper_mm'],.2)
        self.assertEqual(r['method'].split(',')[0],'Median of human-selected apparent 2D grain intercepts')
    def test_zooming_or_resizing_both_coordinates_does_not_change_mm(self):
        first=self.measurement()
        second=assess_measurements([0,0,2000,0],5.,[[20,40+i*40,100,40+i*40] for i in range(5)],(2400,800),
                                   same_plane=True,grains_visible=True,sand_context=True)
        self.assertAlmostEqual(first['median_apparent_mm'],second['median_apparent_mm'])
        self.assertEqual(first['label'],second['label'])
    def test_unresolved_grains_never_become_fine_grained_by_default(self):
        r=self.measurement(3);self.assertEqual(r['label'],'');self.assertEqual(r['usable_grains'],0)
        self.assertIn('исходных пикселей',r['reason'])
    def test_boundary_within_endpoint_allowance_stays_unassigned(self):
        r=self.measurement(50);self.assertEqual(r['label'],'');self.assertEqual(r['status'],'boundary')
    def test_physical_and_observation_confirmations_are_required(self):
        for flag in ('same_plane','grains_visible','sand_context'):
            self.assertEqual(self.measurement(**{flag:False})['label'],'')
        self.assertEqual(self.measurement(same_plane='false')['label'],'')
    def test_censored_fine_grains_cannot_be_discarded_to_manufacture_a_coarse_label(self):
        lines=[[10,20+i*20,50,20+i*20] for i in range(5)]+[[10,160,13,160]]
        r=assess_measurements([0,0,1000,0],5.,lines,(1200,400),same_plane=True,grains_visible=True,sand_context=True)
        self.assertEqual(r['label'],'');self.assertIn('исказит',r['reason'])
    def test_duplicate_lines_do_not_count_as_independent_grains(self):
        r=assess_measurements([0,0,1000,0],5.,[[10,20,50,20]]*5,(1200,400),same_plane=True,grains_visible=True,sand_context=True)
        self.assertEqual(r['usable_grains'],1);self.assertEqual(r['label'],'')
    def test_invalid_reference_and_off_image_coordinates_are_rejected(self):
        for line in (None,[0,0,0,0],[0,0,10,0],[0,0,5000,0],[0,0,float('nan'),0]):
            with self.assertRaises(ValueError):assess_measurements(line,10.,[],(1000,500))


class GrainMeasurementUiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PySide6.QtWidgets import QApplication
        cls.app=QApplication.instance() or QApplication([])
    def test_measurement_ui_uses_scene_pixels_and_restores_only_same_image(self):
        from PySide6.QtGui import QPixmap,QColor
        from app.ui.dialogs.grain_measurement_dialog import GrainMeasurementDialog
        pix=QPixmap(1200,400);pix.fill(QColor(130,130,130))
        d=GrainMeasurementDialog(pix);self.addCleanup(d.close)
        d.reference_mm.setValue(5);d.add_line([0,0,1000,0]);d.view.set_mode('grain')
        for i in range(5):d.add_line([10,20+i*20,50,20+i*20])
        for c in (d.same_plane,d.visible,d.sand):c.setChecked(True)
        self.assertEqual(d.measurement['label'],'Мелкозернистый');d.accept();entry=d.result_entry
        self.assertEqual(len(entry['grain_lines']),5)
        different=QPixmap(1200,400);different.fill(QColor(140,140,140))
        other=GrainMeasurementDialog(different,entry=entry);self.addCleanup(other.close)
        self.assertIsNone(other.reference);self.assertFalse(other.same_plane.isChecked())
    def test_card_preserves_manual_grain_when_measurement_cannot_assign_and_stores_valid_result(self):
        from PySide6.QtGui import QPixmap,QColor
        from PySide6.QtCore import QSettings
        from app.ui.dialogs.facies_dialog import FaciesDialog
        from app.infrastructure.attribute_examples import pixel_digest
        with TemporaryDirectory() as folder:
            settings=QSettings(str(Path(folder)/'test.ini'),QSettings.IniFormat)
            pix=QPixmap(1200,400);pix.fill(QColor(130,130,130))
            with patch('app.ui.dialogs.facies_dialog.QSettings',return_value=settings):
                d=FaciesDialog('',0,{'Зернистость':'Крупнозернистый'},preview_pixmap=pix)
            self.addCleanup(d.close)
            unresolved=assess_measurements([0,0,1000,0],5.,[],(1200,400),image_sha256=pixel_digest(pix))
            d._apply_grain_measurement(unresolved)
            self.assertEqual(d.attribute_combos['Зернистость'].currentData(),'Крупнозернистый')
            measured=GrainMeasurementTests().measurement(image_sha256=pixel_digest(pix));d._apply_grain_measurement(measured)
            attrs=d.selected_attributes();self.assertEqual(attrs['Зернистость'],'Мелкозернистый')
            self.assertEqual(json.loads(attrs['__grain_measurement'])['reference_mm'],5.)
            self.assertIn('Зернистость',json.loads(attrs['__description_assumptions']))

if __name__=='__main__':unittest.main()
