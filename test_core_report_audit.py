"""Completeness is not accuracy; interval auditing never repairs source data."""
import copy
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch

os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
from PySide6.QtCore import QPointF,QSettings
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QApplication
from app.domain.models import FaciesDetection,PhotoRecord
from app.infrastructure.core_report_audit import audit_report_wells
from app.infrastructure.core_report_export import export_core_description_report,prepare_report_rows
from app.ui.dialogs.project_readiness_dialog import ProjectReadinessDialog


def layer(top,base,**extra):
    return {'label':'F1','depth_from':top,'depth_to':base,'description_confirmed':True,
        'source_id':str(extra.pop('source_id',f'{top}:{base}')),'attributes':{
            '__geometry_only':'1','Название породы':'Песчаник','Цвет':'Серый','Зернистость':'Мелкозернистый',
            'Слоистость':'Волнистая','Контакт':'Резкий','Краткое описание':'Серый песчаник.',
            '__vlm_description':'Серый песчаник.'},**extra}


class CoreReportAuditTests(unittest.TestCase):
    def test_empty_project_and_unknown_depths_never_look_complete(self):
        result=audit_report_wells([])
        self.assertEqual(result.totals['rows'],0);self.assertFalse(result.to_dict()['model_accuracy_evaluated'])
        result=audit_report_wells([{'name':'А','layers':[layer(None,None)]}])
        self.assertEqual(result.totals['valid_depths'],0);self.assertEqual(result.totals['gaps'],0)
        self.assertIsNone(result.wells[0]['depth_from'])
        self.assertIn('depth_missing',[issue.code for issue in result.issues])

    def test_nested_overlaps_do_not_create_false_gaps_or_double_count_length(self):
        wells=[{'name':'А','layers':[layer(100,110),layer(101,102),layer(110,112)]}]
        before=copy.deepcopy(wells);result=audit_report_wells(wells)
        self.assertEqual(result.totals['gaps'],0);self.assertEqual(result.totals['overlaps'],1)
        self.assertEqual(result.wells[0]['mapped_length'],12)
        self.assertEqual(wells,before)

    def test_gaps_and_overlaps_are_separate_review_questions(self):
        result=audit_report_wells([{'name':'А','layers':[layer(100,101),layer(102,103),layer(102.5,104)]}])
        self.assertEqual(result.totals['gaps'],1);self.assertEqual(result.totals['overlaps'],1)
        gap=next(issue for issue in result.issues if issue.code=='data_gap')
        self.assertIn('перерыв отбора',gap.message);self.assertEqual(gap.rows,(0,1))
        overlap=next(issue for issue in result.issues if issue.code=='data_overlap')
        self.assertIn('повторные снимки',overlap.message)
        self.assertEqual(result.wells[0]['mapped_length'],3)

    def test_declared_edges_and_passport_bounds_are_checked_without_filling_them(self):
        result=audit_report_wells([{'name':'А','core_interval':[99,105],'layers':[layer(100,103)]}])
        self.assertEqual(result.totals['gaps'],2)
        result=audit_report_wells([{'name':'А','excel_report_inputs':{'core_top':100,'core_base':102},'layers':[layer(99,103)]}])
        self.assertIn('outside_core_interval',[issue.code for issue in result.issues])

    def test_wells_and_depth_units_do_not_mix(self):
        result=audit_report_wells([{'name':'А','layers':[layer(100,101)]},
            {'name':'Б','depth_reference':'TVD','depth_settings':{'unit':'ft'},'layers':[layer(100,101)]}])
        self.assertEqual(result.totals['overlaps'],0)
        self.assertIn('depth_units',[issue.code for issue in result.issues])
        self.assertEqual([well['mapped_length'] for well in result.wells],[1,1])

    def test_assumptions_and_confirmation_are_counted_separately(self):
        row=layer(100,101,description_confirmed=False)
        row['attributes']['__description_assumptions']=json.dumps(['Зернистость','Название фации'],ensure_ascii=False)
        result=audit_report_wells([{'name':'А','layers':[row]}])
        self.assertEqual(result.totals['filled_fields'],5);self.assertEqual(result.totals['assumed_fields'],1)
        self.assertEqual(result.totals['confirmed_descriptions'],0);self.assertEqual(result.totals['assumed_facies'],1)

    def test_invalid_nonfinite_and_reversed_depths_are_excluded_from_coverage(self):
        result=audit_report_wells([{'layers':[layer(102,100),layer(float('nan'),103),layer(104,float('inf'))]}])
        self.assertEqual(result.totals['valid_depths'],0);self.assertEqual(result.wells[0]['mapped_length'],0)
        self.assertIn('depth_invalid',[issue.code for issue in result.issues])
        json.dumps(result.to_dict(),allow_nan=False)

    def test_prepared_drilling_rows_preserve_all_parent_review_requirements(self):
        row=layer(100,104,source_id='one')
        well={'layers':[row],'excel_report_inputs':{'drilling_runs':[
            {'number':'1','top':100,'base':102},{'number':'2','top':102,'base':104}]}}
        _,rows=prepare_report_rows(well);self.assertEqual(len(rows),2)
        audit=audit_report_wells([{**well,'layers':rows}])
        self.assertEqual(audit.totals['rows'],2);self.assertEqual(audit.totals['overlaps'],0)
        self.assertTrue(all(part['source_id']=='one' for part in rows))
        first,second=layer(100,101,source_id='one'),layer(101,102,source_id='two',description_confirmed=False)
        first['attributes'].pop('__geometry_only');second['attributes'].pop('__geometry_only')
        _,merged=prepare_report_rows({'layers':[first,second]})
        self.assertEqual(len(merged),1);self.assertFalse(merged[0]['description_confirmed'])
        self.assertEqual(merged[0]['source_ids'],['one','two'])
        first['attributes']['__description_assumptions']=json.dumps(['Цвет'],ensure_ascii=False)
        second['attributes']['__description_assumptions']=json.dumps(['Зернистость'],ensure_ascii=False)
        _,merged=prepare_report_rows({'layers':[first,second]})
        self.assertEqual(set(json.loads(merged[0]['attributes']['__description_assumptions'])),{'Цвет','Зернистость'})

    def test_excel_keeps_31_columns_and_adds_gaps_to_both_related_rows(self):
        from openpyxl import load_workbook
        well={'name':'А','layers':[layer(100,101),layer(102,103)]}
        original=copy.deepcopy(well)
        with TemporaryDirectory() as folder:
            path=Path(folder)/'report.xlsx';export_core_description_report(path,'Проверка',[well],'xlsx')
            book=load_workbook(path)
            try:
                self.assertEqual(book.active.max_column,31)
                self.assertIn('101–102',book.active['AD4'].comment.text)
                self.assertIn('101–102',book.active['AD5'].comment.text)
                self.assertEqual(book.active['AD4'].value,'Серый песчаник.')
            finally:book.close()
        self.assertEqual(well,original)


class ProjectReadinessUiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.app=QApplication.instance() or QApplication([])

    def setUp(self):
        self.temp=TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)

    def test_summary_refresh_and_source_navigation_are_read_only(self):
        wells=[{'name':'А','layers':[layer(100,101),layer(102,103)]}]
        audit=audit_report_wells(wells);record=SimpleNamespace(path='one.jpg');target=SimpleNamespace(record=record,detection=None)
        targets={source:target for issue in audit.issues for source in issue.source_ids}
        state=[audit,targets,0];opened=Mock();export=Mock()
        dialog=ProjectReadinessDialog(lambda:tuple(state),opened,export);self.addCleanup(dialog.close)
        self.assertIn('строк отчёта: 2',dialog.overall.text())
        self.assertFalse(opened.called);self.assertFalse(export.called)
        dialog.query.setText('нет описаний');self.assertEqual(dialog.issue_table.rowCount(),1)
        dialog.open_selected();opened.assert_called_once_with(target)
        self.assertEqual(dialog.target_selector.count(),2)
        dialog.target_selector.setCurrentIndex(1);selected=dialog.target_selector.currentData()
        dialog.refresh();self.assertEqual(dialog.target_selector.currentData(),selected)
        state[0]=audit_report_wells([{'name':'А','layers':[layer(100,101),layer(101,102)]}])
        dialog.open_selected();self.assertEqual(opened.call_count,1)

    def test_main_snapshot_does_not_render_stack_or_count_withdrawn_confirmation(self):
        from app.ui.windows.main_window import MainWindow
        from app.infrastructure.vlm_training_data import expert_description_digest,expert_description_is_confirmed
        settings=QSettings(str(self.root/'private.ini'),QSettings.Format.IniFormat)
        with patch('app.ui.windows.main_window.QSettings',return_value=settings):window=MainWindow()
        self.addCleanup(window.close)
        photo=QPixmap(120,240);photo.fill();path=self.root/'photo.png';photo.save(str(path))
        attrs=layer(100,101)['attributes']
        detection=FaciesDetection('F1',.9,[QPointF(0,0),QPointF(100,0),QPointF(100,200)],attrs,100,101,True)
        attrs['__expert_description_confirmed']='1';attrs['__expert_memory_confirmed']='0'
        attrs['__expert_description_sha256']=expert_description_digest(detection)
        self.assertTrue(expert_description_is_confirmed(detection))
        record=PhotoRecord('one',str(path),photo,[detection],well_name='А')
        window._records=[record];window._active_well_name='А'
        with patch.object(window,'_show_stack') as render:
            audit,targets,dirty=window._project_readiness_snapshot()
        render.assert_not_called();self.assertEqual(audit.totals['confirmed_descriptions'],0)
        self.assertIs(targets['one:0'].detection,detection);self.assertEqual(dirty,0)

    def test_export_with_missing_data_stays_direct_and_does_not_modify_project(self):
        from app.ui.windows.main_window import MainWindow
        from openpyxl import load_workbook
        settings=QSettings(str(self.root/'export.ini'),QSettings.Format.IniFormat)
        with patch('app.ui.windows.main_window.QSettings',return_value=settings):window=MainWindow()
        self.addCleanup(window.close)
        photo=QPixmap(100,200);photo.fill();path=self.root/'core.png';photo.save(str(path))
        detection=FaciesDetection('Не определена',.5,[QPointF(0,0),QPointF(90,0),QPointF(90,190)],
                                  {'__geometry_only':'1','__vlm_description':'Черновое описание.'})
        window._records=[PhotoRecord('one',str(path),photo,[detection],well_name='А')]
        original=copy.deepcopy(detection.attributes);target=self.root/'draft.xlsx'
        with patch('app.ui.windows.main_window.QFileDialog.getSaveFileName',return_value=(str(target),'Книга Excel (*.xlsx)')), \
             patch('app.ui.windows.main_window.QMessageBox.critical') as error,patch.object(window,'_show_stack'):
            window.export_core_description_report()
        error.assert_not_called();self.assertTrue(target.is_file())
        self.assertEqual(detection.attributes,original)
        self.assertIn('замечаний:',window.statusBar().currentMessage())
        book=load_workbook(target)
        try:self.assertEqual(book.active.max_column,31)
        finally:book.close()

    def test_summary_closes_with_main_window_and_stops_its_refresh_timer(self):
        from app.ui.windows.main_window import MainWindow
        settings=QSettings(str(self.root/'close.ini'),QSettings.Format.IniFormat)
        with patch('app.ui.windows.main_window.QSettings',return_value=settings):window=MainWindow()
        self.addCleanup(window.close)
        window.show();window.show_project_readiness();dialog=window._readiness_dialog
        self.addCleanup(dialog.close)
        window.show_interval_review_queue();queue=window._review_queue_dialog
        self.addCleanup(queue.close)
        self.assertTrue(dialog.isVisible())
        window.close()
        self.assertFalse(dialog.isVisible())
        self.assertFalse(dialog.timer.isActive())
        self.assertFalse(queue.isVisible());self.assertFalse(queue.timer.isActive())


if __name__=='__main__':unittest.main()
