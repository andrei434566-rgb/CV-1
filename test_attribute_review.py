"""Review-only rules, pixel-bound feature labels, revocation and async retrieval."""
import copy
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
from PySide6.QtCore import QPointF, QRectF, QSettings
from PySide6.QtGui import QColor, QPainter, QPixmap
from PySide6.QtWidgets import QApplication
from app.infrastructure.attribute_examples import (EVIDENCE_KEY, export_attribute_dataset, find_similar,
    pixel_digest, png_bytes, read_examples, revoke_example, valid_evidence, write_example)
from app.infrastructure.description_review import facies_candidates, review_description
from app.ui.dialogs.facies_dialog import FaciesDialog
from app.ui.dialogs.feature_evidence_dialog import FeatureEvidenceDialog


class ReviewRulesTests(unittest.TestCase):
    def test_rules_are_non_mutating_review_questions_not_forced_geology(self):
        attrs={'Текстура':'Массивная','Слоистость':'Волнистая'}; before=dict(attrs)
        hints=review_description(attrs)
        self.assertEqual(attrs,before);self.assertEqual(hints[0].code,'massive_and_bedded')
        self.assertIn('масштабам',hints[0].message)
        self.assertFalse(review_description({'Текстура':'Массивная','Слоистость':''}))
        self.assertTrue(review_description({'Текстура':'Биотурбационная','Биотурбация':'Видимые следы не обнаружены'}))

    def test_facies_needs_two_verified_non_assumed_diagnostic_matches(self):
        row={'id':'a','description_reviewed':True,'facies_reviewed':True,'facies':'F1',
             'observations':{'Название породы':'Песчаник','Слоистость':'Волнистая'}}
        self.assertFalse(facies_candidates({'Название породы':'Песчаник'},[row]))
        query={'Название породы':'Песчаник','Слоистость':'Волнистая'}
        result=facies_candidates(query,[row]);self.assertEqual(result[0]['matched'],list(query))
        self.assertFalse(result[0]['score_is_probability'])
        self.assertFalse(facies_candidates(query,[row],assumptions=['Слоистость']))
        row['facies_reviewed']=False;self.assertFalse(facies_candidates(query,[row]))


class AttributeLibraryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app=QApplication.instance() or QApplication([])

    def setUp(self):
        self.temp=TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)/'library'
        self.photo=QPixmap(120,240);self.photo.fill(QColor('#888888'))
        painter=QPainter(self.photo);painter.fillRect(0,90,120,20,QColor('#444444'));painter.end()
        self.digest=pixel_digest(self.photo);self.data=png_bytes(self.photo)
        self.entry={'roi':[.1,.2,.9,.8],'value':'Волнистая','image_sha256':self.digest,'reviewed':True}
        self.attrs={'Слоистость':'Волнистая','Название породы':'Песчаник','Краткое описание':'Песчаник. Слоистость волнистая.',
            '__description_mode':'template',EVIDENCE_KEY:json.dumps({'Слоистость':self.entry},ensure_ascii=False)}
        self.payload={'id':'a'*32,'well':'Скважина А','attributes':self.attrs,'pixel_sha256':self.digest,
                      'description_reviewed':True,'training_ready':True,'facies':'F1'}

    def test_only_reviewed_pixel_matched_roi_is_exported_and_revoke_removes_it(self):
        row=write_example(self.payload,self.data,root=self.root)
        self.assertEqual(len(row['targets']),1)
        self.assertEqual(row['image_kind'],'displayed_mask_crop')
        report=export_attribute_dataset(Path(self.temp.name)/'dataset',root=self.root)
        self.assertEqual(report['examples'],1);self.assertFalse(report['trained'])
        revoke_example(self.payload['id'],root=self.root)
        self.assertFalse(read_examples(root=self.root))
        report=export_attribute_dataset(Path(self.temp.name)/'revoked_dataset',root=self.root)
        self.assertEqual(report['examples'],0)

    def test_assumption_value_change_and_moved_pixels_cannot_be_training_labels(self):
        payload=copy.deepcopy(self.payload)
        payload['attributes']['__description_assumptions']=json.dumps(['Слоистость'],ensure_ascii=False)
        self.assertFalse(write_example(payload,self.data,root=self.root)['targets'])
        payload=copy.deepcopy(self.payload);payload['attributes']['Слоистость']='Мелкая косая'
        self.assertFalse(write_example(payload,self.data,root=self.root)['targets'])
        payload=copy.deepcopy(self.payload);payload['pixel_sha256']='not_the_picture'
        with self.assertRaises(ValueError):write_example(payload,self.data,root=self.root)
        self.assertFalse(valid_evidence({'roi':[0,0,float('nan'),1],'value':'Волнистая','image_sha256':self.digest},'Волнистая',self.digest))

    def test_withdrawn_description_and_unreviewed_targets_do_not_enter_analogs(self):
        payload=copy.deepcopy(self.payload);payload['description_reviewed']=False
        evidence=copy.deepcopy(self.entry);evidence['reviewed']=False
        payload['attributes'][EVIDENCE_KEY]=json.dumps({'Слоистость':evidence})
        write_example(payload,self.data,root=self.root)
        self.assertFalse(read_examples(root=self.root))

    def test_visual_retrieval_excludes_self_and_corrupt_files(self):
        row=write_example(self.payload,self.data,root=self.root)
        other=QPixmap(120,240);other.fill(QColor('#909090'))
        hits=find_similar(png_bytes(other),root=self.root)
        self.assertEqual(hits[0]['id'],row['id'])
        self.assertFalse(find_similar(self.data,root=self.root))
        self.assertFalse(find_similar(png_bytes(other),root=self.root,exclude_id=row['id']))
        (self.root/row['image']).write_bytes(b'corrupt')
        self.assertFalse(find_similar(png_bytes(other),root=self.root))

    def test_unmentioned_manual_form_field_is_not_approved_by_text_checkbox(self):
        payload=copy.deepcopy(self.payload);payload['attributes']['__description_mode']='manual'
        payload['attributes']['Краткое описание']='Серый керн.'
        row=write_example(payload,self.data,root=self.root)
        self.assertNotIn('Название породы',row['observations'])
        self.assertEqual(row['observations']['Слоистость'],'Волнистая')

    def test_conflicting_labels_for_identical_pixels_are_quarantined(self):
        write_example(self.payload,self.data,root=self.root)
        payload=copy.deepcopy(self.payload);payload['id']='b'*32
        payload['attributes']['Слоистость']='Мелкая косая'
        entry=copy.deepcopy(self.entry);entry['value']='Мелкая косая'
        payload['attributes'][EVIDENCE_KEY]=json.dumps({'Слоистость':entry})
        write_example(payload,self.data,root=self.root)
        report=export_attribute_dataset(Path(self.temp.name)/'conflicts',root=self.root)
        self.assertEqual(report['examples'],0)
        self.assertEqual(len(report['conflicting_labels_excluded']),1)

    def test_excel_attaches_review_question_and_keeps_the_expert_text_intact(self):
        from app.infrastructure.core_report_export import export_core_description_report
        from openpyxl import load_workbook
        attrs={'__vlm_description':'Текст специалиста без перезаписи.',
               '__description_review':json.dumps([h.to_dict() for h in review_description({'Текстура':'Массивная','Слоистость':'Волнистая'})],ensure_ascii=False)}
        target=Path(self.temp.name)/'review.xlsx'
        export_core_description_report(target,'Проверка',[{'name':'А','layers':[{'label':'F1','attributes':attrs}]}],'xlsx')
        book=load_workbook(target)
        try:
            cell=book.active['V4'];self.assertEqual(cell.value,attrs['__vlm_description'])
            self.assertIn('разным прослоям',cell.comment.text)
            self.assertTrue(any(rule.formula[0].startswith('EXACT(V4,') for group in book.active.conditional_formatting._cf_rules.values() for rule in group))
        finally:book.close()

    def test_card_invalidates_old_evidence_and_shows_contradiction_without_overwriting(self):
        settings=QSettings(str(Path(self.temp.name)/'settings.ini'),QSettings.Format.IniFormat)
        with patch('app.ui.dialogs.facies_dialog.QSettings',return_value=settings):
            dialog=FaciesDialog('',.8,self.attrs,preview_pixmap=self.photo)
        self.addCleanup(dialog.close)
        self.assertIn('Слоистость',dialog._current_evidence())
        texture=dialog.attribute_combos['Текстура'];texture.setCurrentIndex(texture.findData('Массивная'))
        self.assertIn('разным прослоям',dialog.review_notes.toPlainText())
        combo=dialog.attribute_combos['Слоистость'];combo.setCurrentIndex(combo.findData('Мелкая косая'))
        self.assertFalse(dialog._current_evidence())
        self.assertEqual(dialog.attribute_combos['Текстура'].currentData(),'Массивная')

    def test_roi_edit_revokes_previous_training_approval(self):
        dialog=FeatureEvidenceDialog(self.photo,'Слоистость','Волнистая',self.entry)
        self.addCleanup(dialog.close)
        self.assertTrue(dialog.reviewed.isChecked())
        dialog.view.selection_changed.emit()
        self.assertFalse(dialog.reviewed.isChecked())
        dialog.view.rectangle.setRect(QRectF(10,20,60,80));dialog.accept()
        self.assertFalse(dialog.result_entry['reviewed'])
        assumed=FeatureEvidenceDialog(self.photo,'Слоистость','Волнистая',self.entry,assumed=True)
        self.addCleanup(assumed.close)
        self.assertFalse(assumed.reviewed.isEnabled());self.assertFalse(assumed.reviewed.isChecked())

    def test_worker_saves_then_retrieves_without_generation(self):
        from app.infrastructure.attribute_tasks import submit,wait_for_tasks
        replies=[]
        submit('save',{'payload':self.payload,'image_png':self.data,'root':self.root},replies.append)
        wait_for_tasks();self.app.processEvents()
        self.assertTrue(replies[-1]['ok'])
        other=QPixmap(120,240);other.fill(QColor('#909090'))
        submit('similar',{'image_png':png_bytes(other),'root':self.root,'exclude_id':'',
                         'attributes':{'Название породы':'Песчаник','Слоистость':'Волнистая'},
                         'assumptions':[],'query_signature':'one'},replies.append)
        wait_for_tasks();self.app.processEvents()
        self.assertTrue(replies[-1]['ok'])
        result=replies[-1]['result'];self.assertEqual(result['query_signature'],'one')
        self.assertEqual(len(result['similar']),1)
        self.assertEqual(result['candidates'][0]['facies'],'F1')

    def test_main_window_saves_reviewed_roi_and_revokes_it_after_contour_edit(self):
        from app.domain.models import FaciesDetection,PhotoRecord
        from app.infrastructure.attribute_tasks import wait_for_tasks
        from app.ui.dialogs.facies_dialog import make_interval_preview
        from app.ui.windows.main_window import MainWindow
        settings=QSettings(str(Path(self.temp.name)/'app.ini'),QSettings.Format.IniFormat)
        patches=[patch('app.ui.windows.main_window.QSettings',return_value=settings),
                 patch('app.ui.dialogs.facies_dialog.QSettings',return_value=settings),
                 patch('app.infrastructure.attribute_examples.default_root',return_value=self.root),
                 patch('app.infrastructure.expert_corrections._root',return_value=Path(self.temp.name)/'feedback')]
        for item in patches:item.start();self.addCleanup(item.stop)
        window=MainWindow();self.addCleanup(window.close)
        path=Path(self.temp.name)/'core.png';self.photo.save(str(path))
        detection=FaciesDetection('Интервал YOLO',.8,[QPointF(10,10),QPointF(110,10),QPointF(110,230),QPointF(10,230)],{'__geometry_only':'1'})
        record=PhotoRecord('real-ui',str(path),self.photo,[detection],well_name='А',analysis_mode='yolo_vlm')
        window._records=[record];window._active_well_name='А';window._project_folder=Path(self.temp.name)/'project'
        crop=make_interval_preview(record.pixmap,detection.polygon)
        attrs=copy.deepcopy(self.attrs);entry=copy.deepcopy(self.entry);entry['image_sha256']=pixel_digest(crop)
        attrs[EVIDENCE_KEY]=json.dumps({'Слоистость':entry},ensure_ascii=False)
        dialog=FaciesDialog('',.8,attrs,preview_pixmap=crop)
        self.addCleanup(dialog.close);dialog.description_confirmed.setChecked(True)
        window._save_lithology(record,detection,dialog)
        wait_for_tasks();self.app.processEvents()
        rows=read_examples(root=self.root)
        self.assertEqual(len(rows),1);self.assertEqual(len(rows[0]['targets']),1)
        self.assertFalse(rows[0]['facies_reviewed'])
        detection.polygon[0]=QPointF(20,10)
        window._on_facies_geometry_changed(record,detection)
        wait_for_tasks();self.app.processEvents()
        self.assertFalse(read_examples(root=self.root))


if __name__=='__main__':unittest.main()
