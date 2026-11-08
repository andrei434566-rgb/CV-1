"""Photo-only heuristics, label masks and review state cannot approve training."""
import copy
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock,patch

os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
import numpy as np
from PIL import Image,ImageDraw,ImageFilter
from PySide6.QtCore import QPointF,QSettings
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QApplication
from app.domain.models import FaciesDetection,PhotoRecord
from app.infrastructure.photo_quality import audit_file,audit_image,exclusion_mask,label_boxes,tile_usable_mask
from app.infrastructure.interval_review_queue import ACK_KEY,acknowledge,build_interval_review_queue
from app.infrastructure.local_vision import visual_band_hint
from app.ui.dialogs.interval_review_queue_dialog import IntervalReviewQueueDialog


def labelled_photo():
    image=Image.new('RGB',(180,500),(110,110,110));draw=ImageDraw.Draw(image)
    draw.rectangle((40,180,120,210),fill='white');draw.text((46,188),'123456',fill='black')
    return image


class PhotoQualityTests(unittest.TestCase):
    def test_text_rectangle_is_excluded_but_pale_strata_are_not(self):
        image=labelled_photo();before=image.tobytes()
        boxes=label_boxes(image)
        self.assertEqual(len(boxes),1)
        mask=exclusion_mask(image.size,boxes)
        self.assertTrue(mask[190,70]);self.assertFalse(mask[50,70]);self.assertEqual(image.tobytes(),before)
        pale=Image.new('RGB',(180,500),(110,110,110));ImageDraw.Draw(pale).rectangle((0,180,180,210),fill='#dddddd')
        self.assertEqual(label_boxes(pale),[])

    def test_mask_projects_into_tile_without_repainting_the_source(self):
        image=labelled_photo();boxes=label_boxes(image)
        valid=tile_usable_mask((120,120),(30,140,150,260),image.size,boxes)
        self.assertFalse(valid[50,40]);self.assertTrue(valid[5,5])
        tile=image.crop((30,140,150,260))
        hint=visual_band_hint([tile],[valid])
        self.assertNotIn('контрастные полосы',hint)

    def test_blur_score_is_only_a_photographic_hint_and_exposure_is_measured(self):
        data=np.random.default_rng(42).integers(70,180,(240,160,3),dtype=np.uint8)
        sharp=Image.fromarray(data);blurred=sharp.filter(ImageFilter.GaussianBlur(4))
        self.assertGreater(audit_image(sharp)['metrics']['detail_score'],audit_image(blurred)['metrics']['detail_score'])
        white=Image.new('RGB',(200,300),'white')
        report=audit_image(white,[[0,0,200,300]])
        self.assertIn('bright_areas',[row['code'] for row in report['issues']])
        self.assertTrue(report['heuristic'])
        self.assertIn('светлая порода',next(row['message'] for row in report['issues'] if row['code']=='bright_areas'))

    def test_file_errors_are_results_and_exif_orientation_is_consistent(self):
        with TemporaryDirectory() as folder:
            self.assertEqual(audit_file(Path(folder)/'missing.jpg')['issues'][0]['code'],'photo_unreadable')
            image=Image.new('RGB',(100,240),'gray');exif=image.getexif();exif[274]=6
            path=Path(folder)/'oriented.jpg';image.save(path,exif=exif)
            result=audit_file(path)
            self.assertEqual(result['source_size'],[240,100]);self.assertEqual((result['width'],result['height']),(240,100))

    def test_bad_roi_metadata_does_not_crash_the_photo_pipeline(self):
        result=audit_image(labelled_photo(),[{'left':None,'right':10,'top':0,'bottom':10},[float('nan'),0,10,20]])
        self.assertEqual(result['schema'],'cv1-photo-quality-v1')

    def test_white_pixels_outside_yolo_polygon_are_not_reported_as_glare(self):
        image=Image.new('RGB',(200,300),'white');mask=Image.new('L',image.size)
        polygon=[(0,0),(199,0),(0,299)]
        ImageDraw.Draw(image).polygon(polygon,fill='gray');ImageDraw.Draw(mask).polygon(polygon,fill=255)
        report=audit_image(image,[[0,0,200,300]],valid_mask=np.asarray(mask)>0,scope='interval')
        self.assertTrue(report['geometry_mask_applied'])
        self.assertEqual(report['metrics']['overexposed_fraction'],0)
        self.assertNotIn('bright_areas',[row['code'] for row in report['issues']])


class ReviewQueueTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.app=QApplication.instance() or QApplication([])

    def setUp(self):
        self.temp=TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        pix=QPixmap(120,240);pix.fill();path=self.root/'core.png';pix.save(str(path))
        self.detection=FaciesDetection('Не определена',.4,[QPointF(10,10),QPointF(100,10),QPointF(100,220)],
            {'__geometry_only':'1','__vision_lithology_review_required':'1','Краткое описание':'Черновик.'})
        self.record=PhotoRecord('photo-id',str(path),pix,[self.detection])
        self.record.photo_quality={'schema':'cv1-photo-quality-v1','sample_sha256':'one','issues':[]}

    def interval_item(self):
        return next(row for row in build_interval_review_queue([self.record]) if row.detection is self.detection)

    def test_queue_separates_geometry_lithology_missing_fields_and_confirmation(self):
        item=self.interval_item();codes={row['code'] for row in item.reasons}
        self.assertIn('geometry_score',codes);self.assertIn('lithology_score',codes);self.assertIn('missing_fields',codes)
        attrs=copy.deepcopy(self.detection.attributes);ready=self.detection.training_ready
        acknowledge(item)
        self.assertTrue(self.interval_item().acknowledged)
        self.assertEqual(self.detection.training_ready,ready)
        self.assertEqual({k:v for k,v in self.detection.attributes.items() if k!=ACK_KEY},attrs)
        self.detection.attributes['Цвет']='Серый'
        self.assertFalse(self.interval_item().acknowledged)

    def test_new_pixels_or_geometry_reopen_a_review(self):
        item=self.interval_item();acknowledge(item)
        self.record.photo_quality['sample_sha256']='two'
        self.assertFalse(self.interval_item().acknowledged)
        acknowledge(self.interval_item());self.detection.polygon[0]=QPointF(20,10)
        self.assertFalse(self.interval_item().acknowledged)

    def test_photo_quality_and_acknowledgement_round_trip_without_changing_training(self):
        from app.infrastructure.project_storage import load_project,save_project
        self.record.photo_quality['issues']=[{'code':'few_details','message':'Проверьте резкость.','severity':1}]
        photo_item=next(row for row in build_interval_review_queue([self.record]) if row.detection is None)
        acknowledge(photo_item);save_project(self.root/'project','Проверка',[self.record],{})
        _,records,_,missing,_=load_project(self.root/'project');self.assertFalse(missing)
        self.assertEqual(records[0].photo_quality,self.record.photo_quality)
        self.assertTrue(next(row for row in build_interval_review_queue(records) if row.detection is None).acknowledged)
        self.assertFalse(records[0].detections[0].training_ready)

    def test_stale_row_cannot_be_acknowledged_and_filters_are_navigable(self):
        opened=[];changed=Mock()
        dialog=IntervalReviewQueueDialog(lambda:[self.record],lambda item:opened.append(item),changed,Mock())
        self.addCleanup(dialog.close)
        self.assertEqual(dialog.table.rowCount(),1)
        self.detection.attributes['Цвет']='Серый'
        dialog.mark_reviewed();self.assertNotIn(ACK_KEY,self.detection.attributes)
        dialog.open_selected();self.assertIs(opened[0].detection,self.detection)
        dialog.mark_reviewed();self.assertEqual(dialog.table.rowCount(),0);changed.assert_called_once()
        dialog.mode.setCurrentIndex(2);self.assertEqual(dialog.table.rowCount(),1)
        dialog.filter.setText('несуществующая');self.assertEqual(dialog.table.rowCount(),0)

    def test_deferred_items_return_after_closing_the_queue(self):
        dialog=IntervalReviewQueueDialog(lambda:[self.record],Mock(),Mock(),Mock())
        self.addCleanup(dialog.close)
        dialog.defer();self.assertEqual(dialog.table.rowCount(),0)
        dialog.close();dialog.refresh();self.assertEqual(dialog.table.rowCount(),1)

    def test_cancelled_quality_jobs_do_not_read_queued_photos(self):
        from threading import Event
        from app.infrastructure.attribute_tasks import submit,wait_for_tasks,cancel_quality_jobs
        started,release=Event(),Event();replies=[]
        def audit(*args,**kwargs):
            started.set();release.wait(2);return {'issues':[]}
        with patch('app.infrastructure.photo_quality.audit_file',side_effect=audit) as fake:
            submit('quality',{'path':'first.png','record_id':'first'},replies.append)
            self.assertTrue(started.wait(1))
            submit('quality',{'path':'second.png','record_id':'second'},replies.append)
            cancel_quality_jobs({'second'});release.set();wait_for_tasks();self.app.processEvents()
            self.assertEqual(fake.call_count,1)
        self.assertTrue(any(reply.get('cancelled') and reply['record_id']=='second' for reply in replies))

    def test_fifty_photo_preflights_are_queued_and_deduplicated(self):
        from app.infrastructure.attribute_tasks import wait_for_tasks
        from app.ui.windows.main_window import MainWindow
        settings=QSettings(str(self.root/'batch.ini'),QSettings.Format.IniFormat)
        with patch('app.ui.windows.main_window.QSettings',return_value=settings):window=MainWindow()
        self.addCleanup(window.close)
        for index in range(50):
            path=self.root/f'{index}.png';self.record.pixmap.save(str(path))
            window._records.append(PhotoRecord(str(index),str(path),self.record.pixmap))
        window._audit_review_photos();window._audit_review_photos()
        self.assertEqual(len(window._quality_pending),50)
        wait_for_tasks();self.app.processEvents()
        self.assertFalse(window._quality_pending)
        self.assertTrue(all(record.photo_quality.get('schema')=='cv1-photo-quality-v1' for record in window._records))

    def test_quality_warning_reaches_excel_without_rewriting_description(self):
        from app.infrastructure.core_report_export import export_core_description_report
        from openpyxl import load_workbook
        target=self.root/'quality.xlsx'
        quality={'issues':[{'code':'few_details','message':'Проверьте резкость.','severity':1}]}
        attrs={'__vlm_description':'Серый керн.','__image_quality':json.dumps(quality,ensure_ascii=False)}
        export_core_description_report(target,'Проверка',[{'name':'А','layers':[{'label':'F1','attributes':attrs}]}],'xlsx')
        book=load_workbook(target)
        try:
            self.assertEqual(book.active['V4'].value,'Серый керн.')
            self.assertIn('Фото: Проверьте резкость.',book.active['V4'].comment.text)
        finally:book.close()

    def test_recomputed_color_updates_only_previous_automatic_color(self):
        from app.infrastructure.cv1_analysis import IntervalJob,apply_interval_description
        response={'ok':True,'engine':'vision','lithology_review_required':True,
                  'observations':{'color':'Серый'},'fields':{'facies_name':'Не определена','lithology':'Песчаник',
                  'texture':'Требует проверки','facies_association':'Не установлена','depositional_environment':'Не установлена','description':'Серый керн.'}}
        self.detection.attributes.update({'Цвет':'Коричневый','__vision_color':'Коричневый'})
        apply_interval_description(self.record,IntervalJob.create(self.record,self.detection),response)
        self.assertEqual(self.detection.attributes['Цвет'],'Серый')
        self.detection.attributes['Цвет']='Светло-серый'
        apply_interval_description(self.record,IntervalJob.create(self.record,self.detection),response)
        self.assertEqual(self.detection.attributes['Цвет'],'Светло-серый')

    def test_queue_ack_does_not_discard_an_inflight_description(self):
        from app.infrastructure.cv1_analysis import IntervalJob,apply_interval_description
        job=IntervalJob.create(self.record,self.detection)
        acknowledge(self.interval_item())
        self.assertTrue(job.is_current(self.record))
        response={'ok':True,'engine':'vision','fields':{'facies_name':'Не определена','lithology':'Песчаник',
                  'texture':'Требует проверки','facies_association':'Не установлена',
                  'depositional_environment':'Не установлена','description':'Новый черновик.'}}
        apply_interval_description(self.record,job,response)
        self.assertFalse(self.interval_item().acknowledged)

    def test_main_window_background_quality_and_dirty_card_navigation(self):
        from app.infrastructure.attribute_tasks import wait_for_tasks
        from app.ui.windows.main_window import MainWindow
        settings=QSettings(str(self.root/'settings.ini'),QSettings.Format.IniFormat)
        with patch('app.ui.windows.main_window.QSettings',return_value=settings),patch('app.ui.dialogs.facies_dialog.QSettings',return_value=settings):
            window=MainWindow();self.addCleanup(window.close)
            window._records=[self.record];self.record.photo_quality={}
            window._audit_review_photos();wait_for_tasks();self.app.processEvents()
            self.assertEqual(self.record.photo_quality['schema'],'cv1-photo-quality-v1')
            self.assertFalse(window._quality_pending)
            window._open_facies_editor(self.record,self.detection)
            card=window._facies_dialogs[-1]
            self.assertFalse(card.has_unsaved_changes())
            card.depth_from.setText('100')
            self.assertTrue(card.has_unsaved_changes())
            self.assertFalse(window._open_review_item(self.interval_item()))


if __name__=='__main__':unittest.main()
