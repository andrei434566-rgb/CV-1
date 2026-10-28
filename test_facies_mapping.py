"""Expert facies mappings preserve provenance and do not leak across intervals."""
import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

from PySide6.QtCore import QPointF
from app.domain.models import FaciesDetection
from app.infrastructure.facies_mapping import (
    MAPPING_KEY, current_expert_facies_mapping, facies_mapping_context,
    save_expert_facies_mapping, source_checkpoint_identity,
)


class FaciesMappingTests(unittest.TestCase):
    def setUp(self):
        temp_root = Path(__file__).resolve().parent / ".test-temp-local-runtime"
        temp_root.mkdir(exist_ok=True)
        self.folder = TemporaryDirectory(dir=temp_root)
        self.addCleanup(self.folder.cleanup)
        self.photo = Path(self.folder.name) / 'core.jpg'
        self.photo.write_bytes(b'photo')
        self.record = SimpleNamespace(path=str(self.photo), identifier='photo-1', well_name='R-31',
                                      pixmap=SimpleNamespace(width=lambda:100, height=lambda:200))
        self.attrs = {'__source_model_class':'core-layer', '__source_model_class_id':'2',
                      '__source_model_name':'best.pt', '__source_model_sha256':'b'*64,
                      'Индекс фации':'80', 'Название фации':'Русловая фация'}
        self.det = FaciesDetection('DWCh', 0.81, [QPointF(1,2), QPointF(50,2), QPointF(50,80)],
                                   dict(self.attrs), 10., 12., True)

    def test_record_keeps_original_yolo_identity_and_expert_context(self):
        row = save_expert_facies_mapping(self.record, self.det, confirmed=True)
        self.assertTrue(row['confirmed'])
        self.assertTrue(row['training_ready'])
        for key, value in self.attrs.items():
            self.assertEqual(self.det.attributes[key],value)
        self.assertEqual(row['source_class'], 'core-layer')
        self.assertEqual(row['model_sha256'], 'b'*64)
        self.assertEqual(current_expert_facies_mapping(self.record,self.det), row)
        context=facies_mapping_context(self.record,self.det)
        self.assertIn('DWCh',context)
        self.assertIn('core-layer',context)
        self.assertIn('только к текущему контуру',context)
        self.assertIn('сама по себе не доказывает фацию',context)

    def test_new_saved_facies_replaces_previous_mapping_not_yolo_source(self):
        save_expert_facies_mapping(self.record,self.det,confirmed=True)
        self.det.label='DSf'
        self.det.attributes['Индекс фации']='42'
        self.assertIsNone(current_expert_facies_mapping(self.record,self.det))
        row=save_expert_facies_mapping(self.record,self.det,confirmed=True)
        self.assertEqual(row['expert_facies_label'],'DSf')
        self.assertEqual(row['expert_facies_index'],'42')
        self.assertEqual(self.det.attributes['__source_model_class'],'core-layer')
        self.assertEqual(current_expert_facies_mapping(self.record,self.det),row)

    def test_unconfirmed_revokes_old_context(self):
        save_expert_facies_mapping(self.record,self.det,confirmed=True)
        save_expert_facies_mapping(self.record,self.det,confirmed=False)
        self.assertNotIn(MAPPING_KEY,self.det.attributes)
        self.assertEqual(facies_mapping_context(self.record,self.det),'')

    def test_changed_photo_contour_well_class_and_training_status_invalidate(self):
        def make():
            save_expert_facies_mapping(self.record,self.det,confirmed=True)
        make(); self.photo.write_bytes(b'other'); self.assertIsNone(current_expert_facies_mapping(self.record,self.det))
        self.photo.write_bytes(b'photo'); make(); self.det.polygon[0]=QPointF(4,2); self.assertIsNone(current_expert_facies_mapping(self.record,self.det))
        make(); self.record.well_name='other'; self.assertIsNone(current_expert_facies_mapping(self.record,self.det))
        make(); self.det.attributes['__source_model_class_id']='3'; self.assertIsNone(current_expert_facies_mapping(self.record,self.det))
        make(); self.det.training_ready=False; self.assertIsNone(current_expert_facies_mapping(self.record,self.det))

    def test_exact_image_bytes_and_normalized_polygon_survive_project_copy_and_scaling(self):
        row=save_expert_facies_mapping(self.record,self.det,confirmed=True)
        copy=Path(self.folder.name)/'copy.jpg'; copy.write_bytes(b'photo'); self.record.path=str(copy)
        self.record.pixmap=SimpleNamespace(width=lambda:200,height=lambda:400)
        self.det.polygon=[QPointF(p.x()*2,p.y()*2) for p in self.det.polygon]
        self.assertEqual(current_expert_facies_mapping(self.record,self.det),row)

    def test_missing_weights_are_explicitly_unidentified(self):
        identity=source_checkpoint_identity(Path(self.folder.name)/'best.pt')
        self.assertEqual(identity,{'name':'best.pt','sha256':''})
        self.det.attributes['__source_model_name']=identity['name']
        self.det.attributes['__source_model_sha256']=identity['sha256']
        row=save_expert_facies_mapping(self.record,self.det,confirmed=True)
        self.assertFalse(row['model_sha256'])
        self.assertIn('идентичность файла весов не подтверждена',facies_mapping_context(self.record,self.det))
        self.assertNotIn('SHA256',facies_mapping_context(self.record,self.det))

    def test_weights_fingerprint_depends_on_bytes_not_name(self):
        a=Path(self.folder.name)/'best.pt'; b=Path(self.folder.name)/'renamed.pt'
        a.write_bytes(b'weights'); b.write_bytes(b'weights')
        self.assertEqual(source_checkpoint_identity(a)['sha256'], hashlib.sha256(b'weights').hexdigest())
        self.assertEqual(source_checkpoint_identity(a)['sha256'],source_checkpoint_identity(b)['sha256'])
        b.write_bytes(b'other')
        self.assertNotEqual(source_checkpoint_identity(a)['sha256'],source_checkpoint_identity(b)['sha256'])

    def test_geometry_only_without_chosen_facies_is_not_confirmable(self):
        self.det.label='Интервал YOLO'
        with self.assertRaises(ValueError): save_expert_facies_mapping(self.record,self.det,confirmed=True)
        self.assertNotIn(MAPPING_KEY,self.det.attributes)

    def test_expert_mapping_survives_saving_and_reopening_project(self):
        from PySide6.QtGui import QPixmap
        from PySide6.QtWidgets import QApplication
        from app.domain.models import PhotoRecord
        from app.infrastructure.project_storage import load_project, save_project

        app = QApplication.instance() or QApplication([])
        self.assertIsNotNone(app)
        image = QPixmap(100, 200)
        image.fill()
        path = Path(self.folder.name) / "core.png"
        self.assertTrue(image.save(str(path)))
        record = PhotoRecord("photo-1", str(path), image, detections=[self.det], well_name="R-31")
        save_expert_facies_mapping(record, self.det, confirmed=True)
        folder = Path(self.folder.name) / "project"
        save_project(folder, "Test", [record], {})
        _, records, _, missing, _ = load_project(folder)
        self.assertEqual(missing, [])
        self.assertEqual(len(records), 1)
        self.assertIn("DWCh", facies_mapping_context(records[0], records[0].detections[0]))
        self.assertEqual(records[0].detections[0].attributes["__source_model_class"], "core-layer")


if __name__=='__main__':
    unittest.main()
