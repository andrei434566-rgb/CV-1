"""Non-generative boundary, process protocol and geometry protection tests."""
from __future__ import annotations
import json, os, subprocess, sys, tempfile, unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
from PIL import Image
from PySide6.QtCore import QPointF
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QApplication
from app.domain.models import FaciesDetection,PhotoRecord
from app.infrastructure.local_vision import VisionEngine,read_model_card,image_tiles,reference_answer
from app.infrastructure.cv1_analysis import IntervalJob,apply_interval_description
from app.infrastructure.facies_postprocess import UNRECOGNIZED_FACIES

ROOT=Path(__file__).resolve().parent

class VisionContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.app=QApplication.instance() or QApplication([])
    def test_rejects_generative_model_folder(self):
        with tempfile.TemporaryDirectory() as d:
            Path(d,'config.json').write_text(json.dumps({'model_type':'multimodal','architectures':['ModelForConditionalGeneration']}))
            with self.assertRaisesRegex(ValueError,'Генеративные'):read_model_card(d)
    def test_whole_column_is_tiled_without_aspect_distortion_or_white_page(self):
        image=Image.new('RGB',(600,1200),'white')
        from PIL import ImageDraw
        ImageDraw.Draw(image).rectangle((200,100,300,1100),fill=(110,110,110))
        tiles,boxes=image_tiles(image,[{'left':200,'top':100,'right':300,'bottom':1100}])
        self.assertGreater(len(tiles),1);self.assertLessEqual(len(tiles),8)
        self.assertEqual(boxes,[(200,100,300,1100)])
        for t in tiles:self.assertEqual(t.width,t.height);self.assertEqual(t.getpixel((0,0)),(110,110,110))
    def test_white_mask_does_not_become_a_rock(self):
        tiles,_=image_tiles(Image.new('RGB',(80,500),'white'),[[0,0,80,500]])
        self.assertEqual(tiles,[])
    def test_batch_replies_are_bound_to_ids_and_one_failure_is_local(self):
        engine=VisionEngine()
        def describe(r):
            if r['image']=='bad':raise OSError('unreadable')
            return {'ok':True,'answer':r['image']}
        with patch.object(engine,'describe',side_effect=describe):
            r=engine.answer({'operation':'describe_photo','items':[{'id':'b','image':'second'},{'id':'a','image':'bad'}]})
        self.assertEqual(r['items']['b']['answer'],'second');self.assertFalse(r['items']['a']['ok'])
    def test_duplicate_batch_ids_are_rejected_before_inference(self):
        with self.assertRaises(ValueError):VisionEngine().answer({'operation':'describe_batch','items':[{'id':'x'},{'id':'x'}]})
    def test_unknown_operation_fails_closed(self):
        with self.assertRaises(ValueError):VisionEngine().answer({'operation':'generate'})
    def test_network_sync_rejects_legacy_generative_release_before_copying(self):
        from app.infrastructure.model_hub import atomic_json,sync_models,SCHEMA
        with tempfile.TemporaryDirectory() as d:
            hub=Path(d)/'hub';cache=Path(d)/'cache'
            atomic_json(hub/'registry/models.json',{'schema':SCHEMA,'release_id':'legacy','models':[{
                'id':'legacy-model','version':'v1','role':'vlm_base','entrypoint':'config.json','approval':{'approved':True},
                'files':[{'path':'config.json','size':0,'sha256':'0'*64}]}]})
            with self.assertRaisesRegex(ValueError,'Генеративные'):sync_models(hub,cache)
            self.assertFalse(cache.exists())
    def response(self):
        return {'ok':True,'engine':'vision','fields':{'facies_name':UNRECOGNIZED_FACIES,'lithology':'Песчаник',
                'texture':'Требует проверки','facies_association':'Не установлена','depositional_environment':'Не установлена',
                'description':'Предположительно песчаник, серый.','hydrodynamic_regime':'Не установлен по фото'},
                'model_info':{'engine':'vision','generative':False},'field_confidence':{'lithology':.87,'facies_name':None}}
    def test_fusion_preserves_geometry_depth_and_separate_confidence(self):
        pix=QPixmap(100,200);pix.fill()
        det=FaciesDetection('Интервал YOLO',.93,[QPointF(5,5),QPointF(70,5),QPointF(70,100),QPointF(5,100)],
                            {'__geometry_only':'1'},1000,1001)
        photo=PhotoRecord('p','unused.png',pix,[det]);job=IntervalJob.create(photo,det)
        before=[(p.x(),p.y()) for p in det.polygon]
        apply_interval_description(photo,job,self.response())
        self.assertEqual(before,[(p.x(),p.y()) for p in det.polygon]);self.assertEqual((det.depth_from,det.depth_to),(1000,1001))
        self.assertEqual(det.confidence,.93);self.assertEqual(det.label,UNRECOGNIZED_FACIES)
        self.assertEqual(json.loads(det.attributes['__vision_field_confidence'])['lithology'],.87)
        self.assertFalse(det.training_ready);self.assertEqual(det.attributes['__vision_facies_trained'],'0')
    def test_expert_confirmed_interval_is_not_overwritten(self):
        pix=QPixmap(30,30);pix.fill()
        det=FaciesDetection('Эксперт',1,[QPointF(0,0),QPointF(20,0),QPointF(20,20)],{'__geometry_only':'1'})
        photo=PhotoRecord('p','x',pix,[det]);job=IntervalJob.create(photo,det);det.training_ready=True
        with self.assertRaises(ValueError):apply_interval_description(photo,job,self.response())
        self.assertEqual(det.label,'Эксперт')
    def test_reference_lookup_uses_primary_source_without_a_language_model(self):
        with tempfile.TemporaryDirectory() as d:
            index={'schema':'cv1-primary-description-source-v1','scope':['facies_name','hydrodynamic_regime'],
                   'documents':[{'name':'Источник.docx'}],'chunks':[{'document':'Источник.docx','sha256':'x','offset':0,
                   'text':'Русловая фация: утверждённое специалистом определение.'}]}
            Path(d,'index.json').write_text(json.dumps(index,ensure_ascii=False),encoding='utf-8')
            result=reference_answer({'question':'Что такое русловая фация?','primary_reference':d})
        self.assertIn('утверждённое специалистом',result['answer']);self.assertEqual(result['engine'],'reference')
        self.assertNotIn('transformers',sys.modules);self.assertNotIn('peft',sys.modules)
    def test_old_model_setting_is_not_used(self):
        from app.ui.windows import main_window
        class Settings:
            def value(self,k,default=''):return {'vlm/base':'obsolete-model','vlm/model_mode':'lora'}.get(k,default)
        with patch.object(main_window,'default_model_folder',return_value=Path('safe-classifier')):
            config=main_window.MainWindow._local_vlm_config(SimpleNamespace(_settings=Settings()))
        self.assertEqual(config['base'],'safe-classifier');self.assertEqual(config['model_mode'],'vision')
        self.assertNotIn('adapter',config);self.assertNotIn('theory_adapter',config)
    def test_worker_entry_point_routes_to_reference_engine(self):
        env=dict(os.environ,PYTHONIOENCODING='utf-8',PYTHONDONTWRITEBYTECODE='1')
        requests='\n'.join(json.dumps(r,ensure_ascii=False) for r in [
            {'id':'hi','operation':'chat','question':'работаешь?'},{'id':'bad','operation':'generate'},
            {'id':'again','operation':'chat','question':'привет'}])+'\n'
        completed=subprocess.run([sys.executable,'-B',str(ROOT/'run.py'),'--vision-worker'],
                                 input=requests,capture_output=True,text=True,encoding='utf-8',env=env,timeout=25)
        self.assertEqual(completed.returncode,0,completed.stderr)
        replies=[json.loads(s) for s in completed.stdout.splitlines()]
        self.assertEqual([r['id'] for r in replies],['hi','bad','again'])
        self.assertEqual(replies[0]['engine'],'reference');self.assertFalse(replies[1]['ok']);self.assertTrue(replies[2]['ok'])

if __name__=='__main__':unittest.main()
