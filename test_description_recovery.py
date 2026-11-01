"""Premature EOS, partial fields and token limits do not masquerade as geology."""
import json,tempfile,unittest
import subprocess,sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock,patch
from PIL import Image
from app.infrastructure.description_output import validate_description,description_budget,record_description_diagnostics
from app.infrastructure.local_vlm_runner import LocalModel,DESCRIPTION_FIELDS

def valid():return {key:'Серый керн со слойками.' for key in DESCRIPTION_FIELDS}

class DescriptionRecoveryTests(unittest.TestCase):
    def test_worker_imports_in_portable_isolated_python(self):
        script=Path(__file__).parent/'app/infrastructure/local_vlm_runner.py'
        run=subprocess.run([sys.executable,'-I',str(script)],input=json.dumps({'operation':'search','id':'isolated','folders':[]})+'\n',
                           text=True,encoding='utf8',capture_output=True,timeout=30)
        self.assertEqual(run.returncode,0,run.stderr)
        self.assertTrue(json.loads(run.stdout)['ok'])

    def test_early_eos_is_distinct_from_a_token_limit(self):
        result=validate_description({'answer':'{ "description": "С','incomplete':False,
            'diagnostics':{'finish_reason':'eos','output_tokens':9}},DESCRIPTION_FIELDS)
        self.assertFalse(result['ok']);self.assertEqual(result['diagnostics']['validation_error'],'invalid_json')
        self.assertEqual(result['diagnostics']['finish_reason'],'eos')
        self.assertNotIn('fields',result)

    def test_complete_object_inside_prose_is_accepted_without_inventing_fields(self):
        result=validate_description({'answer':'Результат:\n```JSON\n'+json.dumps(valid())+'\n```'},DESCRIPTION_FIELDS)
        self.assertEqual(result['fields'],valid())
        missing=valid();missing.pop('texture')
        result=validate_description({'answer':json.dumps(missing)},DESCRIPTION_FIELDS)
        self.assertEqual(result['diagnostics']['missing_fields'],['texture']);self.assertFalse(result['ok'])

    def test_primary_reference_requires_seven_real_fields(self):
        result=validate_description({'answer':json.dumps(valid())},(*DESCRIPTION_FIELDS,'hydrodynamic_regime'))
        self.assertIn('hydrodynamic_regime',result['error']);self.assertNotIn('fields',result)

    def test_single_retries_only_once_and_preserves_failure_evidence(self):
        model=LocalModel()
        with patch.object(model,'_answer_uncached',side_effect=[{'answer':'{"description":"С'},
               {'answer':json.dumps(valid()),'incomplete':False}]) as infer:
            result=model.answer({'operation':'describe'},[])
        self.assertTrue(result['ok']);self.assertEqual(infer.call_count,2)
        retry=infer.call_args_list[1].args[0]
        self.assertTrue(retry['_description_retry']);self.assertEqual(retry['max_new_tokens'],768)
        self.assertEqual(result['diagnostics']['recovery']['initial']['validation_error'],'invalid_json')

    def test_repeated_failure_stops_and_is_logged_in_russian(self):
        model=LocalModel()
        with patch.object(model,'_answer_uncached',side_effect=lambda *_:{'answer':'{"description":"С'}) as infer:
            result=model.answer({'operation':'describe'},[])
        self.assertFalse(result['ok']);self.assertTrue(result['recoverable']);self.assertEqual(infer.call_count,2)
        with tempfile.TemporaryDirectory() as folder:
            record_description_diagnostics(result,folder)
            row=json.loads((Path(folder)/'vlm_diagnostics.jsonl').read_text(encoding='utf8'))
            self.assertEqual(row['diagnostics']['recovery']['initial_answer'],'{"description":"С')

    def test_batched_failure_retries_only_its_own_image_and_identity(self):
        import torch
        class Inputs(dict):
            def to(self,_device):return self
        class Processor:
            def apply_chat_template(self,*a,**k):return 'prompt'
            def __call__(self,**k):return Inputs(input_ids=torch.tensor([[1,2],[1,2]]))
            def batch_decode(self,*a,**k):return ['{"description":"С',json.dumps(valid())]
        model=LocalModel();model.processor=Processor()
        model.model=SimpleNamespace(generation_config=SimpleNamespace(eos_token_id=9),
            generate=Mock(return_value=torch.tensor([[1,2,3,9],[1,2,4,9]])))
        with tempfile.TemporaryDirectory() as folder:
            image=Path(folder)/'core.png';Image.new('RGB',(40,160),'gray').save(image)
            items=[{'id':'bad','image':str(image),'context':'first'},{'id':'good','image':str(image),'context':'second'}]
            with patch.object(model,'load'),patch('torch.cuda.synchronize'),patch.object(model,'_retry_description',return_value={'ok':True,'fields':valid()}) as retry:
                result=model._answer_photo_batch({'items':items})
        retry.assert_called_once();self.assertEqual(retry.call_args.args[1]['id'],'bad')
        self.assertEqual(set(result['items']),{'bad','good'});self.assertTrue(result['items']['good']['ok'])
        self.assertFalse(model._photo_batch_unreliable)

    def test_bad_full_batch_switches_subsequent_photos_to_serial(self):
        import torch
        class Inputs(dict):
            def to(self,_device):return self
        class Processor:
            def apply_chat_template(self,*a,**k):return 'prompt'
            def __call__(self,**k):return Inputs(input_ids=torch.tensor([[1,2]]))
            def batch_decode(self,*a,**k):return ['{"description":"С']
        model=LocalModel();model.processor=Processor()
        model.model=SimpleNamespace(generation_config=SimpleNamespace(eos_token_id=9),
            generate=Mock(return_value=torch.tensor([[1,2,3,9]])))
        with tempfile.TemporaryDirectory() as folder:
            image=Path(folder)/'core.png';Image.new('RGB',(40,160),'gray').save(image)
            with patch.object(model,'load'),patch('torch.cuda.synchronize'),patch.object(model,'_retry_description',return_value={'ok':True,'fields':valid()}):
                model._answer_photo_batch({'items':[{'id':'a','image':str(image)}]})
        with patch.object(model,'_serial_photo_descriptions',return_value={'items':{}}) as serial,patch.object(model,'load') as load:
            model._answer_photo_batch({'items':[{'id':'x','image':'not needed'}]})
        serial.assert_called_once();load.assert_not_called()

    def test_explicit_description_budget_is_bounded_and_used_for_retry(self):
        self.assertEqual(description_budget({'max_tokens':700},batch=True),700)
        self.assertEqual(description_budget({'max_new_tokens':99999}),1024)
        self.assertEqual(description_budget({'max_tokens':100},retry=True),768)

if __name__=='__main__':unittest.main()
