"""CPU tests for source/focal alignment and adding visual adapters safely."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

from PIL import Image
from scripts import train_unibo_visual_facies as source
from scripts.prepare_unibo_context import context_panels,blank_context,CANVAS_SIZE,TARGET_BOX,CONTEXT_PROMPT,FOCAL_SIDE
from scripts.prepare_unibo_vlm_facies import pixel_digest
from scripts.train_unibo_context_vision import select_vision_targets,check_loaded_state,validate_context
from test_unibo_visual_facies_training import source_row


class ContextViewTests(unittest.TestCase):
    def test_right_panel_is_exact_resampling_of_the_labelled_crop(self):
        photo=Image.new('RGB',(900,600),'gray')
        focal=Image.new('RGB',(128,128),(37,65,93))
        panel,geometry=context_panels(photo,focal,(500,200,628,328))
        expected=focal.resize((FOCAL_SIDE,FOCAL_SIDE),Image.Resampling.LANCZOS)
        self.assertEqual(panel.size,CANVAS_SIZE)
        self.assertEqual(panel.crop(TARGET_BOX).tobytes(),expected.tobytes())
        self.assertEqual(geometry['target_panel_box'],list(TARGET_BOX))
        self.assertEqual(focal.getpixel((0,0)),(37,65,93))

    def test_context_ablation_preserves_every_focal_pixel_and_original_input(self):
        photo=Image.new('RGB',(800,800),'white'); focal=Image.new('RGB',(128,128),'brown')
        panel,_=context_panels(photo,focal,(300,300,428,428))
        before=panel.tobytes(); blank=blank_context(panel)
        self.assertEqual(panel.tobytes(),before)
        self.assertEqual(blank.crop(TARGET_BOX).tobytes(),panel.crop(TARGET_BOX).tobytes())
        self.assertNotEqual(blank.tobytes(),before)

    def test_small_photo_is_padded_without_invalid_focal_coordinates(self):
        photo=Image.new('RGB',(240,180),'white'); focal=photo.crop((50,30,178,158))
        panel,geometry=context_panels(photo,focal,(50,30,178,158))
        self.assertEqual(panel.size,CANVAS_SIZE)
        self.assertEqual(geometry['context_padding_xy'],[136,166])
        for bad in ((-1,0,127,128),(200,100,328,228),(1,1,130,130)):
            with self.assertRaises(ValueError): context_panels(photo,focal,bad)

    def test_only_last_four_complete_visual_attention_blocks_are_selected(self):
        names=[f'model.visual.blocks.{i}.attn.{part}' for i in range(24) for part in ('qkv','proj')]
        names += ['model.language_model.layers.0.self_attn.q_proj','model.visual.patch_embed.proj']
        selected=select_vision_targets(names)
        self.assertEqual(len(selected),8)
        self.assertTrue(all(any(f'blocks.{i}.' in name for i in (20,21,22,23)) for name in selected))
        self.assertFalse(any('language_model' in name for name in selected))
        with self.assertRaises(ValueError): select_vision_targets([n for n in names if n!='model.visual.blocks.23.attn.proj'])
        with self.assertRaises(ValueError): select_vision_targets(['model.visual.blocks.0.attn.qkv'])

    def test_state_loading_refuses_missing_language_weights_or_unexpected_keys(self):
        targets=['model.visual.blocks.23.attn.qkv']
        missing=[f'base_model.model.{targets[0]}.lora_{part}.default.weight' for part in ('A','B')]
        check_loaded_state(missing,[],targets)
        with self.assertRaises(ValueError): check_loaded_state(missing+['language.lora_A.weight'],[],targets)
        with self.assertRaises(ValueError): check_loaded_state(missing,['unexpected.weight'],targets)

    def fixture(self,root):
        result={}
        for split in ('train','val'):
            old=[]; context=[]
            for i in range(6):
                raw=source_row(i,split)
                focal=Image.new('RGB',(128,128),(i*20+10,35 if split=='train' else 80,55))
                path=root/f'{split}_{i}.png'; focal.save(path)
                raw['image']=str(path); raw['metadata']['image_sha256']=source.digest(path)
                previous=source.transform_facies(raw); old.append(previous)
                panel,_=context_panels(Image.new('RGB',(700,700),(120,120,120)),focal,(200,200,328,328))
                target=root/f'context_{split}_{i}.png'; panel.save(target)
                row=deepcopy(previous); row['image']=str(target); row['conversations'][0]['value']='<image>\n'+CONTEXT_PROMPT
                row['metadata'].update({'focal_record_id':previous['metadata']['record_id'],'focal_image_sha256':previous['metadata']['image_sha256'],
                    'image_sha256':source.digest(target),'pixel_sha256':pixel_digest(panel),'source_mask_passed_to_model':False,
                    'class_text_drawn_on_input':False,'quality_scope':'focal_crop_only_not_context'})
                context.append(row)
            result[split]=(old,context)
        return result

    def test_actual_panel_pixels_and_unchanged_source_labels_validate(self):
        with tempfile.TemporaryDirectory() as temp:
            fixture=self.fixture(Path(temp))
            validate_context(fixture['train'][1],fixture['val'][1],fixture['train'][0],fixture['val'][0])

    def test_rehashed_panel_tampering_cannot_change_the_target_region(self):
        with tempfile.TemporaryDirectory() as temp:
            f=self.fixture(Path(temp)); row=f['train'][1][0]
            with Image.open(row['image']) as im: panel=im.convert('RGB')
            panel.putpixel((TARGET_BOX[0]+2,TARGET_BOX[1]+2),(255,0,0)); panel.save(row['image'])
            row['metadata']['image_sha256']=source.digest(row['image']); row['metadata']['pixel_sha256']=pixel_digest(panel)
            with self.assertRaisesRegex(ValueError,'focal panel'): validate_context(f['train'][1],f['val'][1],f['train'][0],f['val'][0])

    def test_label_change_or_mask_input_is_refused(self):
        with tempfile.TemporaryDirectory() as temp:
            f=self.fixture(Path(temp)); row=f['train'][1][0]; row['conversations'][1]['value']='B'
            with self.assertRaisesRegex(ValueError,'target changed'): validate_context(f['train'][1],f['val'][1],f['train'][0],f['val'][0])
            row['conversations'][1]['value']='A'; row['metadata']['source_mask_passed_to_model']=True
            with self.assertRaisesRegex(ValueError,'provenance'): validate_context(f['train'][1],f['val'][1],f['train'][0],f['val'][0])

    def test_same_source_group_cannot_cross_new_context_splits(self):
        with tempfile.TemporaryDirectory() as temp:
            f=self.fixture(Path(temp))
            f['val'][0][0]['metadata']['source_group']='S_train'
            f['val'][1][0]['metadata']['source_group']='S_train'
            with self.assertRaisesRegex(ValueError,'leakage'): validate_context(f['train'][1],f['val'][1],f['train'][0],f['val'][0])


if __name__=='__main__': unittest.main()
