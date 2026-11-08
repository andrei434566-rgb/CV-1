import copy,hashlib,json,unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from PIL import Image
from app.infrastructure.bedding_expansion import TYPES,known_targets,masked_balanced_loss,promotion_gate,validate_examples

class PartialBeddingTests(unittest.TestCase):
    def test_unmentioned_types_remain_unknown(self):
        y,m=known_targets([{'positive_type':TYPES[0]}])
        self.assertEqual(float(y.sum()),1);self.assertEqual(float(m.sum()),1)
        self.assertEqual(float(m[0,1]),0)

    def test_known_negative_requires_source_annotation(self):
        valid={'negative_origin':'author_explicit_massive_structure','source':'rf100',
               'label':2,'label_origin':'author_yolo_bbox'}
        y,m=known_targets([valid]);self.assertEqual(float(y.sum()),0);self.assertEqual(float(m.sum()),len(TYPES)-1)
        self.assertEqual(float(m[0,TYPES.index('Градационная')]),0)
        for key,value in [('label',0),('source','guessed'),('label_origin','caption_keyword'),('positive_type',TYPES[0])]:
            with self.assertRaises(ValueError):known_targets([dict(valid,**{key:value})])

    def test_unknown_logit_has_no_training_gradient(self):
        import torch
        pos=[{'positive_type':name} for name in TYPES]
        neg={'negative_origin':'author_explicit_massive_structure','source':'rf100','label':2,'label_origin':'author_yolo_bbox'}
        y,m=known_targets(pos+[neg]);logits=torch.zeros_like(y,requires_grad=True)
        masked_balanced_loss(logits,y,m).backward()
        self.assertEqual(float(logits.grad[0,1]),0)
        self.assertLess(float(logits.grad[0,0]),0);self.assertGreater(float(logits.grad[-1,0]),0)

    def test_massive_cannot_train_a_negative_grading_classifier(self):
        import torch
        neg={'negative_origin':'author_explicit_massive_structure','source':'rf100','label':2,'label_origin':'author_yolo_bbox'}
        y,m=known_targets([{'positive_type':name} for name in TYPES]+[neg])
        logits=torch.zeros_like(y,requires_grad=True)
        masked_balanced_loss(logits,y,m).backward()
        self.assertEqual(float(logits.grad[:,TYPES.index('Градационная')].abs().sum()),0)

    def test_no_automatic_promotion_from_tiny_train_loss(self):
        rows=[{'positive_type':n,'split':'train','source_group':'one_core'} for n in TYPES]
        result=promotion_gate(rows)
        self.assertFalse(result['eligible_for_automatic_assignment']);self.assertFalse(result['all_16_covered'])
        self.assertTrue(any('16' in r for r in result['reasons']))
        self.assertTrue(any('ухудшения' in r for r in result['reasons']))

    def test_same_core_and_exact_crop_cannot_cross_split(self):
        with TemporaryDirectory() as d:
            files=[]
            for i,color in enumerate(['gray','brown']):
                p=Path(d)/f'{i}.png';Image.new('RGB',(40,80),color).save(p);files.append(p)
            def row(i,split,group):
                return {'id':str(i),'image':str(files[i]),'sha256':hashlib.sha256(files[i].read_bytes()).hexdigest(),
                        'positive_type':TYPES[0],'split':split,'source_group':group,'source':'author','well':'well1',
                        'domain':'photographic_core','research_training_approved':split=='train',
                        'label_provenance':'caption_and_spatially_matched_author_label'}
            a,b=row(0,'train','core'),row(1,'test','core')
            with self.assertRaisesRegex(ValueError,'group crosses'):validate_examples([a,b])
            b['source_group']='another_group'
            with self.assertRaisesRegex(ValueError,'Well crosses'):validate_examples([a,b])
            b['well']='well2';self.assertEqual(len(validate_examples([a,b])),2)
            b.update(image=a['image'],sha256=a['sha256'])
            with self.assertRaisesRegex(ValueError,'Duplicate photographic'):validate_examples([a,b])

    def test_sketches_and_changed_files_are_rejected(self):
        with TemporaryDirectory() as d:
            p=Path(d)/'im.png';Image.new('RGB',(40,40),'gray').save(p)
            row={'id':'x','image':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'positive_type':TYPES[1],
                 'split':'train','domain':'diagram','source_group':'g','source':'author','research_training_approved':True,
                 'label_provenance':'caption_and_spatially_matched_author_label'}
            with self.assertRaises(ValueError):validate_examples([row])
            row['domain']='photographic_core';row['sha256']='0'*64
            with self.assertRaisesRegex(ValueError,'Changed crop'):validate_examples([row])

if __name__=='__main__':unittest.main()
