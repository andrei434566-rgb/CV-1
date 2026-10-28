"""Offline exact TRAIN reference tests; no neural model is imported."""
import hashlib
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from app.infrastructure import local_theory_lookup as lookup
from scripts.export_facies_reasoning_v1_reference import build, INPUTS

ROOT = Path(__file__).resolve().parent
RES = ROOT / 'app/resources/theory'
BASE = RES / 'theory_cards_v5.json'
SUPPLEMENT = RES / 'theory_cards_facies_v1.json'
SOURCE = ROOT / 'vlm_source/facies_reasoning_v1'
GROUP = 'facies_reasoning_v1'


class FaciesOfflineReferenceTests(unittest.TestCase):
    def setUp(self):
        lookup._load_cards.cache_clear()
        self.addCleanup(lookup._load_cards.cache_clear)
        self.data = json.loads(SUPPLEMENT.read_text(encoding='utf-8'))
        self.train = [c for name in list(INPUTS)[:2] for c in json.loads((SOURCE / name).read_text(encoding='utf-8'))['cards']]

    def previous_corpus(self):
        corpus = json.loads(BASE.read_text(encoding='utf-8'))
        for name, group in (('theory_cards_round2.json', 'theory_sources_20261025'),
                            ('theory_cards_round4.json', 'theory_sources_20261027'),
                            ('theory_cards_sft_v2.json', 'geology_sft_v2_20261027')):
            corpus = lookup._merge_reference_supplement(corpus, json.loads((RES / name).read_text(encoding='utf-8')), expected_source_group=group)
        return corpus

    def test_reproducible_train_only_exact_reference(self):
        self.assertEqual(self.data, build())
        self.assertEqual(self.data['input_sha256'], INPUTS)
        self.assertEqual(self.data['counts'], {'cards': 70, 'sources': 5, 'validation_cards_included': 0})
        self.assertEqual(len(self.train), 70)
        for c, old in zip(self.data['cards'], self.train):
            self.assertEqual((c['id'], c['question'], c['answer']), (old['id'], old['question'], old['answer']))
            self.assertTrue(c['exact_only'])
            self.assertEqual(c['aliases'], [])
            self.assertIs(c['holdout'], False)
            self.assertEqual(c['metadata']['origin_split'], 'train')
            self.assertEqual(c['metadata']['origin_source_group'], GROUP)
            self.assertEqual(c['metadata']['reference_scope'], 'general_theory')
            self.assertIs(c['metadata']['geology_approved'], False)
            self.assertTrue(all(identity.startswith('facies_v1_') for identity in c['sources']))
        self.assertTrue(all(s['license'] and s['attribution'] for s in self.data['sources'].values()))
        self.assertNotIn(b'\r\n', SUPPLEMENT.read_bytes())
        self.assertEqual(len(hashlib.sha256(SUPPLEMENT.read_bytes()).hexdigest()), 64)

    def test_all_70_exact_answers(self):
        for c in self.train:
            with self.subTest(identity=c['id']):
                self.assertEqual(lookup.lookup_theory_answer(c['question'], BASE), c['answer'])

    def test_near_misses_do_not_match_new_reference(self):
        isolated = lookup._merge_reference_supplement({'cards': [], 'sources': {}}, self.data, expected_source_group=GROUP)
        self.assertEqual(len(isolated['cards']), 70)
        with patch.object(lookup, '_load_cards', return_value=isolated):
            for c in self.train:
                with self.subTest(identity=c['id']):
                    self.assertIsNone(lookup.lookup_theory_answer(c['question'] + ' Слоистость отсутствует.', BASE))
                    self.assertIsNone(lookup.lookup_theory_answer('Объясни подробнее: ' + c['question'], BASE))

    def test_sources_only_when_requested(self):
        for c in self.train:
            with self.subTest(identity=c['id']):
                ordinary = lookup.lookup_theory_answer(c['question'], BASE)
                self.assertEqual(ordinary, c['answer'])
                self.assertNotRegex(ordinary, r'https?://')
                cited = lookup.lookup_theory_answer(c['question'] + ' Дай источники.', BASE, include_sources=True)
                self.assertTrue(cited.startswith(c['answer'] + '\n\nИсточники:\n'))
                reference = next(item for item in self.data['cards'] if item['id'] == c['id'])
                for identity in reference['sources']:
                    self.assertIn(self.data['sources'][identity]['url'], cited)

    def test_frozen_evaluation_is_not_delivered_or_changed(self):
        exam = json.loads((SOURCE / 'evaluation.json').read_text(encoding='utf-8'))['items']
        self.assertEqual(len(exam), 12)
        previous = self.previous_corpus()
        resource = SUPPLEMENT.read_text(encoding='utf-8')
        for item in exam:
            with self.subTest(identity=item['id']):
                with patch.object(lookup, '_load_cards', return_value=previous):
                    before = lookup.lookup_theory_answer(item['question'], BASE)
                self.assertEqual(lookup.lookup_theory_answer(item['question'], BASE), before)
                self.assertNotIn(item['id'], resource)
                self.assertNotIn(item['question'], resource)
                if item.get('answer'):
                    self.assertNotIn(item['answer'], resource)


if __name__ == '__main__':
    unittest.main()
