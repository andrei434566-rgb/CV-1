"""Export only the frozen, source-checked facies TRAIN cards as an offline reference."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'vlm_source/facies_reasoning_v1'
OUTPUT = ROOT / 'app/resources/theory/theory_cards_facies_v1.json'
GROUP = 'facies_reasoning_v1'
INPUTS = {
    'training_reasoned_cards_v3.json': '4b365938146ee247f942c16e5d9a6c2b601da81af35aed02beeb44b19602093f',
    'training_positive_extension_v3.json': '9e987c967267a6b7b27a34f7f917d9741190b1c4f7d6d39ff001fe2944936128',
    'source_registry.json': '34ebb45b0f3766c924a77bce038736c29caa1557cea1dfad5712588a46c5bac9',
}

def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def build(source: Path = SOURCE) -> dict:
    for name, expected in INPUTS.items():
        if digest(source / name) != expected:
            raise ValueError('Unexpected frozen source digest: ' + name)
    registry = json.loads((source / 'source_registry.json').read_text(encoding='utf-8'))['sources']
    sources, cards, ids, questions = {}, [], set(), set()
    for name in list(INPUTS)[:2]:
        for card in json.loads((source / name).read_text(encoding='utf-8'))['cards']:
            if (card.get('split') != 'train' or card.get('source_checked') is not True
                    or card.get('geologist_reviewed') is not False or not card.get('id')
                    or card['id'] in ids or not card.get('question') or card['question'] in questions
                    or not card.get('answer') or not card.get('source_ids')):
                raise ValueError('Invalid or duplicate checked train card')
            ids.add(card['id'])
            questions.add(card['question'])
            names = []
            for identity in card['source_ids']:
                entry = registry[identity]
                if not all(entry.get(k) for k in ('title', 'url', 'license', 'attribution')) or not entry['url'].startswith('https://'):
                    raise ValueError('Missing source attribution: ' + identity)
                names.append('facies_v1_' + identity)
                sources[names[-1]] = {**copy.deepcopy(entry), 'license_url': 'https://creativecommons.org/licenses/by/4.0/'}
            cards.append({
                'id': card['id'], 'question': card['question'], 'answer': card['answer'],
                'sources': names, 'aliases': [], 'holdout': False, 'exact_only': True,
                'metadata': {'origin_manifest': 'vlm_source/' + GROUP + '/' + name,
                             'origin_split': 'train', 'origin_source_group': GROUP,
                             'geology_approved': False, 'reference_scope': 'general_theory',
                             'review_status': 'source_checked_not_geologist_reviewed'},
            })
    if len(cards) != 70 or len(sources) != 5:
        raise ValueError('Expected 70 TRAIN cards and five cited sources')
    return {'schema': 'kern_source_checked_reference_supplement_v1',
            'delivery': 'offline_reference_answers_not_model_weight_training',
            'review_status': 'source_checked_original_russian_paraphrases_not_geologist_reviewed',
            'geologist_reviewed': False,
            'scope': 'Exact TRAIN question lookup only; does not certify neural weights or photo analysis.',
            'input_sha256': dict(INPUTS), 'counts': {'cards': len(cards), 'sources': len(sources), 'validation_cards_included': 0},
            'sources': sources, 'cards': cards}

if __name__ == '__main__':
    content = (json.dumps(build(), ensure_ascii=False, indent=2) + '\n').encode('utf-8')
    OUTPUT.write_bytes(content)
    print('cards=70 sha256=' + hashlib.sha256(content).hexdigest())
