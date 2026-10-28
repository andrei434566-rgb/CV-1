"""Freeze sourced facies reasoning TRAIN examples and separate evaluation."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.infrastructure.geology_reference import format_grounded_question
from scripts.prepare_qwen_theory_round2 import check_partitions, normalized_question

SOURCE = ROOT / 'vlm_source/facies_reasoning_v1'
OLD = ROOT / 'open_datasets/qwen_geology_sft_v2_20261027'
OUT = ROOT / 'open_datasets/qwen_facies_reasoning_v1_v2'


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8-sig').splitlines() if line.strip()]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def record(card, question, *, training, mode, sources):
    return {'conversations': [{'from': 'human', 'value': question}, {'from': 'gpt', 'value': card['answer']}],
            'metadata': {'record_id': card['id'] + '__' + mode, 'record_type': 'text_only_knowledge',
                         'source_group': 'facies_reasoning_v1', 'sources': sources,
                         'training_approved': training, 'geology_approved': False, 'target_language': 'ru',
                         'split': 'train' if training else 'validation', 'topic': card['topic'],
                         'original_question': card['question'], 'evaluation_mode': mode,
                         'review_status': 'source_checked_assistant_not_geologist_reviewed',
                         'rubric': {'required_facts': card.get('required_facts', []),
                                    'forbidden_facts': card.get('forbidden_facts', [])},
                         'scope': 'Textual reasoning; no photo label or proven geological environment'}}


def main():
    if OUT.exists() and any(OUT.iterdir()):
        raise FileExistsError('Refusing to overwrite frozen data')
    cards_path, registry_path, eval_path = (SOURCE / name for name in ('training_reasoned_cards_v3.json', 'source_registry.json', 'evaluation.json'))
    extension_path = SOURCE / 'training_positive_extension_v3.json'
    audit_path = SOURCE / 'evaluation_source_audit.json'
    cards = json.loads(cards_path.read_text(encoding='utf-8-sig'))['cards']
    extension = json.loads(extension_path.read_text(encoding='utf-8-sig'))['cards']
    if len(cards) != 50 or len(extension) != 20:
        raise ValueError('Expected 50 checked cards plus 20 distinct positive reasoning examples')
    cards += extension
    registry = json.loads(registry_path.read_text(encoding='utf-8-sig'))['sources']
    exam_document = json.loads(eval_path.read_text(encoding='utf-8-sig'))
    audit = json.loads(audit_path.read_text(encoding='utf-8-sig'))
    if audit.get('frozen_evaluation_sha256') != sha(eval_path) or audit.get('geologist_reviewed') is not False:
        raise ValueError('Evaluation source audit does not match frozen questions')
    audit_by_id = {item['id']: item for item in audit['items']}
    if exam_document.get('record_role') != 'evaluation_only_never_train' or exam_document.get('geologist_reviewed') is not False:
        raise ValueError('Missing evaluation-only / expert review status')
    exam = exam_document['items']
    if len(exam) != 12 or len(cards) < 25:
        raise ValueError('Unexpected number of sourced cards or evaluation questions')
    if any(item.get('split') != 'validation' or item.get('geologist_reviewed') is not False for item in exam):
        raise ValueError('Every new scenario must be evaluation-only and not marked expert-approved')
    if set(audit_by_id) != {item['id'] for item in exam}:
        raise ValueError('Every evaluation scenario needs a separate source audit')
    previous = rows(OLD / 'mixed.jsonl')
    old_eval = rows(OLD / 'holdout_theory.jsonl')
    dialogue = rows(OLD / 'holdout_conversation.jsonl')
    images = rows(OLD / 'val_visual.jsonl')
    if len(previous) != 344 or len(dialogue) != 4 or len(images) != 8:
        raise ValueError('Missing earlier regression pools')
    protected = {normalized_question(item['question']) for item in exam}
    if len(protected) != len(exam):
        raise ValueError('Duplicate evaluation question')
    historical = sorted(OLD.parent.glob('qwen*/holdout*.jsonl'))
    for path in historical:
        for row in rows(path):
            if not row.get('image') and row.get('conversations'):
                protected.add(normalized_question(row.get('metadata', {}).get('original_question', row['conversations'][0]['value'])))
    previous = [copy.deepcopy(row) for row in previous if normalized_question(row.get('metadata', {}).get('original_question', row['conversations'][0]['value'])) not in protected]
    previous_questions = {normalized_question(row.get('metadata', {}).get('original_question', row['conversations'][0]['value'])) for row in previous}
    training, omitted, seen = previous, [], set()
    for card in cards:
        if card.get('split') != 'train' or card.get('source_checked') is not True or card.get('geologist_reviewed') is not False:
            raise ValueError('Only source-checked TRAIN cards with explicit expert status allowed')
        key = normalized_question(card['question'])
        if key in protected or key in previous_questions:
            omitted.append(card['id'])
            continue
        if key in seen or not card.get('support_text') or not card.get('answer') or not card.get('source_ids'):
            raise ValueError('Duplicate or incomplete training card: ' + card['id'])
        seen.add(key)
        sources = []
        for identity in card['source_ids']:
            info = registry[identity]
            if not str(info.get('url', '')).startswith('https://') or not info.get('license'):
                raise ValueError('Source URL and reuse license required: ' + identity)
            sources.append(dict(info, source_id=identity))
        training.append(record(card, card['question'], training=True, mode='closed_book_train', sources=sources))
        prompt = format_grounded_question(card['question'], [{'id': card['id'], 'text': card['support_text']}])
        training.append(record(card, prompt, training=True, mode='context_train', sources=sources))
    theory = []
    for item in exam:
        source_audit = audit_by_id[item['id']]
        direct_sources = [{'url': url, 'source_id': url, 'access': 'direct_source_audit'}
                          for url in source_audit.get('source_urls', [])]
        row = record(item, item['question'], training=False, mode='new_facies_reasoning', sources=direct_sources)
        row['metadata']['formal_scoring_recommendation'] = source_audit['formal_scoring_recommendation']
        row['metadata']['source_audit_limitations'] = source_audit.get('limitations', [])
        theory.append(row)
    old_closed = [copy.deepcopy(row) for row in old_eval if row['metadata'].get('evaluation_mode') == 'closed_book_regression']
    if len(old_closed) != 4:
        raise ValueError('Expected four fixed old geology guards')
    theory.extend(old_closed)
    parts = check_partitions(training, [*theory, *dialogue])
    raw_train = {normalized_question(row.get('metadata', {}).get('original_question', row['conversations'][0]['value'])) for row in training}
    if raw_train & {normalized_question(item['question']) for item in exam}:
        raise ValueError('Train/evaluation question overlap')
    OUT.mkdir(parents=True, exist_ok=True)
    exports = {'mixed.jsonl': training, 'holdout_theory.jsonl': theory,
               'holdout_conversation.jsonl': dialogue, 'val_visual.jsonl': images}
    for name, items in exports.items():
        (OUT / name).write_text(''.join(json.dumps(item, ensure_ascii=False) + '\n' for item in items), encoding='utf-8', newline='\n')
    (OUT / 'system_prompt.txt').write_bytes((OLD / 'system_prompt.txt').read_bytes())
    protocol = {'schema': 'qwen_facies_reasoning_v1', 'status': 'frozen_before_training',
                'training_rows': len(training), 'new_train_cards': len(seen), 'positive_extension_cards': len(extension), 'omitted_ids': omitted,
                'new_scenarios': len(exam),
                'excluded_from_formal_scoring': [item['id'] for item in exam if audit_by_id[item['id']]['formal_scoring_recommendation'].startswith('exclude')],
                'old_closed_guards': len(old_closed),
                'dialogue_guards': len(dialogue), 'image_guards': len(images), 'partitions': parts,
                'input_sha256': {p.name: sha(p) for p in (cards_path, extension_path, SOURCE / 'training_positive_extension_v2.json', SOURCE / 'training_reasoned_cards_v2.json', SOURCE / 'training_positive_extension.json', SOURCE / 'training_reasoned_cards.json', SOURCE / 'training_cards.json', registry_path, eval_path, audit_path)},
                'manifest_sha256': {name: sha(OUT / name) for name in [*exports, 'system_prompt.txt']},
                'recipe': {'base': 'fresh Qwen3-VL-2B-Instruct snapshot 89644892e4d85e24eaac8bacfd4f463576704203',
                           'epochs': 2, 'learning_rate': 0.0001, 'targets': 'all-text-linear',
                           'rank': 8, 'alpha': 16, 'seed': 20261028, 'max_tokens': 1024,
                           'theory_max_new_tokens': 256, 'classification_max_new_tokens': 24},
                'prior_untrained_dataset': 'qwen_facies_reasoning_v1 was assembled before three source-review corrections and was not used for training.',
                'limitations': ['Examples are assistant-authored and source checked, not geologist approved.',
                                'New evaluation shares concepts with training, not independent well data.',
                                'A rock name alone cannot prove a facies or environment.',
                                'Text-only training does not establish visual or facies recognition accuracy.',
                                'No automatic activation of candidate weights.']}
    (OUT / 'trial_protocol.json').write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + '\n', encoding='utf-8', newline='\n')
    print(json.dumps({key: protocol[key] for key in ('training_rows', 'new_train_cards', 'new_scenarios', 'omitted_ids', 'partitions')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
