"""Publish audited numbers and contracts only; never images, anchors or weights."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kernel', type=Path, required=True)
    parser.add_argument('--visual', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() or not output.is_relative_to(ROOT / 'research_reports'):
        raise ValueError('Choose a new research report folder')
    metrics, audits, contracts, receipts = {}, {}, {}, {}
    for name, run in (('kernel_v8', args.kernel), ('visual_lora_v9', args.visual)):
        result, audit, config, contract = [read(run / f) for f in
            ('result.json', 'audit_result.json', 'run_config.json', 'MODEL_CONTRACT.json')]
        if audit['head_sha256' if name == 'kernel_v8' else 'adapter_sha256'] != result['head_sha256']:
            raise ValueError('Audited weights differ')
        if audit['status'] != result['status'] or audit['validation_gates'] != result['validation_gates']:
            raise ValueError('Audit status/gates differ')
        weight = run / ('facies_head.safetensors' if name == 'kernel_v8' else 'final/adapter.safetensors')
        if digest(weight) != result['head_sha256']:
            raise ValueError('Local weights changed after audit')
        metrics[name] = {k: v for k, v in result.items() if k != 'training_receipts'}
        audits[name] = audit
        # Whitelist the actual typed contract. Legacy v7 selected_profile fields
        # are irrelevant for visual LoRA and are deliberately not re-exported.
        fields = ('type', 'base_model_id', 'base_revision', 'base_config_sha256',
                  'base_weights_sha256', 'base_weights_bytes', 'head_sha256',
                  'feature_contract_version', 'feature_dimension', 'feature_spec',
                  'code_to_russian', 'package_versions', 'preprocessing_files_sha256',
                  'training_recipe', 'training_source_license', 'acceptance_status',
                  'general_chat_or_description_model', 'automatic_project_catalog_mapping',
                  'distribution_approved', 'input_scope')
        contracts[name] = {k: contract[k] for k in fields if k in contract}
        receipts[name] = dict(recipe=config['recipe'], train_count=config['train_count'],
            validation_count=config.get('validation_count', config.get('val_count')),
            source_rights=config['source_rights'], original_base_sha256=contract['base_weights_sha256'],
            adapter_or_head_sha256=result['head_sha256'],
            selection_frozen_sha256=result['selection_frozen_sha256'],
            dependencies_sha256={str(Path(k).name): v for k, v in config['dependencies'].items()},
            distribution_approved=False, installed_in_application=False,
            prepared_test_pixels_opened=0)
        if name == 'visual_lora_v9':
            receipts[name]['folds'] = [{k: r[k] for k in
                ('tag', 'seed', 'teacher_protected_exact_rows', 'epochs', 'modules',
                 'trainable_vision_parameters', 'trainable_head_parameters',
                 'first_gradient_norms', 'adapter_sha256', 'seconds')}
                | dict(train_count=len(r['training_ids']), held_count=len(r['held_ids']),
                       normalizer_old_train_count=len(r['normalizer_training_ids']))
                for r in result['training_receipts']]
            receipts[name]['staged_process_exits'] = [json.loads(line) for line in
                run.with_suffix('.stages.jsonl').read_text(encoding='utf-8').splitlines()]
    output.mkdir(parents=True)
    write(output / 'research_metrics.json', metrics)
    write(output / 'audit_result.json', audits)
    write(output / 'model_contracts.json', contracts)
    write(output / 'training_receipt.json', receipts)
    preflight = args.visual.parent / 'vision_facies_adapter_v9_preflight.json'
    write(output / 'visual_preflight.json', read(preflight))
    if (args.visual / 'batching_check.json').is_file():
        write(output / 'batching_check.json', read(args.visual / 'batching_check.json'))
    summary = {}
    for name, result in metrics.items():
        summary[name] = dict(status=result['status'], old_oof_correct=result['old_oof']['correct'],
            old_oof_total=result['old_oof']['total'], old_oof_macro_f1=result['old_oof']['macro_f1'],
            val_correct=result['val']['correct'], val_total=result['val']['total'],
            val_macro_f1=result['val']['macro_f1'],
            failed_validation_gates=[k for k, v in result['validation_gates'].items() if not v])
    write(output / 'summary.json', summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
