"""Measure FP16 single-image vs four-image inference, without changing weights."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    from scripts import train_unibo_visual_facies as source
    from scripts.classify_with_qwen_facies_head import streamed_sha
    cfg = json.loads((run / 'run_config.json').read_text(encoding='utf-8'))
    contract = json.loads((run / 'MODEL_CONTRACT.json').read_text(encoding='utf-8'))
    audit = json.loads((run / 'audit_result.json').read_text(encoding='utf-8'))
    base = Path(cfg['base'])
    path = run / 'final/adapter.safetensors'
    if streamed_sha(path) != audit['adapter_sha256'] or streamed_sha(base / 'model.safetensors') != contract['base_weights_sha256']:
        raise ValueError('Audited weights changed')
    from scripts.train_qwen_source_pilot import configure_offline_environment, _deny_network
    configure_offline_environment()
    sys.addaudithook(_deny_network)
    import torch
    from safetensors.torch import load_file
    from transformers import AutoProcessor
    from PIL import Image
    from scripts.vision_facies_adapter import load_original_vision, add_lora, restore_adapter, batch_inputs, post_features, normalized_post
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    rows = source.read_rows(run / 'val.jsonl')
    images = []
    for row in rows:
        if source.digest(row['image']) != row['metadata']['image_sha256']:
            raise ValueError('Validation pixels changed')
        with Image.open(row['image']) as image:
            images.append(image.convert('RGB'))
    state = load_file(str(path))
    model = load_original_vision(base)
    recipe = cfg['recipe']
    add_lora(model, recipe['rank'], recipe['lora_alpha'], recipe['last_vision_blocks'])
    restore_adapter(model, state)
    processor = AutoProcessor.from_pretrained(base, local_files_only=True)
    norm = {key: state[key].cuda() for key in ('mean', 'scale')}
    coeff = (state['head_base'] + state['head_delta']).cuda()

    def scores(batch_size):
        parts = []
        with torch.inference_mode():
            for offset in range(0, len(images), batch_size):
                features = post_features(model, batch_inputs(processor, images[offset:offset + batch_size], 'cuda'))
                parts.append((normalized_post(features, norm) @ coeff).cpu())
        return torch.cat(parts)

    batch, single = scores(4), scores(1)
    saved = json.loads((run / 'val_answers.json').read_text(encoding='utf-8'))
    batch_delta = float((batch - torch.tensor([row['scores'] for row in saved])).abs().max())
    if batch_delta > 1e-5:
        raise ValueError('Same-batch replay differs from audit')
    bp, sp = batch.argmax(1), single.argmax(1)
    expected = torch.tensor(['ABCDEF'.index(row['conversations'][1]['value']) for row in rows])
    changed = [dict(record_id=row['metadata']['record_id'], batch_code='ABCDEF'[int(b)], single_code='ABCDEF'[int(s)])
               for row, b, s in zip(rows, bp, sp) if b != s]
    result = dict(adapter_sha256=audit['adapter_sha256'], validation_count=len(rows),
                  same_batch_max_score_delta_to_saved=batch_delta,
                  maximum_single_vs_batch_score_delta=float((single - batch).abs().max()),
                  changed_predictions=changed, single_correct=int((sp == expected).sum()),
                  batch_correct=int((bp == expected).sum()),
                  limitation='FP16 batch shapes can change raw uncalibrated scores. This is the reused val50, not a new test set.')
    source.write_json(run / 'batching_check.json', result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
