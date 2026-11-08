"""Actual CNN extraction + synthetic head wiring on one real crop. Not geological training/evaluation."""
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))


def main():
    import torch
    from PIL import Image
    from app.infrastructure.local_vision import VisionEngine, default_model_folder
    from app.infrastructure.supervised_learning import activate_head, digest, extract_embedding, write_json
    base = default_model_folder()
    crop = ROOT/'models/vision/pipeline_worker_benchmark_20261104/tape_01.jpg'
    if not crop.is_file(): raise FileNotFoundError(crop)
    engine = VisionEngine(); engine._load(base)
    with Image.open(crop) as image:
        embedding = extract_embedding(engine.model, engine.card['architecture'], image.convert('RGB'), engine.device)
    assert embedding.ndim == 1 and torch.isfinite(embedding).all()
    with TemporaryDirectory(prefix='cv1_head_smoke_') as folder:
        root = Path(folder)
        with patch('app.infrastructure.local_vision.user_data_root', return_value=root):
            request = {'operation':'describe','image':str(crop),'base':str(base)}
            before = engine.describe(request)
            candidate = root/'models/features/demo'; candidate.mkdir(parents=True)
            head = torch.nn.Linear(embedding.numel(), 2)
            with torch.no_grad(): head.weight.zero_(); head.bias.copy_(torch.tensor([6.,0.]))
            torch.save(head.state_dict(),candidate/'head.pt')
            write_json(candidate/'config.json', {'schema':'cv1-feature-head-v1','generative':False,
                'field':'Слоистость','classes':['DEMO — проверка соединения','OTHER'],
                'input_features':embedding.numel(),'base_sha256':engine.card['model_sha256'],
                'head_sha256':digest(candidate/'head.pt'),'deployment':'suggestions_only',
                'validation':{'per_class':[{'name':'DEMO — проверка соединения','support':3,'precision':1.,'recall':1.}]}})
            activate_head(candidate, root/'models/features')
            after = engine.describe(request)
            assert before['predictions'] == after['predictions'], 'CNN outputs changed'
            assert before['fields'] == after['fields'], 'Suggestions must not silently replace descriptions'
            assert after['attribute_suggestions']['Слоистость'] == 'DEMO — проверка соединения'
            assert after['review_required'] is True
    report = {'actual_base':engine.card['architecture'],'feature_dimensions':embedding.numel(),
              'base_predictions_unchanged':True,'suggestions_not_auto_applied':True,
              'real_geological_training':False,'head_is_synthetic_wiring_fixture':True,
              'persistent_user_model_registry_changed':False}
    target = ROOT/'research_reports/description_usability_20261104/actual_cnn_smoke.json'
    write_json(target,report); print(json.dumps(report,ensure_ascii=False))


if __name__ == '__main__': main()
