"""Offline research prediction for an already cropped core interval.

Requires the EXACT original HF base and the separately trained head. Does
not load PEFT, change application settings, or provide a full description.
"""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import train_unibo_visual_facies as source
from scripts.train_qwen_visual_facies_head import pool_features,predict_head

V1='qwen3vl_pre_post_merger_mean_std_v1'
V2='qwen3vl_visual_texture_groupcv_v2'


def validate_head_state(state,contract):
    """Reject incompatible state before loading the large CUDA model."""
    import torch
    version=contract.get('feature_contract_version'); d=contract['feature_dimension']
    if version not in (V1,V2): raise ValueError('Unsupported feature contract')
    if (state['mean'].shape!=(d,) or state['scale'].shape!=(d,)
        or state['coefficients'].shape!=(d+1,6)): raise ValueError('Head shape differs')
    if any(not torch.isfinite(state[k]).all() for k in ('mean','scale','coefficients')) or (state['scale']<=0).any():
        raise ValueError('Invalid normalizer or coefficients')
    if version==V2:
        from scripts.facies_head_v2_features import profile_spec,TEXTURE_VERSION
        if contract['texture_version']!=TEXTURE_VERSION: raise ValueError('Unsupported texture version')
        qdim=contract['qwen_feature_dimension']; pre_dim=contract['pre_merger_feature_dimension']
        columns,sizes,weights=profile_spec(contract['selected_profile']['profile'],qdim,pre_dim,contract['texture_dimension'])
        if (state['columns'].dtype!=torch.int64 or state['block_sizes'].dtype!=torch.int64
            or not torch.equal(state['columns'],columns) or not torch.equal(state['block_sizes'],sizes)
            or not torch.equal(state['block_weights'],weights) or len(columns)!=d):
            raise ValueError('Feature profile differs from contract')


def score_image_features(qwen_features,image,state,contract):
    """Original RGB crop is used for texture, before Qwen thumbnailing."""
    if contract['feature_contract_version']==V1: return predict_head(qwen_features,state)
    import torch
    from scripts.facies_head_v2_features import texture_features,predict
    if qwen_features.shape!=(1,contract['qwen_feature_dimension']): raise ValueError('Qwen feature width differs')
    texture=torch.from_numpy(texture_features(image))[None,:]
    if texture.shape[1]!=contract['texture_dimension']: raise ValueError('Texture feature width differs')
    return predict(torch.cat((qwen_features,texture),1),state)


def streamed_sha(path):
    h=sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024**2),b''): h.update(block)
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir',type=Path,default=ROOT/'research_models/unibo_qwen_facies_head_v1')
    parser.add_argument('--base',type=Path,default=ROOT.parent/'vlm_core_v1/model_cache/hub/models--Qwen--Qwen3-VL-2B-Instruct/snapshots/89644892e4d85e24eaac8bacfd4f463576704203')
    parser.add_argument('--core-crop',type=Path,required=True)
    parser.add_argument('--output-json',type=Path)
    args=parser.parse_args()
    contract=json.loads((args.model_dir/'MODEL_CONTRACT.json').read_text(encoding='utf-8'))
    if contract.get('feature_contract_version') not in (V1,V2): raise ValueError('Unsupported feature contract')
    if streamed_sha(args.model_dir/'facies_head.safetensors')!=contract['head_sha256']: raise ValueError('Head checksum differs')
    if (streamed_sha(args.base/'config.json')!=contract['base_config_sha256']
        or streamed_sha(args.base/'model.safetensors')!=contract['base_weights_sha256']):
        raise ValueError('The exact original Qwen base is required; quantized or adapted weights are incompatible with this verified contract')
    for name,expected in contract.get('preprocessing_files_sha256',{}).items():
        if Path(name).name!=name or streamed_sha(args.base/name)!=expected: raise ValueError('Processor file differs: '+name)
    from scripts.train_qwen_source_pilot import configure_offline_environment,_deny_network
    configure_offline_environment(); sys.addaudithook(_deny_network)
    import torch
    from PIL import Image
    from safetensors.torch import load_file
    from transformers import AutoProcessor,Qwen3VLForConditionalGeneration
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    if not torch.cuda.is_available(): raise RuntimeError('This research CLI requires CUDA')
    if torch.cuda.mem_get_info()[0]<5*1024**3:
        raise RuntimeError('The separate research classifier needs at least5GiB free CUDA memory')
    state=load_file(str(args.model_dir/'facies_head.safetensors'))
    validate_head_state(state,contract)
    processor=AutoProcessor.from_pretrained(args.base,local_files_only=True)
    model=Qwen3VLForConditionalGeneration.from_pretrained(args.base,local_files_only=True,dtype=torch.float16,attn_implementation='sdpa',device_map='cuda')
    model.eval(); model.requires_grad_(False)
    with Image.open(args.core_crop) as opened:
        original=opened.convert('RGB'); image=original.copy(); image.thumbnail((300,300))
    inputs=source.prompt_inputs(processor,image,source.PROMPT).to('cuda')
    with torch.inference_mode():
        vision=model.model.visual(inputs['pixel_values'].to(model.dtype),grid_thw=inputs['image_grid_thw'],return_dict=True)
        feature=pool_features(vision).cpu()[None,:]
    scores=score_image_features(feature,original,state,contract)[0]
    code=source.CODES[int(scores.argmax())]
    result={'source_code':code,'source_facies_ru':contract['code_to_russian'][code],
        'scores_not_probabilities':dict(zip(source.CODES,scores.tolist())),
        'research_only':True,'scope':contract['input_scope'],'project_92_facies_mapping':None,
        'base_revision':contract['base_revision'],'head_sha256':contract['head_sha256'],
        'acceptance_status':contract.get('acceptance_status','see acceptance_review.json')}
    if args.output_json:
        args.output_json.parent.mkdir(parents=True,exist_ok=True)
        source.write_json(args.output_json,result)
    sys.stdout.reconfigure(encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__': main()
