"""Offline vision-only inference for v8 RBF or v9 visual-LoRA research heads.

No language model or network; this is six-class recognition, not a description.
"""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--experiment',type=Path,required=True);p.add_argument('--base',type=Path,default=ROOT.parent/'vlm_core_v1/model_cache/hub/models--Qwen--Qwen3-VL-2B-Instruct/snapshots/89644892e4d85e24eaac8bacfd4f463576704203');p.add_argument('--image',type=Path,required=True);p.add_argument('--output-json',type=Path);a=p.parse_args()
 if not a.image.is_file():raise ValueError('Input image does not exist')
 from scripts.classify_with_qwen_facies_head import streamed_sha
 from scripts import train_unibo_visual_facies as source
 c=json.loads((a.experiment/'MODEL_CONTRACT.json').read_text(encoding='utf-8'));version=c.get('feature_contract_version')
 if version not in ('qwen3vl_rbf_partial_head_v1','qwen3vl_vision_lora_facies_v1'):raise ValueError('Unsupported research contract')
 path=a.experiment/('facies_head.safetensors' if version=='qwen3vl_rbf_partial_head_v1' else 'final/adapter.safetensors')
 if streamed_sha(path)!=c['head_sha256'] or streamed_sha(a.base/'model.safetensors')!=c['base_weights_sha256'] or streamed_sha(a.base/'config.json')!=c['base_config_sha256']:raise ValueError('Exact base/head hashes required')
 for name,h in c['preprocessing_files_sha256'].items():
  if Path(name).name!=name or streamed_sha(a.base/name)!=h:raise ValueError('Processor contract differs')
 from scripts.train_qwen_source_pilot import configure_offline_environment,_deny_network
 configure_offline_environment();sys.addaudithook(_deny_network)
 import torch
 from safetensors.torch import load_file
 from transformers import AutoProcessor
 from PIL import Image
 from scripts.vision_facies_adapter import load_original_vision,add_lora,restore_adapter,batch_inputs,post_features,normalized_post
 torch.set_num_threads(4);torch.set_num_interop_threads(1)
 if not torch.cuda.is_available() or torch.cuda.mem_get_info()[0]<2*1024**3:raise RuntimeError('Need2GiB free CUDA')
 state=load_file(str(path));vision=load_original_vision(a.base);processor=AutoProcessor.from_pretrained(a.base,local_files_only=True)
 if version=='qwen3vl_vision_lora_facies_v1':
  recipe=c['training_recipe'];add_lora(vision,recipe['rank'],recipe['lora_alpha'],recipe['last_vision_blocks']);restore_adapter(vision,state)
 with Image.open(a.image) as opened:image=opened.convert('RGB')
 with torch.inference_mode():
  feature=post_features(vision,batch_inputs(processor,[image],'cuda'))
  if version=='qwen3vl_rbf_partial_head_v1':
   from scripts.kernel_facies_head import validate_state,predict_features
   validate_state(state,c);raw=torch.cat((torch.zeros(1,2048),feature.cpu()),1);scores=predict_features(raw,state)[0]
  else:
   normalizer={k:state[k].cuda() for k in ('mean','scale')};coeff=(state['head_base']+state['head_delta']).cuda();scores=(normalized_post(feature,normalizer)@coeff)[0].cpu()
 code='ABCDEF'[int(scores.argmax())]
 result=dict(source_code=code,source_facies_ru=c['code_to_russian'][code],scores_not_probabilities=dict(zip('ABCDEF',scores.tolist())),
  research_only=True,general_description=False,language_model_loaded=False,visual_adapter_loaded=version=='qwen3vl_vision_lora_facies_v1',
  acceptance_status=c['acceptance_status'],head_sha256=c['head_sha256'],project_92_facies_mapping=None)
 if a.output_json:source.write_json(a.output_json,result)
 sys.stdout.reconfigure(encoding='utf-8');print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
