"""Offline cropped-core inference for research v12/v14/v16/v17, never whole-tray segmentation."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--experiment',type=Path,required=True);p.add_argument('--image',type=Path,required=True);p.add_argument('--context-image',type=Path);p.add_argument('--qwen-base',type=Path);p.add_argument('--dino-base',type=Path);p.add_argument('--teacher-head',type=Path);p.add_argument('--output-json',type=Path);a=p.parse_args()
 if not a.image.is_file():raise ValueError('Cropped interval image missing')
 run=a.experiment.resolve();cfg=json.loads((run/'run_config.json').read_text(encoding='utf-8'));contract=json.loads((run/'MODEL_CONTRACT.json').read_text(encoding='utf-8'));version=contract['feature_contract_version']
 allowed=('qwen3vl_dinov2_facies_v1','qwen_dino_residual_facies_v1','qwen_dino_paired_context_facies_v1','qwen_dino_context_margin_facies_v1')
 if version not in allowed:raise ValueError('Supported research types: v12/v14/v16/v17')
 paired=version in allowed[2:]
 if paired and (a.context_image is None or not a.context_image.is_file()):raise ValueError('This model requires a reviewed along-core context image as well as the focal crop')
 from scripts import train_unibo_visual_facies as source
 from scripts.classify_with_qwen_facies_head import streamed_sha
 base=a.qwen_base or Path(cfg['qwen_base']);dino=a.dino_base or Path(cfg['dino']);head=run/('final/residual.safetensors' if version=='qwen_dino_residual_facies_v1' else 'facies_head.safetensors')
 if streamed_sha(head)!=contract['head_sha256'] or streamed_sha(base/'model.safetensors')!=contract['base_weights_sha256'] or streamed_sha(base/'config.json')!=contract['base_config_sha256']:raise ValueError('Exact head/Qwen base required')
 for f,h in contract['preprocessing_files_sha256'].items():
  if source.digest(base/f)!=h:raise ValueError('Qwen processor differs')
 for f,h in cfg['dino_contract']['sha256'].items():
  if source.digest(dino/f)!=h:raise ValueError('DINO model/processor differs')
 if source.package_versions()!=contract['package_versions']:raise ValueError('Exact tested runtime required')
 from scripts.train_qwen_source_pilot import configure_offline_environment,_deny_network
 configure_offline_environment();sys.addaudithook(_deny_network)
 import torch
 from safetensors.torch import load_file
 from PIL import Image
 from transformers import AutoProcessor
 from scripts.vision_facies_adapter import load_original_vision,batch_inputs
 from scripts.train_qwen_visual_facies_head import pool_features
 from scripts.dino_facies_head import extract_features,predict_features
 from scripts.gray_facies_head import unpack_state
 from scripts.facies_head_v2_features import predict
 torch.set_num_threads(4);torch.set_num_interop_threads(1)
 if not torch.cuda.is_available() or torch.cuda.mem_get_info()[0]<2*1024**3:raise RuntimeError('Need2GiB free CUDA')
 state=load_file(str(head))
 if any(not torch.isfinite(v).all() for v in state.values() if v.is_floating_point()):raise ValueError('Nonfinite model tensors')
 processor=AutoProcessor.from_pretrained(base,local_files_only=True);vision=load_original_vision(base)
 with Image.open(a.image) as im:inputs=batch_inputs(processor,[im.convert('RGB')],'cuda')
 with torch.inference_mode():rgb=pool_features(vision(inputs['pixel_values'],grid_thw=inputs['grid'],return_dict=True)).cpu()[None,:]
 del vision,processor,inputs;torch.cuda.empty_cache()
 records=[dict(image=str(a.image.resolve()),image_sha256=source.digest(a.image))]
 if paired:records.append(dict(image=str(a.context_image.resolve()),image_sha256=source.digest(a.context_image)))
 df=extract_features(records,dino)
 if version=='qwen_dino_residual_facies_v1':
  from scripts.residual_facies_head import infer
  teacher_path=a.teacher_head or ROOT/'research_models/web_partial_qwen_facies_head_v3_candidate/facies_head.safetensors'
  if source.digest(teacher_path)!='63a4e09255f2bf03cfd49c9d0849c856c16fcb76ef324994ae31298e9f63ce78':raise ValueError('Exact v3 teacher head required')
  teacher=load_file(str(teacher_path));teacher_scores=predict(rgb,teacher);scores=infer(df[cfg['recipe']['view']][:1],teacher_scores,state,cfg['recipe']);code='ABCDEF'[int(scores.argmax())]
 else:
  teacher=unpack_state(state,'rgb');teacher_scores=predict(rgb,teacher)
  features=torch.cat((df['letterbox'][0],df['letterbox'][1]))[None,:] if paired else df[contract['selected_profile']['view']]
  scores=predict_features(rgb,features,state);code='ABCDEF'[int(scores.argmax())]
  if version=='qwen_dino_context_margin_facies_v1':
   from scripts.margin_facies_gate import predictions
   policy=json.loads((run/'policy.json').read_text(encoding='utf-8'))
   if source.digest(run/'policy.json')!=contract['policy_sha256'] or policy['head_sha256']!=contract['head_sha256']:raise ValueError('Exact policy required')
   code=predictions(teacher_scores,scores,policy['threshold'])[0]
 result=dict(source_code=code,source_facies_ru=contract['code_to_russian'][code],scores_not_probabilities=dict(zip('ABCDEF',scores[0].tolist())),
  teacher_code='ABCDEF'[int(teacher_scores.argmax())],context_used=paired,research_only=True,segmentation_or_description=False,
  acceptance_status=contract['acceptance_status'],head_sha256=contract['head_sha256'],language_model_loaded=False,pretrained_encoder_weights_changed=False,project_92_facies_mapping=None)
 if a.output_json:source.write_json(a.output_json,result)
 sys.stdout.reconfigure(encoding='utf-8');print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
