"""Verify vision-only loading, exact preprocessing and real trainable gradients."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def main():
 from scripts import train_unibo_visual_facies as source
 from scripts.vision_facies_adapter import load_original_vision,add_lora,batch_inputs,post_features,normalized_post,label_loss
 from scripts.train_qwen_source_pilot import configure_offline_environment,_deny_network
 configure_offline_environment();sys.addaudithook(_deny_network)
 import torch,time
 from transformers import AutoProcessor
 from safetensors.torch import load_file
 from PIL import Image
 torch.set_num_threads(4);torch.manual_seed(20261031)
 run=ROOT.parent/'vlm_core_v1/outputs/core_qwen_core_contrasts_20261031_v7';cfg=json.loads((run/'run_config.json').read_text(encoding='utf-8'));ref=Path(cfg['reference']);base=Path(cfg['base'])
 rows=source.read_rows(run/'train.jsonl');cache=load_file(str(run/'features.safetensors'))['train'];state=load_file(str(ref/'facies_head.safetensors'))
 processor=AutoProcessor.from_pretrained(base,local_files_only=True);vision=load_original_vision(base)
 images=[]
 for r in rows[:4]:
  with Image.open(r['image']) as im:images.append(im.convert('RGB'))
 values=[]
 for i,image in enumerate(images[:2]):
  inp=batch_inputs(processor,[image],'cuda')
  reference_image=image.copy();reference_image.thumbnail((300,300));old=source.prompt_inputs(processor,reference_image,source.PROMPT)
  if not torch.equal(inp['pixel_values'].cpu(),old['pixel_values'].half()):raise ValueError('Image preprocessing differs')
  with torch.inference_mode():features=post_features(vision,inp).cpu()
  delta=float((features-cache[i:i+1,2048:]).abs().max())
  if delta>1e-5:raise ValueError('Vision-only original features differ: '+str(delta))
  values.append(delta)
 modules=add_lora(vision);normalizer={k:state[k].float().cuda() for k in ('mean','scale')};coeff=state['coefficients'].float().cuda()
 parameters=[p for p in vision.parameters() if p.requires_grad];optimizer=torch.optim.AdamW(parameters,lr=1e-4)
 before={n:p.detach().clone() for n,p in vision.named_parameters() if p.requires_grad};start=time.perf_counter()
 inp=batch_inputs(processor,images,'cuda')
 with torch.autocast('cuda',dtype=torch.float16):features=post_features(vision,inp)
 scores=normalized_post(features,normalizer)@coeff;loss=label_loss(scores*5,[r['allowed_codes'] for r in rows[:4]],torch.ones(4,device='cuda'))
 loss.backward();grads={n:float(p.grad.norm()) for n,p in vision.named_parameters() if p.requires_grad and p.grad is not None and n.endswith('lora_b')}
 if not grads or not all(torch.isfinite(p.grad).all() for p in parameters if p.grad is not None) or max(grads.values())<=0:raise ValueError('Missing/nonfinite LoRA gradients')
 optimizer.step();torch.cuda.synchronize()
 changed=sum(not torch.equal(before[n],p) for n,p in vision.named_parameters() if p.requires_grad)
 if changed==0:raise ValueError('Optimizer did not change LoRA weights')
 result=dict(status='passed',original_feature_max_deltas=values,preprocessing_equal=True,real_gradient_norms=grads,changed_parameter_tensors=changed,trainable_parameters=sum(p.numel() for p in parameters),modules=modules,
  one_batch_seconds=time.perf_counter()-start,peak_cuda_bytes=torch.cuda.max_memory_allocated(),language_model_loaded=False,language_weights_changed=False)
 out=ROOT.parent/'vlm_core_v1/outputs/vision_facies_adapter_v9_preflight.json';source.write_json(out,result);print(json.dumps(result,indent=2))


if __name__=='__main__':main()
