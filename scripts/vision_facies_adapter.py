"""Vision-only loading and small LoRA/classification adapter; no language model."""
import math


def load_original_vision(base,device='cuda'):
    import json
    from pathlib import Path
    import torch
    from safetensors import safe_open
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLVisionConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel
    base=Path(base);config=Qwen3VLVisionConfig(**json.loads((base/'config.json').read_text(encoding='utf-8'))['vision_config'])
    config._attn_implementation='sdpa'
    model=Qwen3VLVisionModel(config)
    # HF loading keeps generated rotary buffers in FP32. A blanket .half()
    # would round frequencies and silently change the original visual features.
    buffers={n:b.clone() for n,b in model.named_buffers() if b.is_floating_point()}
    model.half()
    for n,b in buffers.items():
        path,_,name=n.rpartition('.');setattr(model.get_submodule(path),name,b)
    with safe_open(base/'model.safetensors',framework='pt',device='cpu') as f:
        weights={k.removeprefix('model.visual.'):f.get_tensor(k) for k in f.keys() if k.startswith('model.visual.')}
    model.load_state_dict(weights,strict=True);del weights
    model.to(device);model.eval();model.requires_grad_(False)
    return model


def add_lora(vision,rank=8,alpha=16,last_blocks=4):
    import torch
    from torch import nn
    import torch.nn.functional as functional
    class LoRALinear(nn.Module):
        def __init__(self,base):
            super().__init__();self.base=base;self.scale=alpha/rank
            self.lora_a=nn.Parameter(torch.empty(rank,base.in_features,device=base.weight.device,dtype=torch.float32))
            self.lora_b=nn.Parameter(torch.zeros(base.out_features,rank,device=base.weight.device,dtype=torch.float32))
            nn.init.kaiming_uniform_(self.lora_a,a=math.sqrt(5));self.base.requires_grad_(False)
        def forward(self,x):
            update=functional.linear(functional.linear(x.float(),self.lora_a),self.lora_b)
            return self.base(x)+update.to(x.dtype)*self.scale
    names=[]
    for i in range(len(vision.blocks)-last_blocks,len(vision.blocks)):
        for name in ('qkv','proj'):
            module=vision.blocks[i].attn;setattr(module,name,LoRALinear(getattr(module,name)));names.append(f'blocks.{i}.attn.{name}')
    return names


def batch_inputs(processor,images,device):
    import torch
    resized=[]
    for im in images:
        im=im.copy();im.thumbnail((300,300));resized.append(im)
    inputs=processor.image_processor(images=resized,return_tensors='pt')
    return dict(pixel_values=inputs['pixel_values'].to(device=device,dtype=torch.float16),grid=inputs['image_grid_thw'].to(device))


def post_features(vision,inputs):
    import torch
    out=vision(inputs['pixel_values'],grid_thw=inputs['grid'],return_dict=True).pooler_output.float()
    lengths=(inputs['grid'].prod(-1)//4).tolist();chunks=torch.split(out,lengths)
    return torch.stack([torch.cat((x.mean(0),x.std(0,unbiased=False))) for x in chunks])


def normalized_post(features,normalizer):
    import torch
    z=(features.float()-normalizer['mean'])/normalizer['scale'];z=z/z.norm(dim=1,keepdim=True).clamp_min(1e-12)
    return torch.cat((z,torch.ones((len(z),1),device=z.device,dtype=z.dtype)),1)


def label_loss(logits,allowed,weights):
    import torch
    mask=torch.zeros_like(logits,dtype=torch.bool)
    for i,codes in enumerate(allowed):
        for c in codes:mask[i,'ABCDEF'.index(c)]=True
    logp=torch.log_softmax(logits.float(),1)
    return (-torch.logsumexp(logp.masked_fill(~mask,-torch.inf),1)*weights).mean()


def adapter_state(vision,head_delta,normalizer,base_coefficients):
    state={n:p.detach().float().cpu().contiguous() for n,p in vision.named_parameters() if p.requires_grad}
    state.update(head_delta=head_delta.detach().cpu().contiguous(),head_base=base_coefficients.detach().cpu().contiguous(),
        mean=normalizer['mean'].detach().cpu().contiguous(),scale=normalizer['scale'].detach().cpu().contiguous())
    return state


def restore_adapter(vision,state):
    import torch
    params=dict(vision.named_parameters());names=[n for n,p in params.items() if p.requires_grad]
    if set(names)!={k for k in state if 'lora_' in k}:raise ValueError('Adapter module names differ')
    with torch.no_grad():
        for n in names:
            if params[n].shape!=state[n].shape:raise ValueError('Adapter dimensions differ')
            params[n].copy_(state[n].to(params[n].device))
