"""Bounded learned corrections to a frozen teacher, not a language-model adapter."""


def make_network(hidden=64):
    from torch import nn
    net=nn.Sequential(nn.Linear(1158,hidden),nn.GELU(),nn.Dropout(.1),nn.Linear(hidden,6))
    nn.init.zeros_(net[-1].weight);nn.init.zeros_(net[-1].bias)
    return net


def score(features,teacher,network,cap=.15):
    import torch
    return teacher+cap*torch.tanh(network(torch.cat((features,teacher),1)))


def fit(dino_raw,teacher_scores,rows,old_flags,seed,recipe,device='cuda'):
    import torch
    from scripts.kernel_facies_head import grouped_weights
    from scripts.vision_facies_adapter import label_loss
    from scripts.facies_head_v2_features import normalize
    import math
    torch.manual_seed(seed)
    z=dino_raw.double();norm=dict(columns=torch.arange(1152),block_sizes=torch.tensor([384]*3),block_weights=torch.full((3,),1/math.sqrt(3),dtype=torch.float64),mean=z.mean(0),scale=z.std(0,unbiased=False).clamp_min(1e-3))
    features=normalize(dino_raw,norm)[:,:-1].float().to(device);teacher=teacher_scores.float().to(device)
    network=make_network(recipe['hidden']).to(device);weights=grouped_weights(rows,'class_group',.5).float().to(device)
    protect=torch.tensor([old and len(r['allowed_codes'])==1 and 'ABCDEF'[p]==r['allowed_codes'][0] for old,r,p in zip(old_flags,rows,teacher.argmax(1).tolist())],device=device)
    teacher_prob=torch.softmax(teacher*recipe['logit_scale'],1).detach();optimizer=torch.optim.AdamW(network.parameters(),lr=recipe['lr'],weight_decay=recipe['weight_decay']);losses=[];first_grad=None
    for epoch in range(recipe['epochs']):
        generator=torch.Generator().manual_seed(seed+epoch);order=torch.randperm(len(rows),generator=generator).tolist();total=0
        network.train()
        for offset in range(0,len(order),recipe['batch_size']):
            ids=order[offset:offset+recipe['batch_size']];optimizer.zero_grad(set_to_none=True)
            logits=score(features[ids],teacher[ids],network,recipe['cap'])*recipe['logit_scale']
            loss=label_loss(logits,[rows[i]['allowed_codes'] for i in ids],weights[ids]);keep=protect[ids]
            if keep.any():loss=loss+recipe['teacher_KL']*torch.nn.functional.kl_div(torch.log_softmax(logits[keep],1),teacher_prob[ids][keep],reduction='batchmean')
            if not torch.isfinite(loss):raise ValueError('Nonfinite residual loss')
            loss.backward()
            if first_grad is None:first_grad=float(network[-1].weight.grad.norm())
            torch.nn.utils.clip_grad_norm_(network.parameters(),1.);optimizer.step();total+=float(loss.detach())*len(ids)
        losses.append(total/len(rows))
    network.eval();state={'network__'+k:v.detach().cpu().contiguous() for k,v in network.state_dict().items()}
    state.update({'norm__'+k:v for k,v in norm.items()})
    if first_grad is None or first_grad<=0 or float(state['network__3.weight'].abs().max())<=0:raise ValueError('Residual head did not learn')
    receipt=dict(seed=seed,epochs=recipe['epochs'],epoch_mean_losses=losses,first_output_gradient_norm=first_grad,protected_old_exact_count=int(protect.sum()),trainable_parameters=sum(p.numel() for p in network.parameters()),train_count=len(rows))
    return state,receipt


def infer(dino_raw,teacher_scores,state,recipe):
    import torch
    from scripts.facies_head_v2_features import normalize
    network=make_network(recipe['hidden']);network.load_state_dict({k.removeprefix('network__'):v for k,v in state.items() if k.startswith('network__')},strict=True);network.eval()
    norm={k.removeprefix('norm__'):v for k,v in state.items() if k.startswith('norm__')}
    with torch.inference_mode():return score(normalize(dino_raw,norm)[:,:-1].float(),teacher_scores.float(),network,recipe['cap'])
