"""
Training utilities: logging, lr schedule, checkpointing, model init, Muon optimizer.
"""
import os
import sys
__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import random
import math
import numpy as np
import torch
import torch.distributed as dist
from transformers import AutoTokenizer
from model.model_ncp import NCPConfig, NCPForCausalLM

def get_model_params(model, config):
    total = sum(p.numel() for p in model.parameters()) / 1e6
    Logger(f'Model Params: {total:.2f}M')

def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0

def Logger(content):
    if is_main_process():
        print(content)

def get_lr(current_step, total_steps, lr):
    return lr * (0.1 + 0.45 * (1 + math.cos(math.pi * current_step / total_steps)))

def setup_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

@torch.no_grad()
def zeropower_via_newtonschulz5(G, steps=5):
    """Newton-Schulz iteration to orthogonalize G (Muon update direction)."""
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    transposed = G.size(-2) > G.size(-1)
    if transposed: X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X
    return X.mT if transposed else X

class Muon(torch.optim.Optimizer):
    """Muon (Moonlight flavor): orthogonalized momentum update for 2D matrix params.
    W <- W - lr * (muon_scale * sqrt(max(d_in, d_out)) * O + weight_decay * W)
    matching Eq. 25 of the paper."""
    def __init__(self, params, lr=6e-5, momentum=0.95, nesterov=True, ns_steps=5,
                 muon_scale=0.2, weight_decay=0.0):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps,
                        muon_scale=muon_scale, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(g)
                buf = state['momentum_buffer']
                buf.lerp_(g, 1 - group['momentum'])
                g = g.lerp_(buf, group['momentum']) if group['nesterov'] else buf
                u = zeropower_via_newtonschulz5(g, group['ns_steps'])
                scale = group['muon_scale'] * math.sqrt(max(p.size(0), p.size(1)))
                p.mul_(1 - group['lr'] * group['weight_decay'])
                p.add_(u.type_as(p), alpha=-group['lr'] * scale)

def build_optimizers(model, args):
    """Paper recipe: Muon for matrix-valued params, AdamW for embeddings/other."""
    muon_params, adam_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim == 2 and 'embed_tokens' not in name and 'lm_head' not in name \
                and 'codebook' not in name:
            muon_params.append(p)
        else:
            adam_params.append(p)
    if args.optimizer == 'muon':
        opts = [Muon(muon_params, lr=args.learning_rate, weight_decay=args.weight_decay),
                torch.optim.AdamW(adam_params, lr=args.learning_rate, betas=(0.9, 0.95), weight_decay=args.weight_decay)]
        Logger(f'Optimizer: Muon ({sum(p.numel() for p in muon_params) / 1e6:.2f}M params) + AdamW ({sum(p.numel() for p in adam_params) / 1e6:.2f}M params)')
    else:
        opts = [torch.optim.AdamW(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.95), weight_decay=args.weight_decay)]
        Logger('Optimizer: AdamW')
    return opts

def lm_checkpoint(lm_config, weight='pretrain', model=None, optimizers=None, epoch=0, step=0, save_dir='../checkpoints', **kwargs):
    os.makedirs(save_dir, exist_ok=True)
    suffix = f'_{weight}_{lm_config.hidden_size}'
    ckp_path = f'{save_dir}/{suffix}.pth'
    resume_path = f'{save_dir}/{suffix}_resume.pth'
    if model is not None:
        state_dict = {k: v.half().cpu() for k, v in model.state_dict().items()}
        torch.save(state_dict, ckp_path + '.tmp'); os.replace(ckp_path + '.tmp', ckp_path)
        resume_data = {'model': state_dict, 'epoch': epoch, 'step': step,
                       'optimizers': [o.state_dict() for o in optimizers] if optimizers else None}
        torch.save(resume_data, resume_path + '.tmp'); os.replace(resume_path + '.tmp', resume_path)
    else:
        if os.path.exists(resume_path):
            return torch.load(resume_path, map_location='cpu', weights_only=False)
        return None

def init_model(lm_config, from_weight='none', tokenizer_path=None, save_dir='./out', device='cpu'):
    tokenizer_path = tokenizer_path or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'model')
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model = NCPForCausalLM(lm_config)
    if from_weight != 'none':
        weight_path = f'{save_dir}/{from_weight}_{lm_config.hidden_size}.pth'
        if os.path.exists(weight_path):
            model.load_state_dict(torch.load(weight_path, map_location=device), strict=False)
            Logger(f'Loaded weights from {weight_path}')
    get_model_params(model, lm_config)
    Logger(f'Trainable Params: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f}M')
    return model.to(device), tokenizer
