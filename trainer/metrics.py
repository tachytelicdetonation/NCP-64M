"""
wandb metric computation for NCP pretraining.

Three cadences, driven by train_pretrain.py:
- scalars every --log_interval micro-steps (health floor: losses, lr, grad norm,
  throughput, codebook usage, concept accuracy)
- diagnostics every --diag_interval optimizer steps (update ratios, spectral
  entropy of dW, activation/routing probe, attention entropy, grad noise scale)
- showcase every --showcase_interval (tables/images: generations, codebook
  projector, routing heatmaps, per-chunk concept accuracy, attention maps)

All heavy pieces only run on their cadence; the probe forward reuses one
eval-mode pass to collect activation RMS, routing weights, and attention probs.
"""
import math
import time
from collections import defaultdict, deque

import numpy as np
import torch
import torch.nn.functional as F


def _short_mod(name):
    for k, short in [('token_encoder', 'enc'), ('concept_module', 'cm'),
                     ('token_decoder', 'dec'), ('vanilla', 'van')]:
        if k in name:
            return short
    return 'other'


def _pgroup(name):
    """Optimizer/ownership bucket for a parameter name."""
    if 'embed_tokens' in name or 'lm_head' in name:
        return 'embed'
    if 'quantizer.codebook' in name:
        return 'codebook'
    if 'quantizer.pred_heads' in name:
        return 'pred_heads'
    if 'token_encoder' in name:
        return 'encoder'
    if 'concept_module' in name:
        return 'concept_module'
    if 'token_decoder' in name:
        return 'decoder'
    if 'vanilla' in name:
        return 'vanilla'
    return 'other'


def _is_muon_param(name, p):
    """Same predicate as trainer_utils.build_optimizers."""
    return p.ndim == 2 and 'embed_tokens' not in name and 'lm_head' not in name and 'codebook' not in name


def scalar_metrics(res, lr):
    """Cheap per-log-step scalars from the training forward result."""
    m = {
        'loss/total': res['loss'].item(),
        'loss/ntp': res['loss_ntp'].item(),
        'loss/ppl': math.exp(min(res['loss_ntp'].item(), 20.0)),
        'opt/lr': lr,
        'model/logit_absmax': res['logits'].detach().abs().max().item(),
        'model/hidden_rms': res['hidden_states'].detach().float().pow(2).mean().sqrt().item(),
    }
    if res['logits'].is_cuda:
        m['perf/gpu_mem_gb'] = torch.cuda.max_memory_allocated() / 2**30
    if res.get('loss_ncp') is not None:
        m['loss/ncp'] = res['loss_ncp'].item()
        m['loss/vq'] = res['loss_vq'].item()
    return m


def codebook_metrics(code_idx, pi):
    """VQ usage + concept-prediction quality.

    code_idx: (B, M, S) nearest-codeword indices of encoder-pooled concepts.
    pi:       (B, M, S, N) Concept Module softmax over each codebook; pi[:, j]
              predicts the concept of chunk j+1.

    Returns (scalars, chunk_acc) where chunk_acc is (M-1,) per-chunk-position
    top-1 accuracy, for the showcase table.
    """
    code_idx = code_idx.detach()
    pi = pi.detach().float()
    B, M, S = code_idx.shape
    N = pi.shape[-1]
    m = {}
    pplx, active, dead = [], [], 0
    for s in range(S):
        hist = torch.bincount(code_idx[..., s].flatten(), minlength=N).float()
        p = hist / hist.sum().clamp_min(1)
        nz = p[p > 0]
        ent = -(nz * nz.log()).sum()
        pplx.append(ent.exp().item())
        active.append((hist > 0).float().mean().item())
        dead += int((hist == 0).sum())
        m[f'vq/pplx_s{s}'] = pplx[-1]
    m['vq/pplx_mean'] = float(np.mean(pplx))
    m['vq/active_frac'] = float(np.mean(active))
    m['vq/dead_codes'] = dead

    if M > 1:
        pred = pi[:, :-1].argmax(-1)          # (B, M-1, S)
        tgt = code_idx[:, 1:]                 # (B, M-1, S)
        acc = (pred == tgt).float()
        m['ncp/acc_top1'] = acc.mean().item()
        for s in range(S):
            m[f'ncp/acc_s{s}'] = acc[..., s].mean().item()
        chunk_acc = acc.mean(dim=(0, 2))      # (M-1,)
        lp = pi[:, :-1].clamp_min(1e-9).log()
        m['ncp/pred_entropy_norm'] = (-(pi[:, :-1] * lp).sum(-1).mean() / math.log(N)).item()
        m['ncp/pred_conf'] = pi[:, :-1].max(-1).values.mean().item()
    else:
        chunk_acc = None
    return m, chunk_acc


def grad_group_norms(model):
    """Per-module gradient L2 norms from the current .grad tensors. Called
    between clip and optimizer.step — post-clip (the gradients the optimizer
    actually saw) and pre-Muon, which rewrites p.grad in place during step."""
    sq = defaultdict(float)
    for n, p in model.named_parameters():
        if p.grad is not None:
            sq[_pgroup(n)] += p.grad.detach().float().pow(2).sum().item()
    return {f'grad/{g}': v ** 0.5 for g, v in sq.items()}


def weight_norm(model):
    return math.sqrt(sum(p.detach().float().pow(2).sum().item()
                         for p in model.parameters() if p.requires_grad))


class ActivationMonitor:
    """Forward hooks on NCPBlocks (residual-contribution RMS) and RouteMLPs
    (IRC weights / CRC alphas). Capture is flag-gated: when disabled each hook
    costs one attribute check."""
    def __init__(self, model):
        self.capture = False
        self.block_rms = {}
        self.route_out = {}
        self.handles = []
        for name, mod in model.named_modules():
            cls = type(mod).__name__
            if cls == 'NCPBlock':
                self.handles.append(mod.register_forward_hook(self._block_hook(name)))
            elif cls == 'RouteMLP':
                self.handles.append(mod.register_forward_hook(self._route_hook(name)))

    def _block_hook(self, name):
        def h(module, args, out):
            if self.capture:
                self.block_rms[name] = out.detach().float().pow(2).mean().sqrt().item()
        return h

    def _route_hook(self, name):
        def h(module, args, out):
            if self.capture:
                o = out.detach().float()
                if 'crc_router' in name:   # logits -> softmax alphas
                    o = o.softmax(-1)
                self.route_out[name] = o.mean(dim=(0, 1)).cpu()   # (out_dim,)
        return h

    def scalars(self):
        out = {}
        for name, rms in self.block_rms.items():
            mod = _short_mod(name)
            idx = name.rsplit('.', 1)[-1]
            out[f'act/rms_{mod}{idx}'] = rms
        irc_offdiag, irc_last = defaultdict(list), defaultdict(list)
        crc_last, crc_ent = defaultdict(list), defaultdict(list)
        for name, v in self.route_out.items():
            parts = name.split('.')
            if '.irc.' in name or 'irc' in parts:
                mod = _short_mod(name)
                irc_offdiag[mod].append(v[:-1].sum().item())
                irc_last[mod].append(v[-1].item())
            elif 'crc_router' in name:
                consumer = _short_mod(name)
                src = parts[parts.index('crc_router') + 1]
                nz = v[v > 0]
                crc_ent[f'{consumer}_{src}'].append((-(nz * nz.log()).sum()).item())
                crc_last[f'{consumer}_{src}'].append(v[-1].item())
        for mod, vals in irc_offdiag.items():
            out[f'routing/irc_{mod}_offdiag'] = float(np.mean(vals))
            out[f'routing/irc_{mod}_lastw'] = float(np.mean(irc_last[mod]))
        for key, vals in crc_ent.items():
            out[f'routing/crc_{key}_ent'] = float(np.mean(vals))
            out[f'routing/crc_{key}_last'] = float(np.mean(crc_last[key]))
        return out

    def routing_rows(self):
        """(mod, layer, src, weight) rows for IRC / CRC heatmap tables."""
        irc_rows, crc_rows = [], []
        for name, v in self.route_out.items():
            parts = name.split('.')
            layer = parts[-1]
            if 'irc' in parts:
                for j, w in enumerate(v.tolist()):
                    irc_rows.append([_short_mod(name), int(layer), j, w])
            elif 'crc_router' in parts:
                src = parts[parts.index('crc_router') + 1]
                for j, a in enumerate(v.tolist()):
                    crc_rows.append([_short_mod(name), int(layer), src, j, a])
        return irc_rows, crc_rows

    def close(self):
        for h in self.handles:
            h.remove()


class UpdateTracker:
    """Snapshot-based parameter-update stats: ||dW||/||W|| per group over the
    interval since the last snapshot, per-step update RMS per lr (AdamW ~0.2
    target; Muon Moonlight-scaled likewise — approximate when updates over the
    interval aren't direction-aligned), and spectral entropy of dW (early
    collapse signal, arXiv:2606.28116)."""
    def __init__(self, model, max_svd=16):
        self.prev = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
        self.max_svd = max_svd
        self._last_opt_step = None

    def mark(self, opt_step):
        """Anchor the step counter (e.g. at a resume offset) so the first
        interval is counted correctly."""
        self._last_opt_step = opt_step

    @torch.no_grad()
    def step(self, model, lr, opt_step):
        n_steps = max(1, opt_step - (self._last_opt_step if self._last_opt_step is not None
                                     else opt_step - 1))
        self._last_opt_step = opt_step
        ratios, muon_rms, adam_rms = defaultdict(list), [], []
        deltas_2d = []
        for n, p in model.named_parameters():
            if not p.requires_grad or n not in self.prev:
                continue
            d = p.detach().float() - self.prev[n].float()
            ratios[_pgroup(n)].append((d.norm() / self.prev[n].float().norm().clamp_min(1e-12)).item())
            rms = d.pow(2).mean().sqrt().item()
            (muon_rms if _is_muon_param(n, p) else adam_rms).append(rms)
            if p.ndim == 2 and min(p.shape) >= 8:
                deltas_2d.append(d)
            self.prev[n].copy_(p.detach())
        out = {f'update/ratio_{g}': float(np.mean(v)) / n_steps for g, v in ratios.items()}
        scale = lr * n_steps
        if scale > 0:
            if muon_rms:
                out['update/rms_over_lr_muon'] = float(np.mean(muon_rms)) / scale
            if adam_rms:
                out['update/rms_over_lr_adam'] = float(np.mean(adam_rms)) / scale
        if deltas_2d:
            stride = max(1, len(deltas_2d) // self.max_svd)
            ents = []
            for d in deltas_2d[::stride]:
                s = torch.linalg.svdvals(d.cpu())   # MPS svd falls back/flaky — do it on CPU
                pr = s / s.sum().clamp_min(1e-12)
                ents.append((-(pr * pr.clamp_min(1e-12).log()).sum()).item())
            out['update/dW_spec_ent_mean'] = float(np.mean(ents))
            out['update/dW_spec_ent_min'] = float(np.min(ents))
        return out


def gradient_noise_scale(model, input_ids, labels, autocast_ctx, scaler=None):
    """Two-half-batch GNS estimator (McCandlish et al. 2018): predicts the
    critical batch size. Enters and leaves with zeroed grads; ratio is invariant
    to fp16 loss scaling so no unscale is needed. Costs ~1.5 extra steps."""
    B = input_ids.shape[0]
    if B < 4:
        return {}

    def grad_sq(x, y):
        for p in model.parameters():
            p.grad = None
        with autocast_ctx:
            loss = model(x, labels=y)['loss']
        (scaler.scale(loss) if scaler is not None else loss).backward()
        return sum(p.grad.detach().float().pow(2).sum().item()
                   for p in model.parameters() if p.grad is not None)

    full = grad_sq(input_ids, labels)
    half = (grad_sq(input_ids[:B // 2], labels[:B // 2]) +
            grad_sq(input_ids[B // 2:], labels[B // 2:])) / 2
    for p in model.parameters():
        p.grad = None
    tr_sig = B * (half - full)                    # tr(Sigma)
    g_sq = 2 * full - half                        # ||G||^2
    return {
        'gns/noise_scale': tr_sig / max(g_sq, 1e-12),
        'gns/grad_sq': g_sq,
        'gns/tr_sigma': tr_sig,
    }


@torch.no_grad()
def probe_forward(model, input_ids, actmon):
    """One eval-mode pass with flash off: fills actmon (activation RMS, routing
    weights) and stashes attention probs on each Attention module."""
    attns = [m for m in model.modules() if type(m).__name__ == 'Attention']
    saved = [(a.flash, a.capture_attn) for a in attns]
    for a in attns:
        a.flash, a.capture_attn = False, True
    actmon.capture = True
    was_training = model.training
    model.eval()
    try:
        model(input_ids)
    finally:
        model.train(was_training)
        actmon.capture = False
        for a, (f, c) in zip(attns, saved):
            a.flash, a.capture_attn = f, c
    return attns


def attn_metrics(attns):
    """Per-layer attention entropy (Zhai et al. 2023 collapse signal) + max
    pre-softmax logit (Kimi K2-style divergence signal). Returns (scalars,
    {label: (H,T,T) probs np array} for first/last layer) — probs tensors are
    cleared on read."""
    out, maps, ent_mins, logit_maxs = {}, {}, [], []
    keep = {0, len(attns) - 1}
    for i, a in enumerate(attns):
        p = getattr(a, 'last_attn_probs', None)
        if p is None:
            continue
        ent = -(p * p.clamp_min(1e-9).log()).sum(-1).mean(dim=(0, 2))   # (H,)
        out[f'attn/entropy_l{i}'] = ent.mean().item()
        ent_mins.append(ent.min().item())
        ml = getattr(a, 'last_attn_max_logit', None)
        if ml is not None:
            out[f'attn/max_logit_l{i}'] = ml.item()
            logit_maxs.append(ml.item())
        if i in keep:
            maps[f'l{i}'] = p[0].float().cpu().numpy()                  # (H, T, T)
        a.last_attn_probs = None
        a.last_attn_max_logit = None
    if ent_mins:
        out['attn/entropy_min'] = min(ent_mins)
    if logit_maxs:
        out['attn/max_logit'] = max(logit_maxs)
    return out, maps


@torch.no_grad()
def eval_split(model, ds, eval_idx, batch_size, device, autocast_ctx, max_batches,
               prefix='val/'):
    """Token-weighted val metrics on a fixed index slice."""
    was_training = model.training
    model.eval()
    agg, ntok = defaultdict(float), 0
    try:
        for b in range(min(max_batches, len(eval_idx) // batch_size)):
            idxs = eval_idx[b * batch_size:(b + 1) * batch_size]
            x = torch.stack([ds[i][0] for i in idxs]).to(device)
            y = torch.stack([ds[i][1] for i in idxs]).to(device)
            with autocast_ctx:
                res = model(x, labels=y)
            mask = y != -100
            n = int(mask.sum())
            ntok += n
            agg['loss'] += res['loss'].item() * n
            agg['ntp'] += res['loss_ntp'].item() * n
            agg['acc'] += (res['logits'].argmax(-1)[mask] == y[mask]).float().sum().item()
            if res.get('loss_ncp') is not None:
                agg['ncp'] += res['loss_ncp'].item() * n
                agg['vq'] += res['loss_vq'].item() * n
    finally:
        model.train(was_training)
    out = {f'{prefix}{k}': v / max(ntok, 1) for k, v in agg.items()}
    out[f'{prefix}ppl'] = math.exp(min(out.get(f'{prefix}ntp', 20.0), 20.0))
    return out


class WandbMonitor:
    """Owns the wandb run + all metric cadences. The train loop calls:
    note_micro (every micro-step), on_boundary (every optimizer step, before
    zero_grad), log_scalars (every --log_interval), and finish() at the end.
    Diagnostics/eval/showcase fire inside on_boundary by cadence."""
    GEN_PROMPTS = ['Once upon a time', 'The little girl', 'One day, a cat', 'In a small village']

    def __init__(self, model, tokenizer, optimizers, args, lm_config, autocast_ctx, scaler,
                 val_ds=None, eval_idx=None, extra_evals=None, run_id=None):
        import wandb
        self.wandb = wandb
        self.args = args
        self.model = model
        self.tok = tokenizer
        self.optimizers = optimizers
        self.device = args.device
        self.autocast_ctx = autocast_ctx
        self.scaler = scaler
        self.val_ds = val_ds
        self.eval_idx = eval_idx or []
        self.extra_evals = extra_evals or {}   # name -> (ds, eval_idx)
        self._last_ntp = None
        self.opt_step = 0          # optimizer steps (cadence + alerts)
        self.micro_step = 0        # global micro-step = wandb x-axis
        self.tokens_seen = 0
        self._t_log = time.time()
        self._tok_log = 0
        self.actmon = ActivationMonitor(model)
        self.updates = UpdateTracker(model)
        self.grad_groups = {}
        self.grad_hist = deque(maxlen=200)
        self._clip_events = 0
        self._clip_total = 0
        self._last_alert = -10_000
        self.latest_chunk_acc = None
        self.latest_maps = {}
        self.active = args.wandb_mode != 'disabled'   # heavy cadences skipped when disabled
        n_params = sum(p.numel() for p in model.parameters())
        self.flops_per_token = 6 * n_params
        self.run = wandb.init(
            project=args.wandb_project, name=args.wandb_run_name or None,
            mode=args.wandb_mode, id=run_id, resume='allow' if run_id else None,
            config={**vars(args), 'params_M': n_params / 1e6, **lm_config.to_dict()},
            tags=[args.arch, args.optimizer, f'seed{args.seed}'],
        )
        if args.wandb_watch:
            self.run.watch(model, log='all', log_freq=args.watch_freq)

    def note_micro(self, input_ids, micro_step):
        self.tokens_seen += input_ids.numel()
        self.micro_step = micro_step

    def on_boundary(self, model, grad_norm):
        """Called at each accumulation boundary after clip_grad_norm_, before
        optimizer.step (Muon rewrites p.grad in place, so this must precede it)."""
        self.opt_step += 1
        self.grad_groups = grad_group_norms(model)
        self.grad_hist.append(grad_norm)
        self._clip_total += 1
        if grad_norm > self.args.grad_clip:
            self._clip_events += 1
        if not math.isfinite(grad_norm):
            self._alert('NaN/inf gradient', f'grad_norm={grad_norm} at opt_step {self.opt_step}')
        elif len(self.grad_hist) > 20 and grad_norm > 3 * float(np.median(list(self.grad_hist)[:-1])):
            self._alert('Gradient spike', f'grad_norm={grad_norm:.3f} vs median '
                        f'{float(np.median(list(self.grad_hist)[:-1])):.3f} at opt_step {self.opt_step}')

    def log_scalars(self, res, lr):
        m = scalar_metrics(res, lr)
        self._last_ntp = m.get('loss/ntp')
        m['opt/grad_norm'] = self.grad_hist[-1] if self.grad_hist else 0.0
        m['opt/clip_rate'] = self._clip_events / max(self._clip_total, 1)
        m.update(self.grad_groups)
        now = time.time()
        dt = now - self._t_log
        if dt > 0:
            toks = self.tokens_seen - self._tok_log
            m['perf/tokens_per_sec'] = toks / dt
            m['perf/tflops_est'] = toks / dt * self.flops_per_token / 1e12
            if self.args.peak_flops > 0:
                m['perf/mfu'] = toks / dt * self.flops_per_token / self.args.peak_flops
            self._t_log, self._tok_log = now, self.tokens_seen
        m['perf/tokens_seen'] = self.tokens_seen
        if res.get('code_idx') is not None:
            cm, self.latest_chunk_acc = codebook_metrics(res['code_idx'], res['pi'])
            m.update(cm)
        if not math.isfinite(m['loss/total']):
            self._alert('NaN/inf loss', f"loss={m['loss/total']} at opt_step {self.opt_step}")
        self.run.log(m, step=self.micro_step)

    def on_opt_step(self, input_ids, labels, lr):
        """Cadence-gated heavy metrics; call after optimizer.step + zero_grad."""
        if not self.active:
            return
        a = self.args
        if a.diag_interval > 0 and self.opt_step % a.diag_interval == 0:
            self.run.log(self._diagnostics(input_ids, labels, lr), step=self.micro_step)
        if self.eval_idx and a.eval_interval > 0 and self.opt_step % a.eval_interval == 0:
            vm = eval_split(self.model, self.val_ds, self.eval_idx,
                            min(a.batch_size, 32),  # probes run eager — cap the activation spike
                            self.device, self.autocast_ctx, a.eval_batches)
            if self._last_ntp is not None:
                vm['val/gap_ntp'] = vm['val/ntp'] - self._last_ntp
            for name, (ds, idx) in self.extra_evals.items():
                vm.update(eval_split(self.model, ds, idx, min(a.batch_size, 32),
                                     self.device, self.autocast_ctx,
                                     min(a.eval_batches, 8), prefix=f'val_{name}/'))
            self.run.log(vm, step=self.micro_step)
            for k, v in vm.items():
                best = f'best_{k}'
                prev = self.run.summary.get(best)
                if prev is None or (v > prev if k.endswith('/acc') else v < prev):
                    self.run.summary[best] = v
        if a.showcase_interval > 0 and self.opt_step % a.showcase_interval == 0:
            self.run.log(self._showcase(), step=self.micro_step)

    def _diagnostics(self, input_ids, labels, lr):
        d = self.updates.step(self.model, lr, self.opt_step)
        d['opt/weight_norm'] = weight_norm(self.model)
        probe_x = input_ids[:2]
        attns = probe_forward(self.model, probe_x, self.actmon)
        am, self.latest_maps = attn_metrics(attns)
        d.update(am)
        d.update(self.actmon.scalars())
        if hasattr(self.model.model, 'concept_gain'):
            g = self.model.model.concept_gain.detach().float()
            d['model/concept_gain_mean'] = g.mean().item()
            d['model/concept_gain_rms'] = g.pow(2).mean().sqrt().item()
        for name, mod in self.model.named_modules():
            if type(mod).__name__ == 'TransformerModule':
                for src, scale in mod.crc_scale.items():
                    d[f'routing/crc_scale_{_short_mod(name)}_{src}'] = \
                        scale.detach().float().pow(2).mean().sqrt().item()
        d.update(gradient_noise_scale(self.model, input_ids, labels,
                                      self.autocast_ctx, self.scaler))
        return d

    def _showcase(self):
        w = self.wandb
        out = {}
        rows = []
        was_training = self.model.training
        self.model.eval()
        try:
            # fork_rng: sampling consumes the global RNG that also drives
            # DataLoader shuffling — keep monitored runs comparable to unmonitored
            with torch.random.fork_rng():
                for pr in self.GEN_PROMPTS[:self.args.gen_prompts]:
                    ids = self.tok((self.tok.bos_token or '') + pr, return_tensors='pt').input_ids.to(self.device)
                    gen = self.model.generate(input_ids=ids, max_new_tokens=self.args.gen_tokens,
                                              do_sample=True, eos_token_id=self.tok.eos_token_id,
                                              top_p=0.95, temperature=0.85)
                    rows.append([self.opt_step, pr,
                                 self.tok.decode(gen[0][len(ids[0]):].tolist(), skip_special_tokens=True)])
        finally:
            self.model.train(was_training)
        out['show/generations'] = w.Table(data=rows, columns=['step', 'prompt', 'completion'])
        if self.args.arch == 'ncp':
            cb = self.model.model.quantizer.codebook.detach().float().cpu()
            S, N, D = cb.shape
            out['show/codebook'] = w.Table(
                data=[[s, n] + cb[s, n].tolist() for s in range(S) for n in range(N)],
                columns=['segment', 'code'] + [f'e{i}' for i in range(D)])
        irc_rows, crc_rows = self.actmon.routing_rows()
        if irc_rows:
            out['show/routing_irc'] = w.Table(data=irc_rows,
                                            columns=['module', 'layer', 'src_depth', 'weight'])
        if crc_rows:
            out['show/routing_crc'] = w.Table(data=crc_rows,
                                              columns=['module', 'layer', 'source', 'src_depth', 'alpha'])
        if self.latest_chunk_acc is not None:
            acc = self.latest_chunk_acc.float().cpu()
            out['show/chunk_acc'] = w.Table(
                data=[[i, a] for i, a in enumerate(acc.tolist())], columns=['chunk', 'acc'])
        for tag, layer in [('first', next(iter(self.latest_maps), None)),
                           ('last', next(reversed(self.latest_maps), None))]:
            if layer is None:
                continue
            arr = self.latest_maps[layer][0]          # head 0, (T, T)
            arr = arr / max(arr.max(), 1e-9)
            out[f'show/attn_map_{tag}'] = w.Image((arr * 255).astype(np.uint8))
        return out

    def _alert(self, title, text):
        if self.opt_step - self._last_alert < 200:
            return
        self._last_alert = self.opt_step
        self.run.alert(title=title, text=text, level=self.wandb.AlertLevel.WARN)

    def finish(self):
        self.actmon.close()
        self.run.finish()
