import argparse
import torch
import pytest
from contextlib import nullcontext
from model.model_ncp import NCPConfig, NCPForCausalLM
from trainer import metrics as tmet

VOCAB, T, K = 6400, 64, 4


def make_model(**kw):
    cfg = NCPConfig(max_position_embeddings=T, concept_chunk=K, **kw)
    torch.manual_seed(0)
    return NCPForCausalLM(cfg).eval()


def ids(B=2):
    g = torch.Generator().manual_seed(0)
    return torch.randint(0, VOCAB, (B, T), generator=g)


def xy(B=2):
    """Dataset-convention (input, labels) pair: pre-shifted, buf[:-1]/buf[1:]."""
    g = torch.Generator().manual_seed(0)
    buf = torch.randint(0, VOCAB, (B, T + 1), generator=g)
    return buf[:, :-1], buf[:, 1:]


def test_forward_returns_code_idx_and_pi():
    model = make_model()
    out = model(ids())
    B, M, S, N = 2, T // K, 6, 128
    assert out['code_idx'].shape == (B, M, S)
    assert out['pi'].shape == (B, M, S, N)
    assert (out['code_idx'] >= 0).all() and (out['code_idx'] < N).all()


def test_forward_aux_none_for_vanilla():
    model = make_model(arch='vanilla')
    x, y = xy()
    out = model(x, labels=y)
    assert out['code_idx'] is None and out['pi'] is None
    assert out['loss_ncp'] is None


def test_codebook_metrics_perfect_prediction():
    B, M, S, N = 2, 8, 6, 128
    g = torch.Generator().manual_seed(0)
    idx = torch.randint(0, N, (B, M, S), generator=g)
    pi = torch.full((B, M, S, N), -10.0)
    pi[:, :-1].scatter_(-1, idx[:, 1:].unsqueeze(-1), 10.0)
    pi = pi.softmax(-1)
    m, chunk_acc = tmet.codebook_metrics(idx, pi)
    assert m['ncp/acc_top1'] == pytest.approx(1.0)
    assert chunk_acc.shape == (M - 1,)
    assert chunk_acc.mean().item() == pytest.approx(1.0)
    assert m['ncp/pred_entropy_norm'] < 0.1


def test_codebook_metrics_uniform_usage():
    B, M, S, N = 64, 2, 6, 128
    idx = torch.arange(B * M).view(B, M, 1).expand(B, M, S) % N
    pi = torch.full((B, M, S, N), 1.0 / N)
    m, _ = tmet.codebook_metrics(idx, pi)
    for s in range(S):
        assert m[f'vq/pplx_s{s}'] == pytest.approx(N, rel=1e-4)
    assert m['vq/active_frac'] == pytest.approx(1.0)
    assert m['vq/dead_codes'] == 0


def test_codebook_metrics_dead_codes():
    B, M, S, N = 4, 4, 6, 128
    idx = torch.zeros(B, M, S, dtype=torch.long)   # only code 0 ever used
    pi = torch.full((B, M, S, N), 1.0 / N)
    m, _ = tmet.codebook_metrics(idx, pi)
    assert m['vq/dead_codes'] == S * (N - 1)
    assert m['vq/pplx_mean'] == pytest.approx(1.0, rel=1e-4)


def test_grad_group_norms():
    model = make_model()
    model.train()
    x, y = xy()
    res = model(x, labels=y)
    res['loss'].backward()
    norms = tmet.grad_group_norms(model)
    for grp in ('encoder', 'concept_module', 'decoder', 'codebook', 'pred_heads'):
        assert f'grad/{grp}' in norms and norms[f'grad/{grp}'] > 0


def test_update_tracker_ratio_and_reset():
    model = make_model()
    tracker = tmet.UpdateTracker(model)
    with torch.no_grad():
        model.model.embed_tokens.weight.mul_(1.1)
    out = tracker.step(model, lr=1e-3, opt_step=1)
    assert out['update/ratio_embed'] == pytest.approx(0.1, rel=1e-3)
    assert 'update/rms_over_lr_muon' in out and 'update/rms_over_lr_adam' in out
    out = tracker.step(model, lr=1e-3, opt_step=2)
    assert out['update/ratio_embed'] == pytest.approx(0.0)


def test_update_tracker_normalizes_by_interval():
    model = make_model()
    tracker = tmet.UpdateTracker(model)
    tracker.step(model, lr=1e-3, opt_step=1)
    p = model.model.embed_tokens.weight
    with torch.no_grad():
        p.add_(torch.full_like(p, 0.01))
    one = tracker.step(model, lr=1e-3, opt_step=2)['update/ratio_embed']
    with torch.no_grad():
        p.add_(torch.full_like(p, 0.01))
    five = tracker.step(model, lr=1e-3, opt_step=7)['update/ratio_embed']
    assert five < one          # same absolute drift spread over 5 steps


def test_probe_forward_collects_act_routing_attn():
    model = make_model()
    mon = tmet.ActivationMonitor(model)
    attns = tmet.probe_forward(model, ids(1), mon)
    assert mon.block_rms, 'no activation RMS captured'
    assert any('irc' in k for k in mon.route_out), 'no IRC weights captured'
    assert any('crc_router' in k for k in mon.route_out), 'no CRC alphas captured'
    m, maps = tmet.attn_metrics(attns)
    assert 'attn/entropy_min' in m and m['attn/entropy_min'] >= 0
    assert 'attn/max_logit' in m
    assert maps, 'no attention maps captured'
    s = mon.scalars()
    assert any(k.startswith('routing/irc_') for k in s)
    assert any(k.startswith('routing/crc_') for k in s)
    mon.close()


def test_attention_capture_flag_restored():
    model = make_model()
    attns = [m for m in model.modules() if type(m).__name__ == 'Attention']
    expected_flash = [a.flash for a in attns]
    tmet.probe_forward(model, ids(1), tmet.ActivationMonitor(model))
    for a, f in zip(attns, expected_flash):
        assert a.flash == f
        assert a.capture_attn is False


def test_gradient_noise_scale_runs():
    model = make_model()
    model.train()
    x, y = xy(4)
    out = tmet.gradient_noise_scale(model, x, y, nullcontext())
    assert 'gns/noise_scale' in out
    assert all(p.grad is None for p in model.parameters()), 'GNS must leave grads zeroed'


def _monitor_args(**kw):
    d = dict(device='cpu', wandb_project='test', wandb_run_name='', wandb_mode='disabled',
             wandb_watch=0, watch_freq=500, diag_interval=1, eval_interval=1,
             eval_batches=0, showcase_interval=1, gen_tokens=8, gen_prompts=0,
             peak_flops=0.0, grad_clip=1.0, arch='ncp', optimizer='adamw', seed=42,
             accumulation_steps=1, batch_size=4, max_seq_len=T)
    d.update(kw)
    return argparse.Namespace(**d)


def test_wandb_monitor_smoke_disabled_mode():
    """Full monitor path (scalars + diagnostics + showcase) with wandb disabled —
    exercises every computation without network/auth."""
    model = make_model()
    model.train()
    cfg = model.config
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    mon = tmet.WandbMonitor(model, tokenizer=None, optimizers=[opt], args=_monitor_args(),
                            lm_config=cfg, autocast_ctx=nullcontext(), scaler=None)
    mon.active = True   # 'disabled' mode skips heavy cadences; force them for coverage
    x, y = xy(4)
    for i in range(2):
        mon.note_micro(x, i + 1)
        res = model(x, labels=y)
        res['loss'].backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        mon.on_boundary(model, norm.item())
        opt.step()
        opt.zero_grad(set_to_none=True)
        mon.on_opt_step(x, y, 1e-3)
        mon.log_scalars(res, 1e-3)
    mon.finish()
