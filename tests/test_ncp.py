import torch
import pytest
from model.model_ncp import NCPConfig, NCPForCausalLM

VOCAB, T, K = 6400, 64, 4

def make_model(**kw):
    cfg = NCPConfig(max_position_embeddings=T, concept_chunk=K, **kw)
    torch.manual_seed(0)
    return NCPForCausalLM(cfg).eval()

def ids(B=2):
    g = torch.Generator().manual_seed(0)
    return torch.randint(0, VOCAB, (B, T), generator=g)

def test_param_count_64m_class():
    model = make_model()
    total = sum(p.numel() for p in model.parameters()) / 1e6
    print(f'NCP total params: {total:.2f}M')
    assert 55 <= total <= 75

def test_forward_shapes_and_losses():
    model = make_model()
    x = ids()
    out = model(x, labels=x)
    assert out['logits'].shape == (2, T, VOCAB)
    for key in ('loss', 'loss_ntp', 'loss_ncp', 'loss_vq'):
        assert torch.isfinite(out[key])
    cfg = model.config
    expected = out['loss_ntp'] + cfg.ncp_loss_weight * out['loss_ncp'] + cfg.vq_loss_weight * out['loss_vq']
    assert torch.allclose(out['loss'], expected)

def test_causality_no_future_leakage():
    """Changing token at position j must not change logits at positions < j —
    this covers attention, the concept path, and both shifted CRCs."""
    model = make_model()
    x = ids(1)
    j = 20
    x2 = x.clone()
    x2[0, j] = (x[0, j] + 1) % VOCAB
    with torch.no_grad():
        l1 = model(x)['logits']
        l2 = model(x2)['logits']
    assert torch.allclose(l1[:, :j], l2[:, :j], atol=1e-6), "future token leaked into earlier logits"
    assert not torch.allclose(l1[:, j:], l2[:, j:]), "change had no effect (sanity)"

def test_concept_signal_reaches_decoder():
    """With sliding-window attention, a change in chunk 0 can reach distant
    positions ONLY through the concept pathway (predicted-concept injection and
    the shifted CM->Decoder CRC). Position 20 sees tokens 13-20 in attention but
    c_0 via the concept module, so its logits must differ."""
    model = make_model(sliding_window=4)
    x = ids(1)
    x2 = x.clone()
    x2[0, 1] = (x[0, 1] + 1) % VOCAB   # perturb inside first chunk
    with torch.no_grad():
        l1 = model(x)['logits']
        l2 = model(x2)['logits']
    assert not torch.allclose(l1[:, 20], l2[:, 20]), "concept pathway carried no information"
    assert torch.allclose(l1[:, 0], l2[:, 0], atol=1e-6), "causality violated at position 0"

def test_vq_loss_updates_codebook_only():
    model = make_model()
    x = ids()
    model.zero_grad(set_to_none=True)
    out = model(x, labels=x)
    out['loss_vq'].backward()
    assert model.model.quantizer.codebook.grad is not None and model.model.quantizer.codebook.grad.abs().sum() > 0
    enc_grads = [p.grad for n, p in model.named_parameters() if 'token_encoder' in n and p.grad is not None]
    assert all(g.abs().sum() == 0 for g in enc_grads), "VQ loss must not update the encoder (stop-gradient)"

def test_ncp_loss_trains_encoder_and_concept_module():
    model = make_model()
    x = ids()
    model.zero_grad(set_to_none=True)
    out = model(x, labels=x)
    out['loss_ncp'].backward()
    for name in ('concept_module', 'token_encoder'):
        grads = [p.grad for n, p in model.named_parameters() if name in n and p.grad is not None]
        assert any(g.abs().sum() > 0 for g in grads), f"NCP loss should train {name}"
    head_grads = [p.grad for p in model.model.quantizer.pred_heads.parameters()]
    assert all(g is not None and g.abs().sum() > 0 for g in head_grads)

def test_joint_backward_covers_all_paths():
    model = make_model()
    model.train()
    x = ids()
    out = model(x, labels=x)
    out['loss'].backward()
    missing = [n for n, p in model.named_parameters()
               if p.grad is None or p.grad.abs().sum() == 0]
    print('params without grad:', missing)
    # router first layers see zero grad at init because the output layer is
    # zero-init (IRC must start as an exact residual); they train from step 1.
    # everything else, including router output layers, must get real gradients.
    assert all(('net.0' in n and ('irc' in n or 'crc_router' in n)) for n in missing)
    for n, p in model.named_parameters():
        if 'net.2' in n and ('irc' in n or 'crc_router' in n):
            assert p.grad is not None and p.grad.abs().sum() > 0, n

def test_vanilla_baseline():
    model = make_model(arch='vanilla', use_irc=False)
    x = ids()
    out = model(x, labels=x)
    assert out['loss_ncp'] is None and torch.isfinite(out['loss_ntp'])
    out['loss'].backward()
    total = sum(p.numel() for p in model.parameters()) / 1e6
    print(f'vanilla total params: {total:.2f}M')

def test_generate_runs():
    model = make_model()
    x = ids(1)[:, :8]
    out = model.generate(input_ids=x, max_new_tokens=9, do_sample=False, eos_token_id=None)
    assert out.shape[1] == 17
