import math, torch, torch.nn.functional as F
from torch import nn
from transformers.activations import ACT2FN
from transformers import PreTrainedModel, PretrainedConfig

# 🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏
#                                     NCP Config
# 🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏
class NCPConfig(PretrainedConfig):
    model_type = "ncp"
    def __init__(self, hidden_size=768, n_enc_layers=3, n_concept_layers=2, n_dec_layers=3, arch="ncp", **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        # arch: 'ncp' -> TokenEncoder + ConceptModule + TokenDecoder
        #       'vanilla' -> one standard stack of (n_enc+n_concept+n_dec) layers (parameter-matched baseline)
        self.arch = arch
        self.n_enc_layers = n_enc_layers
        self.n_concept_layers = n_concept_layers
        self.n_dec_layers = n_dec_layers
        self.num_hidden_layers = n_enc_layers + n_concept_layers + n_dec_layers
        self.dropout = kwargs.get("dropout", 0.0)
        self.vocab_size = kwargs.get("vocab_size", 6400)
        self.bos_token_id = kwargs.get("bos_token_id", 1)
        self.eos_token_id = kwargs.get("eos_token_id", 2)
        self.flash_attn = kwargs.get("flash_attn", True)
        self.num_attention_heads = kwargs.get("num_attention_heads", 12)
        self.num_key_value_heads = kwargs.get("num_key_value_heads", self.num_attention_heads)  # MHA, as in the paper
        self.head_dim = kwargs.get("head_dim", self.hidden_size // self.num_attention_heads)
        self.hidden_act = kwargs.get("hidden_act", 'silu')
        self.intermediate_size = kwargs.get("intermediate_size", math.ceil(hidden_size * math.pi / 64) * 64)
        self.max_position_embeddings = kwargs.get("max_position_embeddings", 32768)
        self.rms_norm_eps = kwargs.get("rms_norm_eps", 1e-6)
        self.rope_theta = kwargs.get("rope_theta", 5e5)
        self.tie_word_embeddings = kwargs.get("tie_word_embeddings", True)
        # every `full_attn_every`-th layer attends globally, others use `sliding_window` (None/0 = all full)
        self.full_attn_every = kwargs.get("full_attn_every", 4)
        self.sliding_window = kwargs.get("sliding_window", 0)
        # --- Next Concept Prediction ---
        self.concept_chunk = kwargs.get("concept_chunk", 4)          # k tokens per concept
        self.pq_segments = kwargs.get("pq_segments", hidden_size // 128)  # S codebooks (codeword dim = hidden_size/S)
        self.pq_codewords = kwargs.get("pq_codewords", 128)          # N entries per codebook
        self.ncp_loss_weight = kwargs.get("ncp_loss_weight", 1.0)    # alpha
        self.vq_loss_weight = kwargs.get("vq_loss_weight", 1.0)      # beta
        self.use_irc = kwargs.get("use_irc", True)                   # intra-module residual connections
        self.use_crc = kwargs.get("use_crc", True)                   # cross-module residual connections
        self.residual_hidden = kwargs.get("residual_hidden", 64)     # bottleneck of IRC/CRC routing MLPs
        self.crc_scale_init = kwargs.get("crc_scale_init", 0.01)     # diag scale init for CRC (small, per paper)

# 🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏
#                                     Building blocks
# 🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏
class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        return (self.weight * self.norm(x.float())).type_as(x)

def precompute_freqs_cis(dim: int, end: int = int(32 * 1024), rope_base: float = 1e6):
    freqs = 1.0 / (rope_base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1)
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1)
    return freqs_cos, freqs_sin

def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    def rotate_half(x): return torch.cat((-x[..., x.shape[-1] // 2:], x[..., : x.shape[-1] // 2]), dim=-1)
    q_embed = ((q * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(q) * sin.unsqueeze(unsqueeze_dim))).to(q.dtype)
    k_embed = ((k * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(k) * sin.unsqueeze(unsqueeze_dim))).to(k.dtype)
    return q_embed, k_embed

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    bs, slen, num_key_value_heads, head_dim = x.shape
    if n_rep == 1: return x
    return (x[:, :, :, None, :].expand(bs, slen, num_key_value_heads, n_rep, head_dim).reshape(bs, slen, num_key_value_heads * n_rep, head_dim))

class Attention(nn.Module):
    def __init__(self, config: NCPConfig, sliding_window: int = 0):
        super().__init__()
        self.num_key_value_heads = config.num_attention_heads if config.num_key_value_heads is None else config.num_key_value_heads
        self.n_local_heads = config.num_attention_heads
        self.n_local_kv_heads = self.num_key_value_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = config.head_dim
        self.sliding_window = sliding_window  # 0 -> full causal attention
        self.is_causal = True
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)
        # per-head QK RMSNorm: the stable variant identified in Sec. 4.6 of the paper
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.dropout = config.dropout
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and config.flash_attn

    def forward(self, x, position_embeddings, attention_mask=None):
        bsz, seq_len, _ = x.shape
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        xq, xk = self.q_norm(xq), self.k_norm(xk)
        cos, sin = position_embeddings
        xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)
        xq, xk, xv = (xq.transpose(1, 2), repeat_kv(xk, self.n_rep).transpose(1, 2), repeat_kv(xv, self.n_rep).transpose(1, 2))
        if self.flash and self.sliding_window == 0 and (attention_mask is None or torch.all(attention_mask == 1)):
            output = F.scaled_dot_product_attention(xq, xk, xv, dropout_p=self.dropout if self.training else 0.0, is_causal=self.is_causal)
        else:
            scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
            mask = torch.full((seq_len, seq_len), float("-inf"), device=scores.device).triu(1)
            if self.sliding_window > 0:
                band = torch.arange(seq_len, device=scores.device)
                mask = mask.masked_fill((band[:, None] - band[None, :]) >= self.sliding_window, float("-inf"))
            scores += mask
            if attention_mask is not None: scores += (1.0 - attention_mask.unsqueeze(1).unsqueeze(2)) * -1e9
            output = self.attn_dropout(F.softmax(scores.float(), dim=-1).type_as(xq)) @ xv
        output = output.transpose(1, 2).reshape(bsz, seq_len, -1)
        output = self.resid_dropout(self.o_proj(output))
        return output

class FeedForward(nn.Module):
    def __init__(self, config: NCPConfig, intermediate_size: int = None):
        super().__init__()
        intermediate_size = intermediate_size or config.intermediate_size
        self.gate_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, config.hidden_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

class NCPBlock(nn.Module):
    """Pre-norm Transformer block returning its *residual contribution* R = F(H),
    so that callers (IRC / plain residual) decide how it is combined."""
    def __init__(self, layer_id: int, config: NCPConfig):
        super().__init__()
        window = 0
        if config.sliding_window > 0 and (layer_id + 1) % config.full_attn_every != 0:
            window = config.sliding_window
        self.self_attn = Attention(config, sliding_window=window)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = FeedForward(config)

    def forward(self, hidden_states, position_embeddings, attention_mask=None):
        attn_out = self.self_attn(self.input_layernorm(hidden_states), position_embeddings, attention_mask)
        mlp_out = self.mlp(self.post_attention_layernorm(hidden_states + attn_out))
        return attn_out + mlp_out  # R_l = F_l(H_l)

class RouteMLP(nn.Module):
    """Lightweight token-conditioned MLP producing `out_dim` mixing coefficients."""
    def __init__(self, hidden_size: int, out_dim: int, bottleneck: int, last_bias: float = None):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, bottleneck),
            nn.SiLU(),
            nn.Linear(bottleneck, out_dim),
        )
        self.last_bias = last_bias
        nn.init.zeros_(self.net[2].weight)
        nn.init.zeros_(self.net[2].bias)
        if last_bias is not None:
            self.net[2].bias.data[-1] = last_bias

    def forward(self, x):
        return self.net(x)

class TransformerModule(nn.Module):
    """A stack of NCPBlocks with IRC routing and optional incoming CRCs.

    IRC (Eq. 14-18): next state = unnormalized weighted combination of the module
    input and all post-block states; weights come from a token-conditioned MLP on
    the current block output and are initialized to [0,...,0,1] (plain residual).

    CRC (Eq. 19-20): at each layer, a token-conditioned MLP produces softmax
    weights over the source module's exported per-layer states; the weighted,
    RMS-normalized mix is added with a learned diagonal scale.
    """
    def __init__(self, config: NCPConfig, num_layers: int, layer_offset: int = 0,
                 crc_sources: dict = None):
        super().__init__()
        self.config = config
        self.num_layers = num_layers
        self.use_irc = config.use_irc
        self.layers = nn.ModuleList([NCPBlock(layer_offset + i, config) for i in range(num_layers)])
        if self.use_irc:
            self.irc = nn.ModuleList([
                RouteMLP(config.hidden_size, i + 2, config.residual_hidden, last_bias=1.0)
                for i in range(num_layers)
            ])
        # crc_sources: list of dicts {name, num_source_states}; per-layer routers/scales built later
        self.crc_sources = crc_sources or []
        self.crc_router = nn.ModuleDict()
        self.crc_norm = nn.ModuleDict()
        self.crc_scale = nn.ParameterDict()
        for src in self.crc_sources:
            name, k = src['name'], src['num_source_states']
            self.crc_router[name] = nn.ModuleList([
                RouteMLP(config.hidden_size, k, config.residual_hidden) for _ in range(num_layers)
            ])
            self.crc_norm[name] = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.crc_scale[name] = nn.Parameter(torch.full((num_layers, config.hidden_size), config.crc_scale_init))

    def forward(self, x, position_embeddings, crc_inputs: dict = None, attention_mask=None):
        """x: (B, L, d). crc_inputs: {name: list of (B, L, d) source states, length = num_source_states}.
        Returns (final hidden state, list of all candidate states incl. input)."""
        H = x
        Xs = [H]
        for i, block in enumerate(self.layers):
            if crc_inputs:
                for src in self.crc_sources:
                    name = src['name']
                    states = crc_inputs[name]  # list of (B, L, d)
                    alpha = F.softmax(self.crc_router[name][i](H), dim=-1)          # (B, L, K)
                    mixed = sum(alpha[..., j:j + 1] * self.crc_norm[name](states[j]) for j in range(len(states)))
                    H = H + self.crc_scale[name][i] * mixed
            R = block(H, position_embeddings, attention_mask)
            Xs.append(H + R)
            if self.use_irc:
                w = self.irc[i](R)                                                # (B, L, i+2)
                H = sum(w[..., j:j + 1] * Xs[j] for j in range(len(Xs)))
            else:
                H = H + R
        return H, Xs

class ProductQuantizer(nn.Module):
    """Product-quantized concept vocabulary (Sec. 2.2): S codebooks of N codewords
    each with dimension hidden_size/S, plus S segment prediction heads producing
    softmax distributions whose expectation gives the differentiable concept (Eq. 8-10)."""
    def __init__(self, config: NCPConfig):
        super().__init__()
        self.S = config.pq_segments
        self.N = config.pq_codewords
        self.seg_dim = config.hidden_size // self.S
        assert self.S * self.seg_dim == config.hidden_size, "hidden_size must be divisible by pq_segments"
        self.codebook = nn.Parameter(torch.randn(self.S, self.N, self.seg_dim) * 0.02)
        self.pred_heads = nn.ModuleList([nn.Linear(config.hidden_size, self.N, bias=False) for _ in range(self.S)])

    def quantize(self, c):
        """c: (B, M, d) continuous concepts -> d: (B, M, d) quantized, idx: (B, M, S)"""
        B, M, d = c.shape
        seg = c.view(B, M, self.S, self.seg_dim)                                   # (B, M, S, seg)
        # squared L2 distance to every codeword: ||c||^2 - 2 c·e + ||e||^2
        dist = seg.pow(2).sum(-1, keepdim=True) \
               - 2 * torch.einsum('bmsk,snk->bmsn', seg, self.codebook) \
               + self.codebook.pow(2).sum(-1).view(1, 1, self.S, self.N)            # (B, M, S, N)
        idx = dist.argmin(-1)                                                      # (B, M, S)
        dq = torch.gather(
            self.codebook.unsqueeze(0).unsqueeze(0).expand(B, M, -1, -1, -1), 3,
            idx.unsqueeze(-1).unsqueeze(-1).expand(B, M, self.S, 1, self.seg_dim)
        ).squeeze(3)                                                               # (B, M, S, seg)
        return dq.reshape(B, M, d), idx

    def predict(self, u):
        """u: (B, M, d) concept-module output -> chat: (B, M, d) differentiable concept prediction,
        pi: (B, M, S, N) codebook distributions."""
        pi = torch.stack([F.softmax(head(u), dim=-1) for head in self.pred_heads], dim=2)  # (B, M, S, N)
        chat = torch.einsum('bmsn,snk->bmsk', pi, self.codebook)                   # (B, M, S, seg)
        return chat.reshape(u.shape[0], u.shape[1], -1), pi

# 🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏
#                                     NCP Model
# 🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏
class NCPModel(nn.Module):
    def __init__(self, config: NCPConfig):
        super().__init__()
        self.config = config
        self.k = config.concept_chunk
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        freqs_cos, freqs_sin = precompute_freqs_cis(dim=config.head_dim, end=config.max_position_embeddings, rope_base=config.rope_theta)
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        if config.arch == "ncp":
            self.token_encoder = TransformerModule(config, config.n_enc_layers, layer_offset=0)
            self.concept_module = TransformerModule(
                config, config.n_concept_layers, layer_offset=config.n_enc_layers,
                crc_sources=[{'name': 'enc', 'num_source_states': config.n_enc_layers}] if config.use_crc else []
            )
            self.token_decoder = TransformerModule(
                config, config.n_dec_layers, layer_offset=config.n_enc_layers + config.n_concept_layers,
                crc_sources=[
                    {'name': 'enc', 'num_source_states': config.n_enc_layers},
                    {'name': 'cm', 'num_source_states': config.n_concept_layers},
                ] if config.use_crc else []
            )
            self.quantizer = ProductQuantizer(config)
            # pooled concepts live on a normalized scale so the MSE-based VQ/NCP
            # objectives stay bounded; the decoder relearns injection strength
            self.concept_gain = nn.Parameter(torch.ones(config.hidden_size))
        else:
            self.vanilla = TransformerModule(config, config.num_hidden_layers, layer_offset=0)

    def mean_pool(self, h, m):
        """(B, T, d) -> (B, m, d): mean-pool each of the first m complete k-token chunks."""
        B, T, d = h.shape
        return h[:, : m * self.k].view(B, m, self.k, d).mean(2)

    def causal_repeat(self, concept_states, t):
        """Map per-concept states (B, M, d) to token resolution (B, T, d) with the
        causal shift of Eq. 11: token position i receives concept index floor(i/k)-1
        (the state summarizing only fully completed chunks), zero for i < k."""
        B, M, d = concept_states.shape
        idx = torch.div(torch.arange(t, device=concept_states.device), self.k, rounding_mode='floor') - 1
        valid = idx >= 0
        idx = idx.clamp(min=0, max=M - 1)
        out = concept_states[:, idx] * valid.view(1, t, 1).type_as(concept_states)
        return out

    def forward(self, input_ids, attention_mask=None, **kwargs):
        B, T = input_ids.shape
        position_embeddings = (self.freqs_cos[:T], self.freqs_sin[:T])
        x = self.dropout(self.embed_tokens(input_ids))

        if self.config.arch != "ncp":
            hidden, _ = self.vanilla(x, position_embeddings, attention_mask=attention_mask)
            return self.norm(hidden), {}

        # 1. Token Encoder (Eq. 1)
        h, xs_enc = self.token_encoder(x, position_embeddings, attention_mask=attention_mask)
        m = T // self.k

        if m == 0:  # sequence shorter than one chunk: no concept signal yet
            h_dec, _ = self.token_decoder(h, position_embeddings, crc_inputs=None, attention_mask=attention_mask)
            z = h.new_zeros(())
            return self.norm(h_dec), {'loss_ncp': z, 'loss_vq': z}

        # 2. continuous concepts + VQ (Eq. 2-6); concepts are RMS-normalized to a
        # fixed scale before quantization, prediction, and feedback
        c = F.rms_norm(self.mean_pool(h, m), (self.config.hidden_size,))           # (B, M, d)
        d_quant, code_idx = self.quantizer.quantize(c)
        vq_loss = (d_quant - c.detach()).pow(2).mean()

        # 3. Concept Module predicts next concept (Eq. 7-10); Enc->CM CRC states are chunk-pooled
        cm_crc = {'enc': [self.mean_pool(s, m) for s in xs_enc[1:]]} if self.config.use_crc else None
        u, xs_cm = self.concept_module(
            c, (self.freqs_cos[:m], self.freqs_sin[:m]), crc_inputs=cm_crc
        )
        chat, pi = self.quantizer.predict(u)                                       # chat[:, j] predicts c[:, j+1]
        ncp_loss = (chat[:, :-1] - c[:, 1:].detach()).pow(2).mean() if m > 1 else c.new_zeros(())

        # 4. inject predicted concepts into the token stream (Eq. 11-12)
        b = self.causal_repeat(chat, T) * self.concept_gain
        h_tilde = h + b

        # 5. Token Decoder with Enc->Dec and (causally shifted) CM->Dec CRCs (Eq. 13, 19-20)
        dec_crc = None
        if self.config.use_crc:
            dec_crc = {
                'enc': xs_enc[1:],
                'cm': [self.causal_repeat(s, T) for s in xs_cm[1:]],
            }
        h_dec, _ = self.token_decoder(h_tilde, position_embeddings, crc_inputs=dec_crc, attention_mask=attention_mask)
        return self.norm(h_dec), {'loss_ncp': ncp_loss, 'loss_vq': vq_loss}

class NCPForCausalLM(PreTrainedModel):
    config_class = NCPConfig
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
    def __init__(self, config: NCPConfig = None):
        self.config = config or NCPConfig()
        super().__init__(self.config)
        self.model = NCPModel(self.config)
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
        if self.config.tie_word_embeddings: self.model.embed_tokens.weight = self.lm_head.weight
        self.post_init()
        self._init_routing()  # post_init/_init_weights overwrote the IRC/CRC initializations

    def _init_routing(self):
        for module in self.modules():
            if isinstance(module, RouteMLP):
                nn.init.zeros_(module.net[2].weight)
                nn.init.zeros_(module.net[2].bias)
                if module.last_bias is not None:
                    module.net[2].bias.data[-1] = module.last_bias
        if self.config.arch == "ncp":
            for name, module in [('concept_module', self.model.concept_module), ('token_decoder', self.model.token_decoder)]:
                for scale in module.crc_scale.values():
                    scale.data.fill_(self.config.crc_scale_init)

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        hidden_states, losses = self.model(input_ids, attention_mask, **kwargs)
        logits = self.lm_head(hidden_states)
        loss, loss_ntp = None, None
        if labels is not None:
            x, y = logits[..., :-1, :].contiguous(), labels[..., 1:].contiguous()
            loss_ntp = F.cross_entropy(x.view(-1, x.size(-1)), y.view(-1), ignore_index=-100)
            loss = loss_ntp
            if 'loss_ncp' in losses: loss = loss + self.config.ncp_loss_weight * losses['loss_ncp']
            if 'loss_vq' in losses: loss = loss + self.config.vq_loss_weight * losses['loss_vq']
        return {'loss': loss, 'loss_ntp': loss_ntp, 'loss_ncp': losses.get('loss_ncp'),
                'loss_vq': losses.get('loss_vq'), 'logits': logits, 'hidden_states': hidden_states}

    @torch.inference_mode()
    def generate(self, input_ids=None, max_new_tokens=512, temperature=0.85, top_p=0.85, top_k=50,
                 eos_token_id=2, do_sample=True, repetition_penalty=1.0, **kwargs):
        # full forward per step: each call recomputes the concept path on the grown sequence,
        # so predicted concepts for the chunk under generation are always conditioned on
        # concepts pooled from fully completed chunks only (Eq. 11 causality)
        finished = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
        for _ in range(max_new_tokens):
            logits = self.forward(input_ids)['logits'][:, -1, :] / temperature
            if repetition_penalty != 1.0:
                for i in range(input_ids.shape[0]):
                    seen = torch.unique(input_ids[i]); score = logits[i, seen]
                    logits[i, seen] = torch.where(score > 0, score / repetition_penalty, score * repetition_penalty)
            if top_k > 0:
                logits[logits < torch.topk(logits, top_k)[0][..., -1, None]] = -float('inf')
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                mask = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1) > top_p
                mask[..., 1:], mask[..., 0] = mask[..., :-1].clone(), 0
                logits[mask.scatter(1, sorted_indices, mask)] = -float('inf')
            next_token = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1) if do_sample else torch.argmax(logits, dim=-1, keepdim=True)
            if eos_token_id is not None:
                next_token = torch.where(finished.unsqueeze(-1), next_token.new_full((next_token.shape[0], 1), eos_token_id), next_token)
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            if eos_token_id is not None:
                finished |= next_token.squeeze(-1).eq(eos_token_id)
                if finished.all(): break
        return input_ids
