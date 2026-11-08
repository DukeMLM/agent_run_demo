"""Neural_Adjoint forward model: the Phase 2 final surrogate adapted for inverse design (Code_Modify output).

Produced by the Modifier agent (manuscript Sec. 2.4, Figure 3) from the Forward Modeler's final solution
phase2_forward/final_model/solution.py (run r06_explore_patchxattn_s32-1106-221446-49f4, data set D_k with
k = 16000, validation MSE 6.6336e-4). Interface expected by framework/neural_adjoint.py:

    load_forward_model(model_dir, device) -> ForwardModel (eval mode, trained weights loaded)
    ForwardModel(g_raw): RAW geometry (B, 14) [h, p, rx1..rx4, ry1..ry4, theta1..theta4]
                         -> RAW absorptivity (B, 2001), differentiable w.r.t. g_raw, scalers embedded:
                             x_scaled = (g_raw - x_shift) / x_scale          (minmax to [-1, 1])
                             y_raw    = net(x_scaled) * y_scale + y_shift     (global_standard)

Files in the model directory:
    forward_model.py  this file. Self-contained: the network code below is a verbatim copy of the model section
                      of the Phase 2 solution; nothing is imported from phase2_forward/.
    forward_model.pt  dict {format, network_class, in_dim, out_dim, config, state_dict, provenance}: the run's
                      training config (result.json "config"; the network is rebuilt with it after
                      torch.manual_seed(config["seed"]), as the harness does) and the trained network parameters
                      (state_dict keys and tensors identical to phase2_forward/final_model/best_model.pt).
    scaler.json       x_shift, x_scale, y_shift, y_scale (used here) and the raw training range x_min, x_max (used
                      by the Neural_Adjoint tool), copied from the Phase 2 scaler.
    meta.json         provenance, architecture summary, Phase 2 metrics and the export validation.

Network (PatchXAttnTransformer, 6,164,769 parameters): patch cross-attention encoder-decoder transformer.
Encoder: 1 global (h, p) token + 4 resonator tokens (shared GELU-MLP embedding of the ellipse shape-tensor
invariants), 2 pre-LN self-attention blocks with learned per-head V4-relation logit biases (exactly invariant to
the supercell-shift resonator permutations). Decoder: 63 spectral patch queries (S = 32 points each; learned
position embedding + projection of the pooled geometry latent), 4 pre-LN blocks of [patch self-attention with
relative-position bias, cross-attention to the 5 geometry tokens, GELU FFN 4x]; one shared Linear(256 -> 64)
renderer, overlap-added at stride 32 and centre-cropped to 2001 points, plus a per-point bias. d_model 256,
8 heads. Dropout is inactive in eval mode, so the model is a deterministic function of the geometry.
"""
import json
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# ===== BEGIN verbatim copy: model section of phase2_forward/final_model/solution.py (lines 99-265) =====
N_REL = 7  # 0..3: resonator pair i XOR j; 4: global-global; 5: global query -> resonator key; 6: reverse


def relation_matrix() -> torch.Tensor:
    """(5, 5) relation-type index for the token order [global, res1, res2, res3, res4]."""
    R = torch.zeros(5, 5, dtype=torch.long)
    R[0, 0] = 4
    R[0, 1:] = 5
    R[1:, 0] = 6
    for i in range(4):
        for j in range(4):
            R[1 + i, 1 + j] = i ^ j
    return R


class SelfAttn(nn.Module):
    """Multi-head self-attention (fused qkv, SDPA kernel) with an optional additive per-head bias (h, T, T)."""

    def __init__(self, d, n_heads):
        super().__init__()
        assert d % n_heads == 0, "d_model must be divisible by n_heads"
        self.h, self.dh = n_heads, d // n_heads
        self.qkv = nn.Linear(d, 3 * d)
        self.o = nn.Linear(d, d)

    def forward(self, x, bias=None):
        B, T, D = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.h, self.dh).permute(2, 0, 3, 1, 4).unbind(0)
        m = None if bias is None else bias.to(q.dtype)[None].expand(B, -1, -1, -1)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=m)
        return self.o(y.transpose(1, 2).reshape(B, T, D))


class CrossAttn(nn.Module):
    """Multi-head cross-attention (queries from xq, keys / values from xkv), SDPA kernel."""

    def __init__(self, d, n_heads):
        super().__init__()
        assert d % n_heads == 0, "d_model must be divisible by n_heads"
        self.h, self.dh = n_heads, d // n_heads
        self.q = nn.Linear(d, d)
        self.kv = nn.Linear(d, 2 * d)
        self.o = nn.Linear(d, d)

    def forward(self, xq, xkv):
        B, Tq, D = xq.shape
        Tk = xkv.shape[1]
        q = self.q(xq).view(B, Tq, self.h, self.dh).transpose(1, 2)
        k, v = self.kv(xkv).view(B, Tk, 2, self.h, self.dh).permute(2, 0, 3, 1, 4).unbind(0)
        y = F.scaled_dot_product_attention(q, k, v)
        return self.o(y.transpose(1, 2).reshape(B, Tq, D))


def _ffn(d, mult):
    return nn.Sequential(nn.Linear(d, mult * d), nn.GELU(), nn.Linear(mult * d, d))


class EncBlock(nn.Module):
    """Pre-LN self-attention over [global, res1..res4] with V4-relation logit biases, then a GELU FFN."""

    def __init__(self, d, n_heads, mult, drop, bias_init):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.att = SelfAttn(d, n_heads)
        self.ff = _ffn(d, mult)
        self.drop = nn.Dropout(drop)
        b = torch.zeros(n_heads, N_REL)
        for k in range(n_heads):
            b[k, k % 4] = bias_init
        self.rel_bias = nn.Parameter(b)

    def forward(self, x, rel):
        x = x + self.drop(self.att(self.n1(x), self.rel_bias[:, rel]))
        return x + self.drop(self.ff(self.n2(x)))


class DecBlock(nn.Module):
    """Pre-LN [patch self-attention + relative-position bias] -> [cross-attention to geometry] -> FFN."""

    def __init__(self, d, n_heads, mult, drop, n_patches, alibi_heads):
        super().__init__()
        self.n1, self.n2, self.n3 = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)
        self.sa = SelfAttn(d, n_heads)
        self.ca = CrossAttn(d, n_heads)
        self.ff = _ffn(d, mult)
        self.drop = nn.Dropout(drop)
        off = torch.arange(-(n_patches - 1), n_patches).abs().float()
        rb = torch.zeros(n_heads, 2 * n_patches - 1)
        for k in range(min(n_heads, alibi_heads)):
            rb[k] = -off * (0.5 ** (k + 1))           # slopes 1/2, 1/4, 1/8, ...: local heads
        self.rel_pos = nn.Parameter(rb)

    def forward(self, t, mem, relpos):
        t = t + self.drop(self.sa(self.n1(t), self.rel_pos[:, relpos]))
        t = t + self.drop(self.ca(self.n2(t), mem))
        return t + self.drop(self.ff(self.n3(t)))


def _mlp(i, h, o):
    return nn.Sequential(nn.Linear(i, h), nn.GELU(), nn.Linear(h, o))


class PatchXAttnTransformer(nn.Module):
    N_RES_FEAT = 8
    N_GLOB_FEAT = 2
    EPS = 0.1   # smoothing of |d| = |rx - ry| in scaled units (d_scaled in [-2, 2], grid step 0.25)

    def __init__(self, in_dim, out_dim, cfg):
        super().__init__()
        assert in_dim == 14, "expects [h, p, rx1..4, ry1..4, theta1..4]"
        d, nh, mult = int(cfg["d_model"]), int(cfg["n_heads"]), int(cfg["ffn_mult"])
        drop = float(cfg["dropout"])
        S = int(cfg["patch"])
        P = int(math.ceil(out_dim / S))
        self.S, self.P, self.out_dim = S, P, out_dim
        self.crop = ((P + 1) * S - out_dim) // 2
        self.res_embed = _mlp(self.N_RES_FEAT, d, d)       # shared by the 4 resonators
        self.glob_embed = _mlp(self.N_GLOB_FEAT, d, d)
        self.enc = nn.ModuleList([EncBlock(d, nh, mult, drop, float(cfg["rel_bias_init"]))
                                  for _ in range(int(cfg["enc_layers"]))])
        self.enc_norm = nn.LayerNorm(d)
        self.z_proj = _mlp(2 * d, d, d)
        self.pos = nn.Parameter(torch.randn(P, d) * 0.02)
        self.dec = nn.ModuleList([DecBlock(d, nh, mult, drop, P, int(cfg["alibi_heads"]))
                                  for _ in range(int(cfg["dec_layers"]))])
        self.dec_norm = nn.LayerNorm(d)
        self.out = nn.Linear(d, 2 * S)
        nn.init.normal_(self.out.weight, std=float(cfg["out_init_std"]))
        nn.init.zeros_(self.out.bias)
        self.out_bias = nn.Parameter(torch.zeros(out_dim))
        self.register_buffer("rel", relation_matrix())
        idx = torch.arange(P)
        self.register_buffer("relpos", idx[:, None] - idx[None, :] + (P - 1))

    @classmethod
    def features(cls, x):
        h, p = x[:, 0:1], x[:, 1:2]
        rx, ry, th = x[:, 2:6], x[:, 6:10], x[:, 10:14]
        a = th * (math.pi / 2)          # = 2 * theta (rad) under minmax scaling of theta in +-pi/4
        dr = rx - ry
        s = rx + ry
        u, v = dr * torch.cos(a), dr * torch.sin(a)       # traceless part of the shape tensor
        dn = torch.sqrt(u * u + v * v + cls.EPS ** 2)    # smooth |rx - ry| (invariant)
        res = torch.stack([s, dn, u, v, u / dn, v / dn,
                           h.expand(-1, 4), p.expand(-1, 4)], dim=-1)          # (B, 4, 8)
        glob = torch.cat([h, p], dim=1)                                          # (B, 2)
        return res, glob

    def forward(self, x):
        res, glob = self.features(x)
        tok = torch.cat([self.glob_embed(glob)[:, None, :], self.res_embed(res)], dim=1)  # (B, 5, d)
        for blk in self.enc:
            tok = blk(tok, self.rel)
        mem = self.enc_norm(tok)
        z = self.z_proj(torch.cat([mem[:, 0], mem[:, 1:].mean(dim=1)], dim=-1))         # V4-invariant
        t = self.pos[None, :, :] + z[:, None, :]                                          # (B, P, d)
        for blk in self.dec:
            t = blk(t, mem, self.relpos)
        a = self.out(self.dec_norm(t))                                                    # (B, P, 2S)
        S = self.S
        y = F.pad(a[..., :S], (0, 0, 0, 1)) + F.pad(a[..., S:], (0, 0, 1, 0))           # overlap-add
        y = y.reshape(y.shape[0], (self.P + 1) * S)[:, self.crop:self.crop + self.out_dim]
        return y + self.out_bias


def build_model(in_dim, out_dim, cfg):
    return PatchXAttnTransformer(in_dim, out_dim, cfg)
# ===== END verbatim copy =====


# ============================================================================================ inverse-design wrapper
FORMAT = "metasurface-na-forward-model/1"
NETWORK_CLASS = "PatchXAttnTransformer"
WEIGHTS_FILE = "forward_model.pt"
SCALER_FILE = "scaler.json"
GEOM_DIM, SPEC_DIM = 14, 2001


class ForwardModel(nn.Module):
    """Trained network + data scalers: RAW geometry (B, 14) -> RAW absorptivity (B, 2001).

    The scaler vectors are non-persistent buffers (they follow .to(device) but are not part of the state_dict):
    scaler.json is their single source, forward_model.pt holds only the network."""

    def __init__(self, net: nn.Module, x_shift, x_scale, y_shift, y_scale):
        super().__init__()
        self.net = net
        for name, val, n in (("x_shift", x_shift, GEOM_DIM), ("x_scale", x_scale, GEOM_DIM),
                             ("y_shift", y_shift, SPEC_DIM), ("y_scale", y_scale, SPEC_DIM)):
            t = torch.as_tensor(val, dtype=torch.float32).reshape(-1)
            if t.numel() != n:
                raise ValueError(f"{name} has {t.numel()} values, expected {n}")
            self.register_buffer(name, t, persistent=False)

    def forward(self, g_raw: torch.Tensor) -> torch.Tensor:
        x = (g_raw.to(self.x_shift.dtype) - self.x_shift) / self.x_scale   # raw geometry -> network input
        y = self.net(x).to(self.y_scale.dtype)   # scaled spectrum (bf16 under autocast -> fp32, as the harness)
        return y * self.y_scale + self.y_shift                              # raw absorptivity


def load_forward_model(model_dir: str, device) -> ForwardModel:
    """Rebuild the network with its training config (seeded like the harness), load the trained parameters and
    the scalers from model_dir, move everything to `device` and return the model in eval mode."""
    model_dir = Path(model_dir)
    ckpt = torch.load(model_dir / WEIGHTS_FILE, map_location="cpu", weights_only=True)
    if ckpt.get("format") != FORMAT or ckpt.get("network_class") != NETWORK_CLASS:
        raise ValueError(f"{model_dir / WEIGHTS_FILE}: expected format {FORMAT!r} with network {NETWORK_CLASS!r}, "
                         f"got {ckpt.get('format')!r} / {ckpt.get('network_class')!r}")
    cfg = dict(ckpt["config"])
    sc = json.loads((model_dir / SCALER_FILE).read_text(encoding="utf-8"))
    with torch.random.fork_rng():   # seed as the harness does without changing the caller's RNG state
        torch.manual_seed(int(cfg["seed"]))
        net = build_model(int(ckpt["in_dim"]), int(ckpt["out_dim"]), cfg)
    net.load_state_dict(ckpt["state_dict"], strict=True)
    model = ForwardModel(net, sc["x_shift"], sc["x_scale"], sc["y_shift"], sc["y_scale"])
    return model.to(torch.device(device)).eval()
