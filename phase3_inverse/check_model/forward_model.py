"""Neural_Adjoint check model: an independently trained forward surrogate of a different architecture family,
used as a proxy for re-simulation (no CST model of the ADM cell is available).

Produced by the Modifier agent (manuscript Sec. 2.4, Figure 3) in the same format as phase3_inverse/na_model, from
the Phase 2 run r06_recipe_ens4wd-1106-215908-6c5a (code phase2_forward/runs/r06_recipe_ens4wd-1106-215908-6c5a/
solution.py, identical to phase2_forward/solutions/r06_recipe_ens4wd.py; data set D_k with k = 16000, validation
MSE 1.5030e-3). It is the best successful k = 16000 run outside the final model's patch cross-attention family.
Interface (what framework/neural_adjoint.py expects, also for --check-model-dir):

    load_forward_model(model_dir, device) -> ForwardModel (eval mode, trained weights loaded)
    ForwardModel(g_raw): RAW geometry (B, 14) [h, p, rx1..rx4, ry1..ry4, theta1..theta4]
                         -> RAW absorptivity (B, 2001), differentiable w.r.t. g_raw, scalers embedded:
                             x_scaled = (g_raw - x_shift) / x_scale          (minmax to [-1, 1])
                             y_raw    = net(x_scaled) * y_scale + y_shift     (global_standard)

Files in the model directory:
    forward_model.py  this file. Self-contained: the network code below is a verbatim copy of the model section
                      of the run's solution; nothing is imported from phase2_forward/.
    forward_model.pt  dict {format, network_class, in_dim, out_dim, config, state_dict, provenance}: the run's
                      training config (result.json "config"; the network is rebuilt with it after
                      torch.manual_seed(config["seed"]), as the harness does) and the trained network parameters
                      (state_dict keys and tensors identical to the run's best_model.pt).
    scaler.json       x_shift, x_scale, y_shift, y_scale (used here) and the raw training range x_min, x_max, copied
                      from the run's scaler (identical to the final model's: same D_16000 training split).
    meta.json         provenance, selection rationale, architecture summary, Phase 2 metrics, export validation.

Network (EnsV4RelTransformer, 46,251,716 parameters): batched deep ensemble of K = 4 independently initialised
V4-exact relational transformers with a global MLP head. Per member: 1 global (h, p) token + 4 resonator tokens
(shared GELU-MLP embedding of the ellipse shape-tensor invariants), 4 pre-LN self-attention blocks (d_model 320,
8 heads, GELU FFN 4x) with per-head V4-relation logit biases, mean pool, head concat(global, mean resonator) ->
2 x Linear-GELU(1536) -> Linear(2001). The output is the member mean. Unlike the final model it has no spectral
tokens and no attention over frequency: each member's MLP head emits all 2001 points at once. Dropout is
inactive in eval mode, so the model is a deterministic function of the geometry.
"""
import json
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# ===== BEGIN verbatim copy: model section of phase2_forward/runs/r06_recipe_ens4wd-1106-215908-6c5a/solution.py (lines 115-265) =====
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


class BLinear(nn.Module):
    """K independent Linear layers applied as one batched matmul: (K, N, i) -> (K, N, o).
    Init per member identical to nn.Linear (U(+-1/sqrt(i)) for weight and bias)."""

    def __init__(self, K, i, o):
        super().__init__()
        bound = 1.0 / math.sqrt(i)
        self.weight = nn.Parameter(torch.empty(K, i, o).uniform_(-bound, bound))
        self.bias = nn.Parameter(torch.empty(K, 1, o).uniform_(-bound, bound))

    def forward(self, x):
        return torch.baddbmm(self.bias, x, self.weight)


class BLayerNorm(nn.Module):
    """K independent LayerNorms over the last dim of (K, N, d)."""

    def __init__(self, K, d, eps=1e-5):
        super().__init__()
        self.d, self.eps = d, eps
        self.weight = nn.Parameter(torch.ones(K, 1, d))
        self.bias = nn.Parameter(torch.zeros(K, 1, d))

    def forward(self, x):
        return torch.addcmul(self.bias, F.layer_norm(x, (self.d,), eps=self.eps), self.weight)


class BRelAttention(nn.Module):
    """Per-member multi-head self-attention with a learned per-head logit bias per relation type."""

    def __init__(self, K, d, n_heads, attn_drop, bias_init):
        super().__init__()
        assert d % n_heads == 0, "d_model must be divisible by n_heads"
        self.K, self.h, self.dh = K, n_heads, d // n_heads
        self.qkv = BLinear(K, d, 3 * d)
        self.proj = BLinear(K, d, d)
        b = torch.zeros(K, n_heads, N_REL)
        for k in range(n_heads):
            b[:, k, k % 4] = bias_init
        self.rel_bias = nn.Parameter(b)
        self.attn_drop = float(attn_drop)

    def forward(self, x, rel, B, T):                       # x: (K, B*T, d)
        K, h, dh = self.K, self.h, self.dh
        q, k, v = self.qkv(x).view(K, B, T, 3, h, dh).permute(3, 0, 1, 4, 2, 5).unbind(0)  # (K, B, h, T, dh)
        att = (q @ k.transpose(-2, -1)) * (dh ** -0.5) + self.rel_bias[:, :, rel].unsqueeze(1)
        att = att.softmax(dim=-1)
        if self.attn_drop > 0:
            att = F.dropout(att, self.attn_drop, self.training)
        y = (att @ v).permute(0, 1, 3, 2, 4).reshape(K, B * T, h * dh)
        return self.proj(y)


class BBlock(nn.Module):
    def __init__(self, K, d, n_heads, ffn_mult, drop, attn_drop, bias_init):
        super().__init__()
        self.n1 = BLayerNorm(K, d)
        self.att = BRelAttention(K, d, n_heads, attn_drop, bias_init)
        self.n2 = BLayerNorm(K, d)
        self.f1 = BLinear(K, d, ffn_mult * d)
        self.f2 = BLinear(K, ffn_mult * d, d)
        self.drop = float(drop)

    def forward(self, x, rel, B, T):
        x = x + self.att(self.n1(x), rel, B, T)
        y = self.f2(F.gelu(self.f1(self.n2(x))))
        if self.drop > 0:
            y = F.dropout(y, self.drop, self.training)
        return x + y


class EnsV4RelTransformer(nn.Module):
    N_RES_FEAT = 8
    N_GLOB_FEAT = 2
    EPS = 0.1   # smoothing of |d| = |rx - ry| in scaled units (d_scaled in [-2, 2], grid step 0.25)

    def __init__(self, in_dim, out_dim, cfg):
        super().__init__()
        assert in_dim == 14, "expects [h, p, rx1..4, ry1..4, theta1..4]"
        K = self.K = int(cfg["n_members"])
        d, eh, hw = int(cfg["d_model"]), int(cfg["embed_hidden"]), int(cfg["head_width"])
        self.d = d
        self.res_e1, self.res_e2 = BLinear(K, self.N_RES_FEAT, eh), BLinear(K, eh, d)   # shared by 4 resonators
        self.glob_e1, self.glob_e2 = BLinear(K, self.N_GLOB_FEAT, eh), BLinear(K, eh, d)
        self.blocks = nn.ModuleList([
            BBlock(K, d, int(cfg["n_heads"]), int(cfg["ffn_mult"]), float(cfg["dropout"]),
                   float(cfg["attn_dropout"]), float(cfg["rel_bias_init"]))
            for _ in range(int(cfg["n_layers"]))])
        self.norm = BLayerNorm(K, d)
        self.hd = float(cfg["head_dropout"])
        self.h1, self.h2, self.h3 = BLinear(K, 2 * d, hw), BLinear(K, hw, hw), BLinear(K, hw, out_dim)
        self.register_buffer("rel", relation_matrix())

    @classmethod
    def features(cls, x):
        """x (..., 14) scaled -> resonator features (..., 4, 8), global features (..., 2)."""
        h, p = x[..., 0:1], x[..., 1:2]
        rx, ry, th = x[..., 2:6], x[..., 6:10], x[..., 10:14]
        a = th * (math.pi / 2)          # = 2 * theta (rad) under minmax scaling of theta in +-pi/4
        dr = rx - ry
        s = rx + ry
        u, v = dr * torch.cos(a), dr * torch.sin(a)       # traceless part of the shape tensor
        dn = torch.sqrt(u * u + v * v + cls.EPS ** 2)    # smooth |rx - ry| (invariant)
        res = torch.stack([s, dn, u, v, u / dn, v / dn, h.expand_as(s), p.expand_as(s)], dim=-1)
        glob = torch.cat([h, p], dim=-1)
        return res, glob

    def forward_members(self, x):
        """x: (B, 14) shared by all members, or (K, B, 14) one view per member -> (K, B, out_dim)."""
        K, d = self.K, self.d
        res, glob = self.features(x)
        if x.dim() == 2:
            B = x.shape[0]
            r = res.reshape(1, B * 4, self.N_RES_FEAT).expand(K, -1, -1).contiguous()
            g = glob.reshape(1, B, self.N_GLOB_FEAT).expand(K, -1, -1).contiguous()
        else:
            B = x.shape[1]
            r = res.reshape(K, B * 4, self.N_RES_FEAT)
            g = glob.reshape(K, B, self.N_GLOB_FEAT)
        r = self.res_e2(F.gelu(self.res_e1(r))).view(K, B, 4, d)
        g = self.glob_e2(F.gelu(self.glob_e1(g))).view(K, B, 1, d)
        tok = torch.cat([g, r], dim=2).reshape(K, B * 5, d)
        for blk in self.blocks:
            tok = blk(tok, self.rel, B, 5)
        tok = self.norm(tok).view(K, B, 5, d)
        z = torch.cat([tok[:, :, 0], tok[:, :, 1:].mean(dim=2)], dim=-1)          # (K, B, 2d)
        z = F.dropout(F.gelu(self.h1(z)), self.hd, self.training)
        z = F.dropout(F.gelu(self.h2(z)), self.hd, self.training)
        return self.h3(z)

    def forward(self, x):
        return self.forward_members(x).mean(dim=0)


def build_model(in_dim, out_dim, cfg):
    return EnsV4RelTransformer(in_dim, out_dim, cfg)
# ===== END verbatim copy =====


# ============================================================================================ inverse-design wrapper
FORMAT = "metasurface-na-forward-model/1"
NETWORK_CLASS = "EnsV4RelTransformer"
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
