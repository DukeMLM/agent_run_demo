"""V4-exact relational transformer with a shared-renderer PATCH HEAD (improve of r05 = r00_draft2_v4tfshape; ONE
change: the spectral head). Encoder = the incumbent unchanged: 1 global token embedded from (h, p) and 4 resonator
tokens embedded by ONE shared GELU MLP from ellipse shape-tensor invariants s = rx+ry, u, v = (rx-ry)cos/sin
2theta, |d| = sqrt(u^2 + v^2 + eps^2), u/|d|, v/|d|, plus h, p; n_layers pre-LN transformer blocks (d_model, 8
heads, GELU FFN 4x, dropout) with learned per-head attention-logit biases per token-pair relation (i XOR j =
supercell translation) -> exactly V4-invariant; pooled = concat(global, mean resonator token). New head (no
attention): trunk Linear(2d -> head_width) - GELU - Linear(head_width -> P * C) gives P = 32 patch codes of C =
128 channels (+ learned patch position embedding); 2 pre-LN residual GELU MLP blocks (C -> 4C -> C) SHARED by all
patches; LayerNorm; one shared Linear(C -> 2S) renders each patch over 2S = 128 points at stride S = 64 and
neighbouring halves are overlap-added (= ConvTranspose1d, kernel 2S, stride S); centre crop 2112 -> 2001. The
nonlinear line-shape renderer is shared along frequency, so a resonance shape learned in one band is reusable in
every other band, instead of the incumbent's fixed 2001 x 1536 linear output basis. Recipe as the incumbent:
global_standard y, MSE, AdamW (no decay on norms / biases / relation biases / position embedding), time-aware
cosine + warmup, grad clip 1, EMA; batch / lr / dropout / wd / width / depth / EMA scale with n_train in
adapt_config (k=16k: d 320, 4 layers, trunk 1536, batch 256, lr 1.5e-3, wd 0.01, dropout 0.02, EMA 0.99894).

Why: every incumbent run reaches its best val MSE at epoch 40-80 while the train MSE keeps falling, and a 4-member
ensemble of the incumbent (r06_improve_ens4, 1.510e-3 vs 1.722e-3) shows that the seed-independent part of the
error (about 1.43e-3) dominates, so the inductive bias of the model, not its variance, limits it; the MLP head is
the component least suited to sharp, geometry-dependent resonance positions.

Parameterisation identity (exact): an ellipse (rx, ry, theta) equals (ry, rx, theta -+ pi/2); s, u, v are
invariant under this map. Physics prior: shifting the 2x2 supercell by one sub-cell permutes the resonators by a
double transposition i -> i XOR g (Klein four-group V4); attention that sees only the pair relations i XOR j plus
per-token content is V4-equivariant and mean pooling makes the spectrum V4-invariant.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

CONFIG = {
    "epochs": 2600,              # generous cap; the time-aware cosine anneals by max_train_seconds
    "batch_size": 64,
    "lr": 8.5e-4,
    "weight_decay": 0.05,
    "optimizer": "adamw",
    "betas": [0.9, 0.98],
    "scheduler": "cosine",
    "warmup_epochs": 38,
    "min_lr_ratio": 0.01,
    "time_aware_schedule": True,
    "loss": "mse",
    "grad_clip": 1.0,
    "x_scaling": "minmax",       # column permutations of the resonator triples stay exact under minmax
    "y_scaling": "global_standard",
    "amp": False,
    "tf32": True,
    "ema_decay": 0.996,
    "eval_every": 1,
    "patience": 0,
    # ---- encoder (identical to the incumbent)
    "d_model": 256,
    "n_heads": 8,
    "n_layers": 3,
    "ffn_mult": 4,
    "embed_hidden": 256,
    "dropout": 0.1,              # after each FFN branch (encoder and patch blocks)
    "attn_dropout": 0.0,
    "head_width": 1024,          # trunk hidden width
    "head_dropout": 0.1,
    "rel_bias_init": 1.0,        # head k starts with a soft preference for resonator relation k % 4
    # ---- patch head (new)
    "n_patches": 32,
    "patch_stride": 64,          # each patch renders 2 * stride points; (n_patches + 1) * stride >= 2001
    "patch_channels": 128,
    "patch_blocks": 2,
    "patch_ffn_mult": 4,
    # ---- timing model used by adapt_config (incumbent's)
    "est_step_base_s": 0.007,
    "est_step_per_layer_s": 0.003,
    "est_epoch_overhead_s": 0.01,
    "warmup_steps": 300,
}


def adapt_config(cfg, n_train, max_train_seconds):
    """Scale batch / lr / regularisation / capacity / EMA / warmup with n_train and the time budget (incumbent)."""
    n = max(1, int(n_train))
    S = float(max_train_seconds)
    if n < 1500:
        bs, drop, wd, d, L, hw = 64, 0.10, 0.05, 256, 3, 1024
    elif n < 4000:
        bs, drop, wd, d, L, hw = 64, 0.08, 0.03, 256, 3, 1024
    elif n < 12000:
        bs, drop, wd, d, L, hw = 128, 0.05, 0.02, 256, 4, 1024
    elif n < 30000:
        bs, drop, wd, d, L, hw = 256, 0.02, 0.01, 320, 4, 1536
    else:
        bs, drop, wd, d, L, hw = 512, 0.0, 0.01, 384, 6, 1536
    cfg.update(batch_size=bs, dropout=drop, head_dropout=drop, weight_decay=wd,
               d_model=d, n_layers=L, head_width=hw, embed_hidden=d)
    cfg["lr"] = round(min(1.5e-3, 6e-4 * math.sqrt(bs / 32.0)), 6)
    spe = math.ceil(n / bs)
    est_step = float(cfg["est_step_base_s"]) + float(cfg["est_step_per_layer_s"]) * L
    est_epoch = est_step * spe + float(cfg["est_epoch_overhead_s"])
    cfg["epochs"] = int(min(20000, max(100, 2.0 * S / est_epoch)))
    est_steps = 0.5 * (S / est_epoch) * spe
    horizon = min(2000.0, max(100.0, 0.05 * est_steps))
    cfg["ema_decay"] = round(1.0 - 1.0 / horizon, 6)
    cfg["warmup_epochs"] = round(max(1.0, min(0.05 * cfg["epochs"], float(cfg["warmup_steps"]) / spe)), 2)
    return cfg


# ----------------------------------------------------------------------------------------------- model
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


class RelAttention(nn.Module):
    """Multi-head self-attention with a learned per-head additive logit bias per relation type."""

    def __init__(self, d, n_heads, attn_drop, bias_init):
        super().__init__()
        assert d % n_heads == 0, "d_model must be divisible by n_heads"
        self.h, self.dh = n_heads, d // n_heads
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        b = torch.zeros(n_heads, N_REL)
        for k in range(n_heads):
            b[k, k % 4] = bias_init
        self.rel_bias = nn.Parameter(b)
        self.attn_drop = float(attn_drop)

    def forward(self, x, rel):
        B, T, D = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.h, self.dh).permute(2, 0, 3, 1, 4).unbind(0)
        att = (q @ k.transpose(-2, -1)) * (self.dh ** -0.5) + self.rel_bias[:, rel]
        att = att.softmax(dim=-1)
        if self.attn_drop > 0:
            att = F.dropout(att, self.attn_drop, self.training)
        y = (att @ v).transpose(1, 2).reshape(B, T, D)
        return self.proj(y)


class Block(nn.Module):
    def __init__(self, d, n_heads, ffn_mult, drop, attn_drop, bias_init):
        super().__init__()
        self.n1 = nn.LayerNorm(d)
        self.att = RelAttention(d, n_heads, attn_drop, bias_init)
        self.n2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, ffn_mult * d), nn.GELU(), nn.Linear(ffn_mult * d, d))
        self.drop = nn.Dropout(drop)

    def forward(self, x, rel):
        x = x + self.att(self.n1(x), rel)
        return x + self.drop(self.ff(self.n2(x)))


class PatchBlock(nn.Module):
    """Pre-LN residual GELU MLP applied to every patch code with the same weights."""

    def __init__(self, c, mult, drop):
        super().__init__()
        self.n = nn.LayerNorm(c)
        self.ff = nn.Sequential(nn.Linear(c, mult * c), nn.GELU(), nn.Linear(mult * c, c))
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        return x + self.drop(self.ff(self.n(x)))


def _mlp(i, h, o):
    return nn.Sequential(nn.Linear(i, h), nn.GELU(), nn.Linear(h, o))


class V4RelTransformerPatchHead(nn.Module):
    N_RES_FEAT = 8
    N_GLOB_FEAT = 2
    EPS = 0.1   # smoothing of |d| = |rx - ry| in scaled units (d_scaled in [-2, 2], grid step 0.25)

    def __init__(self, in_dim, out_dim, cfg):
        super().__init__()
        assert in_dim == 14, "expects [h, p, rx1..4, ry1..4, theta1..4]"
        d, eh, hw = int(cfg["d_model"]), int(cfg["embed_hidden"]), int(cfg["head_width"])
        self.P, self.S, self.C = int(cfg["n_patches"]), int(cfg["patch_stride"]), int(cfg["patch_channels"])
        total = (self.P + 1) * self.S
        assert total >= out_dim, "(n_patches + 1) * patch_stride must cover the spectrum"
        self.out_dim = out_dim
        self.crop0 = (total - out_dim) // 2
        self.res_embed = _mlp(self.N_RES_FEAT, eh, d)     # shared by the 4 resonators
        self.glob_embed = _mlp(self.N_GLOB_FEAT, eh, d)
        self.blocks = nn.ModuleList([
            Block(d, int(cfg["n_heads"]), int(cfg["ffn_mult"]), float(cfg["dropout"]),
                  float(cfg["attn_dropout"]), float(cfg["rel_bias_init"]))
            for _ in range(int(cfg["n_layers"]))])
        self.norm = nn.LayerNorm(d)
        hd = float(cfg["head_dropout"])
        self.trunk = nn.Sequential(nn.Linear(2 * d, hw), nn.GELU(), nn.Dropout(hd),
                                   nn.Linear(hw, self.P * self.C))
        self.patch_pos = nn.Parameter(torch.randn(1, self.P, self.C) * 0.02)
        self.patch_blocks = nn.ModuleList([
            PatchBlock(self.C, int(cfg["patch_ffn_mult"]), float(cfg["dropout"]))
            for _ in range(int(cfg["patch_blocks"]))])
        self.patch_norm = nn.LayerNorm(self.C)
        self.render = nn.Linear(self.C, 2 * self.S)       # shared by all patches
        self.register_buffer("rel", relation_matrix())

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
        B = x.shape[0]
        res, glob = self.features(x)
        tok = torch.cat([self.glob_embed(glob)[:, None, :], self.res_embed(res)], dim=1)  # (B, 5, d)
        for blk in self.blocks:
            tok = blk(tok, self.rel)
        tok = self.norm(tok)
        pooled = torch.cat([tok[:, 0], tok[:, 1:].mean(dim=1)], dim=-1)
        z = self.trunk(pooled).view(B, self.P, self.C) + self.patch_pos             # (B, P, C)
        for blk in self.patch_blocks:
            z = blk(z)
        r = self.render(self.patch_norm(z))                                         # (B, P, 2S)
        S = self.S
        first, second = r[..., :S], r[..., S:]                                      # overlap-add, stride S
        y = F.pad(first, (0, 0, 0, 1)) + F.pad(second, (0, 0, 1, 0))                # (B, P + 1, S)
        y = y.reshape(B, (self.P + 1) * S)
        return y[:, self.crop0:self.crop0 + self.out_dim]


def build_model(in_dim, out_dim, cfg):
    return V4RelTransformerPatchHead(in_dim, out_dim, cfg)


def make_optimizer(model, cfg):
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if (p.ndim < 2 or "rel_bias" in name or "patch_pos" in name) else decay).append(p)
    groups = [{"params": decay, "weight_decay": float(cfg["weight_decay"])},
              {"params": no_decay, "weight_decay": 0.0}]
    kw = dict(lr=float(cfg["lr"]), betas=tuple(cfg["betas"]))
    try:
        return torch.optim.AdamW(groups, fused=True, **kw)
    except Exception:  # noqa: BLE001 - fused kernel unavailable
        return torch.optim.AdamW(groups, **kw)
