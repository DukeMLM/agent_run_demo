"""Deep ensemble (improve of r05 = r00_draft2_v4tfshape; ONE change: M = n_members independently initialised
copies of the incumbent V4-exact relational transformer, trained side by side in one module and averaged).
Member = the incumbent unchanged: 1 global token embedded from (h, p) and 4 resonator tokens embedded by ONE
shared GELU MLP from ellipse shape-tensor invariants s = rx+ry, u, v = (rx-ry)cos/sin 2theta, |d| = sqrt(u^2 +
v^2 + eps^2), u/|d|, v/|d|, plus h, p; n_layers pre-LN transformer blocks (d_model, 8 heads, GELU FFN 4x,
dropout) with learned per-head attention-logit biases per token-pair relation (i XOR j = supercell translation)
-> exactly V4-invariant; head concat(global, mean resonator token) -> 2x Linear-GELU (head_width) ->
Linear(2001). Members are vectorised (every Linear / LayerNorm holds an (M, ...) weight stack and runs as one
batched matmul), see different random inits and dropout masks, and are trained on the mean of their
individual MSEs (independent deep ensemble, no co-adaptation); forward() returns the member mean (eval and
Phase 3). Recipe as the incumbent: global_standard y, MSE, AdamW (no decay on norms / biases / relation
biases), time-aware cosine + warmup, EMA, batch / lr / dropout / wd / width / depth / EMA scaled with n_train
in adapt_config (k=16k: d 320, 4 layers, head 1536, batch 256, lr 1.5e-3, wd 0.01, dropout 0.02, EMA 0.99894);
global grad clip 1/sqrt(M) (= clip 1 per member when member gradients are of similar size).

Why: every incumbent run (k = 4k, 8k, 16k) reaches its best validation MSE at epoch 40-70 of 400-850 while the
train MSE keeps falling (r05: 2.3e-4 train vs 1.72e-3 val), i.e. the error is variance / memorisation, not
optimisation, and about 90 % of the time budget is unused. Averaging M members trained on that budget cuts the
seed-dependent part of the error by about (1 - 1/M).

Parameterisation identity (exact): an ellipse (rx, ry, theta) equals (ry, rx, theta -+ pi/2); s, u, v are
invariant under this map. Physics prior: shifting the 2x2 supercell by one sub-cell permutes the resonators by
a double transposition i -> i XOR g (Klein four-group V4); attention that sees only the pair relations i XOR j
plus per-token content is V4-equivariant and mean pooling makes the spectrum V4-invariant.
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
    "grad_clip": 0.5,            # global clip 1/sqrt(M) on the mean-of-members loss (set in adapt_config)
    "x_scaling": "minmax",       # column permutations of the resonator triples stay exact under minmax
    "y_scaling": "global_standard",
    "amp": False,
    "tf32": True,
    "ema_decay": 0.996,
    "eval_every": 1,
    "patience": 0,
    # ---- ensemble
    "n_members": 4,
    "member_grad_clip": 1.0,     # per-member clip of the incumbent; mapped to grad_clip = this / sqrt(M)
    # ---- model (per member, identical to the incumbent)
    "d_model": 256,
    "n_heads": 8,
    "n_layers": 3,
    "ffn_mult": 4,
    "embed_hidden": 256,
    "dropout": 0.1,              # after each FFN branch
    "attn_dropout": 0.0,
    "head_width": 1024,
    "head_dropout": 0.1,
    "rel_bias_init": 1.0,        # head k starts with a soft preference for resonator relation k % 4
    # ---- single-member timing model of the incumbent (drives the EMA horizon exactly as in r05)
    "est_step_base_s": 0.007,
    "est_step_per_layer_s": 0.003,
    "est_epoch_overhead_s": 0.01,
    "warmup_steps": 300,
}


def adapt_config(cfg, n_train, max_train_seconds):
    """Incumbent scaling of batch / lr / regularisation / capacity / EMA / warmup with n_train and the budget;
    the epoch cap stays the single-member one (the ensemble runs fewer epochs, so the wall clock drives the
    cosine), and the EMA horizon is the incumbent's (same number of optimizer steps per member)."""
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
    M = int(cfg["n_members"])
    cfg["grad_clip"] = round(float(cfg["member_grad_clip"]) / math.sqrt(M), 6)
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


class BLinear(nn.Module):
    """M independent Linear layers: (M, N, i) or shared (N, i) -> (M, N, o). nn.Linear default init per member."""

    def __init__(self, M, i, o):
        super().__init__()
        bound = 1.0 / math.sqrt(i)
        self.weight = nn.Parameter(torch.empty(M, i, o).uniform_(-bound, bound))
        self.bias = nn.Parameter(torch.empty(M, 1, o).uniform_(-bound, bound))

    def forward(self, x):
        return torch.matmul(x, self.weight) + self.bias


class BLayerNorm(nn.Module):
    """M independent LayerNorms over the last dim of (M, N, d)."""

    def __init__(self, M, d):
        super().__init__()
        self.d = d
        self.ln_w = nn.Parameter(torch.ones(M, 1, d))
        self.ln_b = nn.Parameter(torch.zeros(M, 1, d))

    def forward(self, x):
        return F.layer_norm(x, (self.d,)) * self.ln_w + self.ln_b


class BRelAttention(nn.Module):
    """Per-member multi-head self-attention with a learned per-head logit bias per relation type."""

    def __init__(self, M, d, n_heads, attn_drop, bias_init):
        super().__init__()
        assert d % n_heads == 0, "d_model must be divisible by n_heads"
        self.M, self.h, self.dh = M, n_heads, d // n_heads
        self.qkv = BLinear(M, d, 3 * d)
        self.proj = BLinear(M, d, d)
        b = torch.zeros(M, n_heads, N_REL)
        for k in range(n_heads):
            b[:, k, k % 4] = bias_init
        self.rel_bias = nn.Parameter(b)
        self.attn_drop = float(attn_drop)

    def forward(self, x, rel, B, T):
        M, h, dh = self.M, self.h, self.dh
        q, k, v = self.qkv(x).view(M, B, T, 3, h, dh).permute(3, 0, 1, 4, 2, 5).unbind(0)  # (M, B, h, T, dh)
        att = (q @ k.transpose(-2, -1)) * (dh ** -0.5) + self.rel_bias[:, :, rel].unsqueeze(1)
        att = att.softmax(dim=-1)
        if self.attn_drop > 0:
            att = F.dropout(att, self.attn_drop, self.training)
        y = (att @ v).permute(0, 1, 3, 2, 4).reshape(M, B * T, h * dh)
        return self.proj(y)


class BBlock(nn.Module):
    def __init__(self, M, d, n_heads, ffn_mult, drop, attn_drop, bias_init):
        super().__init__()
        self.n1 = BLayerNorm(M, d)
        self.att = BRelAttention(M, d, n_heads, attn_drop, bias_init)
        self.n2 = BLayerNorm(M, d)
        self.ff1 = BLinear(M, d, ffn_mult * d)
        self.ff2 = BLinear(M, ffn_mult * d, d)
        self.drop = nn.Dropout(drop)

    def forward(self, x, rel, B, T):
        x = x + self.att(self.n1(x), rel, B, T)
        return x + self.drop(self.ff2(F.gelu(self.ff1(self.n2(x)))))


class EnsembleV4RelTransformer(nn.Module):
    N_RES_FEAT = 8
    N_GLOB_FEAT = 2
    EPS = 0.1   # smoothing of |d| = |rx - ry| in scaled units (d_scaled in [-2, 2], grid step 0.25)

    def __init__(self, in_dim, out_dim, cfg):
        super().__init__()
        assert in_dim == 14, "expects [h, p, rx1..4, ry1..4, theta1..4]"
        M = self.M = int(cfg["n_members"])
        d, eh, hw = int(cfg["d_model"]), int(cfg["embed_hidden"]), int(cfg["head_width"])
        self.d = d
        self.res1, self.res2 = BLinear(M, self.N_RES_FEAT, eh), BLinear(M, eh, d)   # shared by the 4 resonators
        self.glob1, self.glob2 = BLinear(M, self.N_GLOB_FEAT, eh), BLinear(M, eh, d)
        self.blocks = nn.ModuleList([
            BBlock(M, d, int(cfg["n_heads"]), int(cfg["ffn_mult"]), float(cfg["dropout"]),
                   float(cfg["attn_dropout"]), float(cfg["rel_bias_init"]))
            for _ in range(int(cfg["n_layers"]))])
        self.norm = BLayerNorm(M, d)
        self.head1, self.head2, self.head3 = BLinear(M, 2 * d, hw), BLinear(M, hw, hw), BLinear(M, hw, out_dim)
        self.head_drop = nn.Dropout(float(cfg["head_dropout"]))
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

    def forward_members(self, x):
        """(B, 14) -> (M, B, out_dim): every member's prediction."""
        B, M, d = x.shape[0], self.M, self.d
        res, glob = self.features(x)
        r = self.res2(F.gelu(self.res1(res.reshape(B * 4, self.N_RES_FEAT)))).view(M, B, 4, d)
        g = self.glob2(F.gelu(self.glob1(glob))).view(M, B, 1, d)
        tok = torch.cat([g, r], dim=2).reshape(M, B * 5, d)
        for blk in self.blocks:
            tok = blk(tok, self.rel, B, 5)
        tok = self.norm(tok).view(M, B, 5, d)
        pooled = torch.cat([tok[:, :, 0], tok[:, :, 1:].mean(dim=2)], dim=-1)    # (M, B, 2d)
        z = self.head_drop(F.gelu(self.head1(pooled)))
        z = self.head_drop(F.gelu(self.head2(z)))
        return self.head3(z)

    def forward(self, x):
        return self.forward_members(x).mean(dim=0)


def build_model(in_dim, out_dim, cfg):
    return EnsembleV4RelTransformer(in_dim, out_dim, cfg)


def compute_loss(model, xb, yb, cfg):
    """Mean over members of each member's MSE (independent members; equals the plain MSE for M = 1)."""
    pred = model.forward_members(xb).float()
    return ((pred - yb.unsqueeze(0)) ** 2).mean()


def make_optimizer(model, cfg):
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_weight = name.endswith(".weight") and "ln_" not in name
        (decay if is_weight else no_decay).append(p)
    groups = [{"params": decay, "weight_decay": float(cfg["weight_decay"])},
              {"params": no_decay, "weight_decay": 0.0}]
    kw = dict(lr=float(cfg["lr"]), betas=tuple(cfg["betas"]))
    try:
        return torch.optim.AdamW(groups, fused=True, **kw)
    except Exception:  # noqa: BLE001 - fused kernel unavailable
        return torch.optim.AdamW(groups, **kw)
