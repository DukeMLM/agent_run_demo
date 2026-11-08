"""Refine of r06_explore_patchxattn_s32 with ONE hyperparameter change: no dropout (k = 16,000: residual-branch
dropout 0 instead of 0.1 in encoder and decoder; adapt_config dropout 0.10 for n < 1,500, 0.05 for n < 4,000 and 0
above). Everything else identical (batch 256, AdamW lr 1e-3, wd 0.05, EMA, time-aware cosine, bf16).
Architecture unchanged (S = 32 spectral patches, P = 63 patch query tokens, renderer Linear(d -> 64) overlap-added
with stride 32). Encoder-decoder geometry-to-spectrum transformer with spectral patch queries (Perceiver-IO / DETR-style attention
decoder instead of an MLP head). Encoder: 1 global (h, p) token + 4 resonator tokens (shared GELU-MLP embedding of
ellipse shape-tensor invariants s, u, v = (rx-ry)cos/sin 2theta, |d|, u/|d|, v/|d|, h, p), 2 pre-LN blocks with a
learned per-head attention-logit bias per V4 relation (i XOR j): exactly invariant to supercell-shift permutations.
Decoder: P = ceil(2001 / S) patch query tokens (learned position embedding + projection of the pooled geometry
latent) run 4 pre-LN blocks of [patch self-attention with a learned relative-position bias (ALiBi-initialised
locality in 4 of 8 heads), cross-attention to the 5 geometry tokens, GELU FFN 4x]; one shared Linear(d -> 2S)
renders every patch, neighbouring halves are overlap-added (= ConvTranspose1d kernel 2S, stride S), centre-cropped
to 2001, plus a per-point bias. d 256, 8 heads. Recipe: global_standard y, MSE, random all-angle sign-flip
augmentation (mirror symmetry), AdamW lr 1e-3 / batch 256 / wd 0.05 (none on norms, biases, position tables),
betas (0.9, 0.98), time-aware cosine with ~500 warmup steps, grad clip 1, EMA (horizon ~5 % of the expected steps),
dropout 0.1, bf16 autocast; adapt_config scales batch / lr / width / depth / EMA / epoch cap with n_train.

Why: the labels are essentially noiseless (duplicate-pair floor 1.3e-5) and the remaining error sits at sharp
resonances whose positions must be predicted precisely; finer patches (S 64 -> 32) helped, which points to a
resolution / bias limit rather than variance. Residual dropout injects noise into all 63 patch tokens of every
block (train loss 2.5e-3 scaled at the end of the parent), slows fitting in the fixed 720 s budget and costs
about 6 % of the step time.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

CONFIG = {
    "epochs": 1200,              # generous cap; the time-aware cosine anneals by max_train_seconds
    "batch_size": 256,
    "lr": 1e-3,
    "weight_decay": 0.05,
    "optimizer": "adamw",
    "betas": [0.9, 0.98],
    "scheduler": "cosine",
    "warmup_epochs": 8.8,        # overwritten by adapt_config (about warmup_steps optimizer steps)
    "min_lr_ratio": 0.01,
    "time_aware_schedule": True,
    "loss": "mse",
    "grad_clip": 1.0,
    "x_scaling": "minmax",       # theta_scaled = theta / (pi/4); rx_i, ry_i share one affine map
    "y_scaling": "global_standard",
    "amp": True,                 # bf16 autocast for training (the harness evaluates in fp32)
    "tf32": True,
    "ema_decay": 0.9988,
    "eval_every": 1,
    "patience": 0,
    # ---- model
    "d_model": 256,
    "n_heads": 8,
    "enc_layers": 2,
    "dec_layers": 4,
    "ffn_mult": 4,
    "patch": 32,                 # S: points per patch (overlap-add kernel 2S, stride S) -> 63 patch tokens
    "dropout": 0.0,              # residual-branch dropout in encoder and decoder (parent: 0.1)
    "rel_bias_init": 1.0,        # encoder head k starts with a soft preference for resonator relation k % 4
    "alibi_heads": 4,            # decoder self-attention heads with ALiBi-initialised locality bias
    "out_init_std": 0.01,
    # ---- training
    "flip_aug": True,            # negate all four angles with p = 0.5 per sample (mirror symmetry)
    "warmup_steps": 500,
    # ---- timing model used by adapt_config (solo GPU; launch-overhead bound at this size)
    "est_step_base_s": 0.008,
    "est_step_per_layer_s": 0.0025,      # per (decoder layer + 0.5 encoder layer) at 8192 patch tokens, d 256
    "est_epoch_overhead_s": 0.05,
}


def adapt_config(cfg, n_train, max_train_seconds):
    """Scale batch / lr / regularisation / capacity / EMA / warmup / epoch cap with n_train and the budget."""
    n = max(1, int(n_train))
    S = float(max_train_seconds)
    # dropout 0 from n = 4,000 on (parent: 0.10 up to n < 30,000, 0.05 above); all else as the parent
    if n < 1500:
        bs, drop, wd, d, le, ld = 64, 0.10, 0.05, 192, 2, 3
    elif n < 4000:
        bs, drop, wd, d, le, ld = 128, 0.05, 0.05, 256, 2, 3
    elif n < 30000:
        bs, drop, wd, d, le, ld = 256, 0.0, 0.05, 256, 2, 4
    else:
        bs, drop, wd, d, le, ld = 512, 0.0, 0.03, 320, 3, 6
    cfg.update(batch_size=bs, dropout=drop, weight_decay=wd, d_model=d, enc_layers=le, dec_layers=ld)
    cfg["lr"] = round(min(1.4e-3, 1e-3 * math.sqrt(bs / 256.0)), 6)
    spe = math.ceil(n / bs)
    P = math.ceil(2001 / int(cfg["patch"]))
    work = max(1.0, (bs * P) / 8192.0) * (d / 256.0) ** 2
    est_step = float(cfg["est_step_base_s"]) + float(cfg["est_step_per_layer_s"]) * (ld + 0.5 * le) * work
    est_epoch = est_step * spe + float(cfg["est_epoch_overhead_s"])
    # epoch cap = 2x the solo estimate, so the wall clock (time-aware cosine) drives the schedule
    cfg["epochs"] = int(min(20000, max(60, 2.0 * S / est_epoch)))
    # EMA horizon about 5 % of the steps expected under GPU sharing, within [100, 2000] steps
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


def compute_loss(model, xb, yb, cfg):
    if cfg.get("flip_aug", False):
        # mirror symmetry: negating all four angles leaves the spectrum unchanged (theta shift is 0 under minmax)
        sgn = torch.where(torch.rand(xb.shape[0], 1, device=xb.device) < 0.5, -1.0, 1.0).to(xb.dtype)
        xb = torch.cat([xb[:, :10], xb[:, 10:14] * sgn], dim=1)
    return F.mse_loss(model(xb).float(), yb)


def make_optimizer(model, cfg):
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim < 2 or "rel_bias" in name or "rel_pos" in name or name == "pos" or "out_bias" in name:
            no_decay.append(p)
        else:
            decay.append(p)
    groups = [{"params": decay, "weight_decay": float(cfg["weight_decay"])},
              {"params": no_decay, "weight_decay": 0.0}]
    kw = dict(lr=float(cfg["lr"]), betas=tuple(cfg["betas"]))
    try:
        return torch.optim.AdamW(groups, fused=True, **kw)
    except Exception:  # noqa: BLE001 - fused kernel unavailable
        return torch.optim.AdamW(groups, **kw)
