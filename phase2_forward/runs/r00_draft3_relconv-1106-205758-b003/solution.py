"""Pairwise-relational V4-invariant encoder + hybrid structured spectral decoder (learned low-rank basis head
plus a FiLM-conditioned 1-D transposed-convolution decoder with a sub-pixel output head). Improve step of
r00_draft3_symconv: only the encoder changes (the orbit-averaged 14-input MLP is replaced). Encoder: each
resonator i becomes a token [rx_i, ry_i, theta_i, h, p] embedded by one shared MLP phi (Linear-GELU, 2
pre-LN residual GELU blocks, LayerNorm) -> u_i; for each of the three pair partitions that the sub-cell
translations V4 = {e, (12)(34), (13)(24), (14)(23)} preserve ({12|34}, {13|24}, {14|23}) a shared pair MLP
psi acts on the symmetric pair feature [u_a + u_b, u_a * u_b] plus a learned partition-type embedding and is
summed over the partition's 2 pairs (Q_A, Q_B, Q_C); rho = Linear([sum_i u_i, Q_A, Q_B, Q_C, h, p] -> d),
GELU, pre-LN residual GELU MLP blocks, LayerNorm -> latent z. The spectrum is therefore exactly V4-invariant
(single resonator effects shared across the 4 sites, couplings modelled per partition). Decoder output = (a)
learned basis head z -> basis_rank coefficients -> Linear(basis_rank -> 2001) plus (b) convolutional
renderer: Linear z -> (C0 x 32) seed map, residual Conv1d block, 3 stages of [ConvTranspose1d(k=4, stride
2), FiLM from z, residual Conv1d block (GroupNorm-GELU-Conv(k=5)-GELU-Conv(k=5))] to length 256,
GroupNorm-GELU, zero-initialised Conv1d(C -> 8) interleaved (sub-pixel) into 2048 points, centre-cropped to
2001. GELU everywhere. Training: MSE on globally standardised absorptivity, fused AdamW (weight decay 1e-2
on matrices / kernels only), harness time-aware cosine LR (warmup ~300 steps, 1 % floor), grad clip 1.0, no
dropout / EMA, batch 64 at k=500; adapt_config scales batch (64 -> 512), lr (~sqrt(batch), cap 2e-3),
widths / depths / decoder channels, bf16 autocast (n >= 15k) and a generous epoch cap (wall clock ends
training)."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

CONFIG = {
    "epochs": 3000,
    "batch_size": 64,
    "lr": 1.4e-3,
    "weight_decay": 1e-2,
    "optimizer": "adamw",
    "scheduler": "cosine",
    "warmup_epochs": 10,
    "min_lr_ratio": 0.01,
    "time_aware_schedule": True,
    "loss": "mse",
    "grad_clip": 1.0,
    "x_scaling": "minmax",
    "y_scaling": "global_standard",
    "amp": False,
    "tf32": True,
    "ema_decay": 0.0,
    "eval_every": 1,
    "patience": 0,
    "seed": 0,
    # ---- architecture (custom keys; the logged cfg fully determines the structure)
    "d_tok": 256,              # resonator token width (phi)
    "tok_blocks": 2,
    "d_pair": 256,             # pair feature width (psi)
    "pair_blocks": 1,
    "d_model": 512,            # rho width / latent size
    "enc_blocks": 2,           # rho residual MLP blocks
    "enc_mult": 2,             # hidden expansion of every residual MLP block
    "basis_rank": 64,          # learned spectral basis vectors
    "seed_len": 32,            # length of the coarse seed map (x2 per transposed-conv stage)
    "dec_channels": [128, 96, 64, 64],   # channels at lengths 32, 64, 128, 256
    "subpixel": 8,             # output points per position of the last map (256 x 8 = 2048)
    "kernel": 5,
}


# ------------------------------------------------------------------------------------------ config
def adapt_config(cfg, n_train, max_train_seconds):
    """Scale batch / lr / capacity / epoch cap with the data set size so the same file stays sensible
    for every k from 500 to 50,000 (the time-aware cosine anneals by max_train_seconds anyway)."""
    n = max(1, int(n_train))
    S = float(max_train_seconds)
    if n <= 3000:
        bs = 64
    elif n <= 8000:
        bs = 128
    elif n <= 25000:
        bs = 256
    else:
        bs = 512
    cfg["batch_size"] = bs
    cfg["lr"] = min(2e-3, 1.4e-3 * math.sqrt(bs / 64.0))
    if n <= 3000:
        dt, tb, dp, pb, d, nb, ch = 256, 2, 256, 1, 512, 2, [128, 96, 64, 64]
    elif n <= 15000:
        dt, tb, dp, pb, d, nb, ch = 384, 2, 384, 1, 768, 3, [160, 128, 96, 96]
    else:
        dt, tb, dp, pb, d, nb, ch = 512, 3, 512, 2, 1024, 4, [192, 160, 128, 128]
    cfg.update({"d_tok": dt, "tok_blocks": tb, "d_pair": dp, "pair_blocks": pb, "d_model": d,
                "enc_blocks": nb, "dec_channels": ch})
    cfg["weight_decay"] = 1e-2 if n <= 3000 else 5e-3
    cfg["amp"] = n >= 15000
    steps = math.ceil(n / bs)
    cfg["warmup_epochs"] = int(max(1, min(10, round(300 / steps))))
    # rough solo-GPU cost model (per-step launch overhead + size term); the epoch cap is generous (2x) so
    # the wall clock, not the cap, ends training and drives the time-aware cosine schedule.
    width = (d / 512.0) ** 2 * (nb / 2.0)
    t_step = 0.012 + 1.5e-5 * bs * width * (0.6 if cfg["amp"] else 1.0)
    t_epoch = steps * t_step + 0.01 + 2e-6 * n / 10.0
    cfg["epochs"] = int(min(20000, max(50, 2.0 * S / t_epoch)))
    return cfg


# ------------------------------------------------------------------------------------------ geometry
# resonator i (0-based) = columns (2+i, 6+i, 10+i); h, p = columns 0, 1
_TOKEN_COLS = [[2 + i, 6 + i, 10 + i, 0, 1] for i in range(4)]
# the three pair partitions preserved by V4 (pairs listed partition by partition: A1, A2, B1, B2, C1, C2)
_PAIRS = ((0, 1), (2, 3), (0, 2), (1, 3), (0, 3), (1, 2))


# ------------------------------------------------------------------------------------------ blocks
class ResMLP(nn.Module):
    def __init__(self, d, mult):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.fc1 = nn.Linear(d, d * mult)
        self.fc2 = nn.Linear(d * mult, d)

    def forward(self, h):
        return h + self.fc2(F.gelu(self.fc1(self.norm(h))))


class RelationalEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        dt, dp, d, mult = int(cfg["d_tok"]), int(cfg["d_pair"]), int(cfg["d_model"]), int(cfg["enc_mult"])
        self.register_buffer("tok_idx", torch.tensor(_TOKEN_COLS, dtype=torch.long), persistent=False)
        self.register_buffer("pa", torch.tensor([a for a, _ in _PAIRS], dtype=torch.long), persistent=False)
        self.register_buffer("pb", torch.tensor([b for _, b in _PAIRS], dtype=torch.long), persistent=False)
        self.tok_in = nn.Linear(5, dt)
        self.tok_blocks = nn.ModuleList([ResMLP(dt, mult) for _ in range(int(cfg["tok_blocks"]))])
        self.tok_norm = nn.LayerNorm(dt)
        self.pair_in = nn.Linear(2 * dt, dp)
        self.pair_type = nn.Parameter(0.02 * torch.randn(3, dp))
        self.pair_blocks = nn.ModuleList([ResMLP(dp, mult) for _ in range(int(cfg["pair_blocks"]))])
        self.pair_norm = nn.LayerNorm(dp)
        self.rho_in = nn.Linear(dt + 3 * dp + 2, d)
        self.blocks = nn.ModuleList([ResMLP(d, mult) for _ in range(int(cfg["enc_blocks"]))])
        self.norm = nn.LayerNorm(d)

    def forward(self, x):
        b = x.shape[0]
        u = F.gelu(self.tok_in(x[:, self.tok_idx]))          # (B, 4, dt): shared resonator embedding
        for blk in self.tok_blocks:
            u = blk(u)
        u = self.tok_norm(u)
        ua, ub = u[:, self.pa], u[:, self.pb]                 # (B, 6, dt)
        q = self.pair_in(torch.cat([ua + ub, ua * ub], dim=-1))           # symmetric in (a, b)
        q = F.gelu(q + self.pair_type.repeat_interleave(2, dim=0))       # partition-type embedding
        for blk in self.pair_blocks:
            q = blk(q)
        q = self.pair_norm(q).view(b, 3, 2, -1).sum(2).reshape(b, -1)    # sum over each partition's pairs
        h = F.gelu(self.rho_in(torch.cat([u.sum(1), q, x[:, :2]], dim=-1)))
        for blk in self.blocks:
            h = blk(h)
        return self.norm(h)


class ConvRes(nn.Module):
    """Pre-norm residual 1-D conv block: h + Conv(GELU(Conv(GELU(GroupNorm(h)))))."""

    def __init__(self, c, k):
        super().__init__()
        self.norm = nn.GroupNorm(1, c)
        self.c1 = nn.Conv1d(c, c, k, padding=k // 2)
        self.c2 = nn.Conv1d(c, c, k, padding=k // 2)

    def forward(self, h):
        return h + self.c2(F.gelu(self.c1(F.gelu(self.norm(h)))))


class UpStage(nn.Module):
    """ConvTranspose1d(k=4, stride 2) x2 upsampling (cin->cout) -> FiLM(z) -> residual conv block."""

    def __init__(self, cin, cout, dz, k):
        super().__init__()
        self.up = nn.ConvTranspose1d(cin, cout, 4, stride=2, padding=1)
        self.film = nn.Linear(dz, 2 * cout)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)
        self.res = ConvRes(cout, k)

    def forward(self, h, z):
        h = self.up(h)
        g, b = self.film(z).unsqueeze(-1).chunk(2, dim=1)
        return self.res(h * (1.0 + g) + b)


class ConvSpectralDecoder(nn.Module):
    def __init__(self, dz, chans, seed_len, k, subpixel, out_dim):
        super().__init__()
        self.c0, self.l0, self.sub = int(chans[0]), int(seed_len), int(subpixel)
        self.seed = nn.Linear(dz, self.c0 * self.l0)
        self.block0 = ConvRes(self.c0, k)
        self.stages = nn.ModuleList([UpStage(chans[i], chans[i + 1], dz, k) for i in range(len(chans) - 1)])
        self.norm = nn.GroupNorm(1, chans[-1])
        self.out = nn.Conv1d(chans[-1], self.sub, k, padding=k // 2)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        length = self.l0 * 2 ** (len(chans) - 1) * self.sub
        if length < out_dim:
            raise ValueError(f"decoder length {length} < out_dim {out_dim}")
        self.off, self.out_dim = (length - out_dim) // 2, int(out_dim)

    def forward(self, z):
        b = z.shape[0]
        h = self.block0(self.seed(z).view(b, self.c0, self.l0))
        for st in self.stages:
            h = st(h, z)
        y = self.out(F.gelu(self.norm(h)))              # (B, sub, L)
        y = y.transpose(1, 2).reshape(b, -1)            # sub-pixel shuffle: point = sub * l + s
        return y[:, self.off:self.off + self.out_dim]


class RelConvNet(nn.Module):
    def __init__(self, in_dim, out_dim, cfg):
        super().__init__()
        if in_dim != 14:
            raise ValueError("RelConvNet expects the 14 ADM geometry parameters")
        d = int(cfg["d_model"])
        self.enc = RelationalEncoder(cfg)
        r = int(cfg["basis_rank"])
        self.coef = nn.Linear(d, r)
        self.basis = nn.Linear(r, out_dim)            # learned basis vectors (columns) + per-point bias
        self.dec = ConvSpectralDecoder(d, list(cfg["dec_channels"]), int(cfg["seed_len"]), int(cfg["kernel"]),
                                       int(cfg.get("subpixel", 8)), out_dim)

    def forward(self, x):
        z = self.enc(x)
        return self.basis(self.coef(z)) + self.dec(z)


def build_model(in_dim, out_dim, cfg):
    return RelConvNet(in_dim, out_dim, cfg)


# ------------------------------------------------------------------------------------------ optimizer
def make_optimizer(model, cfg):
    decay, no_decay = [], []
    for p in model.parameters():
        if p.requires_grad:
            (decay if p.dim() >= 2 else no_decay).append(p)
    groups = [{"params": decay, "weight_decay": float(cfg["weight_decay"])},
              {"params": no_decay, "weight_decay": 0.0}]
    kw = dict(lr=float(cfg["lr"]), betas=tuple(cfg.get("betas", (0.9, 0.999))))
    try:
        return torch.optim.AdamW(groups, fused=True, **kw)
    except (RuntimeError, TypeError, ValueError):
        return torch.optim.AdamW(groups, **kw)
