"""V4-symmetrised residual-MLP encoder + hybrid structured spectral decoder (learned low-rank basis head plus
a FiLM-conditioned 1-D transposed-convolution decoder with a sub-pixel output head). Encoder: the 14
minmax-scaled inputs are permuted by the 4 elements of the Klein group V4 = {e, (12)(34), (13)(24), (14)(23)}
acting on the resonator triples (rx_i, ry_i, theta_i) (sub-cell translations of the periodic 2x2 supercell);
every copy passes Linear(14->d)-GELU, `enc_blocks` pre-LayerNorm residual GELU MLP blocks (hidden
enc_mult*d) and a LayerNorm, and the 4 latents are averaged, so the spectrum is exactly V4-invariant.
Decoder output = (a) learned basis head: z -> basis_rank coefficients -> Linear(basis_rank -> 2001) (learned
basis vectors + per-point bias) plus (b) convolutional renderer of local resonance detail: Linear z ->
(C0 x 32) seed map, pre-norm residual Conv1d block (GroupNorm-GELU-Conv(k=5)-GELU-Conv(k=5)), then 3 stages
of [ConvTranspose1d(k=4, stride 2) upsampling, FiLM scale/shift from z, residual Conv1d block] up to length
256, GroupNorm-GELU and a zero-initialised Conv1d(C -> 8, k=5) whose 8 channels are interleaved
(sub-pixel shuffle) into 2048 points, centre-cropped to 2001. GELU everywhere (smooth input gradients for
Neural-Adjoint). Training: MSE on globally standardised absorptivity, fused AdamW (weight decay 1e-2 only on
matrices / kernels), harness time-aware cosine LR (warmup ~300 steps, 1 % floor), grad clip 1.0, no
dropout / EMA, batch 64 at k=500; adapt_config scales batch (64 -> 512), lr (~sqrt(batch), cap 2e-3),
width / depth / decoder channels, bf16 autocast (n >= 15k) and a generous epoch cap with n_train and the
time budget (the wall clock ends training)."""
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
    "sym_v4": True,            # average the latent over the 4 resonator double transpositions
    "d_model": 512,            # encoder width / latent size
    "enc_blocks": 3,           # residual MLP blocks
    "enc_mult": 2,             # hidden expansion of each block
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
        d, nb, ch = 512, 3, [128, 96, 64, 64]
    elif n <= 15000:
        d, nb, ch = 768, 4, [160, 128, 96, 96]
    else:
        d, nb, ch = 1024, 6, [192, 160, 128, 128]
    cfg["d_model"], cfg["enc_blocks"], cfg["dec_channels"] = d, nb, ch
    cfg["weight_decay"] = 1e-2 if n <= 3000 else 5e-3
    cfg["amp"] = n >= 15000
    steps = math.ceil(n / bs)
    cfg["warmup_epochs"] = int(max(1, min(10, round(300 / steps))))
    # rough solo-GPU cost model (per-step launch overhead + size term); the epoch cap is generous (2x) so
    # the wall clock, not the cap, ends training and drives the time-aware cosine schedule.
    width = (d / 512.0) ** 2 * (nb / 3.0)
    t_step = 0.010 + 1.5e-5 * bs * width * (0.6 if cfg["amp"] else 1.0)
    t_epoch = steps * t_step + 0.01 + 2e-6 * n / 10.0
    cfg["epochs"] = int(min(20000, max(50, 2.0 * S / t_epoch)))
    return cfg


# ------------------------------------------------------------------------------------------ symmetry
_V4 = ((0, 1, 2, 3), (1, 0, 3, 2), (2, 3, 0, 1), (3, 2, 1, 0))


def _v4_index(in_dim):
    """(4, in_dim) column index: row g permutes the resonator triples (cols 2+i, 6+i, 10+i) by g."""
    rows = []
    for g in _V4:
        idx = list(range(in_dim))
        for i in range(4):
            idx[2 + i] = 2 + g[i]
            idx[6 + i] = 6 + g[i]
            idx[10 + i] = 10 + g[i]
        rows.append(idx)
    return torch.tensor(rows, dtype=torch.long)


# ------------------------------------------------------------------------------------------ blocks
class ResMLP(nn.Module):
    def __init__(self, d, mult):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.fc1 = nn.Linear(d, d * mult)
        self.fc2 = nn.Linear(d * mult, d)

    def forward(self, h):
        return h + self.fc2(F.gelu(self.fc1(self.norm(h))))


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


class SymConvNet(nn.Module):
    def __init__(self, in_dim, out_dim, cfg):
        super().__init__()
        d = int(cfg["d_model"])
        self.sym = bool(cfg.get("sym_v4", True)) and in_dim == 14
        if self.sym:
            self.register_buffer("perm", _v4_index(in_dim), persistent=False)
        self.inp = nn.Linear(in_dim, d)
        self.blocks = nn.ModuleList([ResMLP(d, int(cfg["enc_mult"])) for _ in range(int(cfg["enc_blocks"]))])
        self.norm = nn.LayerNorm(d)
        r = int(cfg["basis_rank"])
        self.coef = nn.Linear(d, r)
        self.basis = nn.Linear(r, out_dim)            # learned basis vectors (columns) + per-point bias
        self.dec = ConvSpectralDecoder(d, list(cfg["dec_channels"]), int(cfg["seed_len"]), int(cfg["kernel"]),
                                       int(cfg.get("subpixel", 8)), out_dim)

    def encode(self, x):
        h = F.gelu(self.inp(x))
        for blk in self.blocks:
            h = blk(h)
        return self.norm(h)

    def forward(self, x):
        if self.sym:
            b = x.shape[0]
            xs = x[:, self.perm]                        # (B, 4, 14): the V4 orbit of each geometry
            z = self.encode(xs.reshape(b * 4, x.shape[1])).view(b, 4, -1).mean(1)
        else:
            z = self.encode(x)
        return self.basis(self.coef(z)) + self.dec(z)


def build_model(in_dim, out_dim, cfg):
    return SymConvNet(in_dim, out_dim, cfg)


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
