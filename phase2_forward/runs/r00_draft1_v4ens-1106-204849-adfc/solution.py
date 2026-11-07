"""Batched deep ensemble (M=4) of pre-LayerNorm residual MLPs with exact supercell-shift (V4) symmetry. Member:
Linear(14->d), 6 residual blocks [LN -> Linear(d->d) -> GELU -> Linear(d->d), zero-init output proj], LN,
Linear(d->2001) (zero-init weights, bias 0.5); d=768 (1024 for n_train>12k); members run as one batched matmul
per layer. V4: shifting the 2x2 supercell by one sub-cell permutes the resonator triples (rx_i, ry_i, theta_i) by
a double transposition, so each member is trained on the full 4-view orbit of every sample (mean of the members'
MSEs on raw absorptivity) and the output is the mean over members and views (exactly invariant). Fused AdamW (no
bias decay), batch 32 / lr 7e-4 at k=500 (lr ~ sqrt(batch), cap 2e-3), wd 1e-2*sqrt(2000/n) in [2e-3, 1e-2],
grad clip 1, EMA 0.999, time-aware cosine (400 warmup steps, floor 1 %), epoch cap 1e5 (wall clock drives the
schedule), minmax x (needed for exact permutation), raw y. adapt_config: batch 32/64/128/256/512 up to n_train
1.5k/4k/12k/25k/above; above 12k one random view per member and sample instead of the full orbit."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

CONFIG = {
    "epochs": 100000,          # generous cap: the time limit + time-aware cosine drive the schedule
    "batch_size": 32,
    "lr": 7e-4,
    "weight_decay": 1e-2,
    "optimizer": "adamw",
    "scheduler": "cosine",
    "warmup_epochs": 27,       # overwritten by adapt_config (about warmup_steps steps)
    "min_lr_ratio": 0.01,
    "time_aware_schedule": True,
    "grad_clip": 1.0,
    "ema_decay": 0.999,
    "x_scaling": "minmax",     # REQUIRED for the exact column-permutation symmetry
    "y_scaling": "none",
    "loss": "mse",
    "amp": False,
    "eval_every": 1,
    # ---- model / symmetry (custom keys) ----
    "n_members": 4,
    "width": 768,
    "depth": 6,
    "ff_mult": 1.0,
    "dropout": 0.0,
    "head_bias_init": 0.5,     # raw absorptivity mean is about 0.52
    "use_flip": False,         # also treat "negate all four angles" as a symmetry (mirror; polarization-dependent)
    "aug_views": 0,            # 0 = full orbit (all views of every sample); 1 = one random view per member/sample
    "warmup_steps": 400,
}

# Klein four-group V4 acting on resonator indices 0..3 (identity + the three double transpositions).
_V4 = [[0, 1, 2, 3], [1, 0, 3, 2], [2, 3, 0, 1], [3, 2, 1, 0]]


def _view_tables(use_flip: bool):
    """Column-index and sign tables of the symmetry views: x_view = x[:, idx] * sgn."""
    idx, sgn = [], []
    for flip in ([False, True] if use_flip else [False]):
        for p in _V4:
            idx.append([0, 1] + [2 + j for j in p] + [6 + j for j in p] + [10 + j for j in p])
            sgn.append([1.0] * 10 + [-1.0 if flip else 1.0] * 4)
    return torch.tensor(idx, dtype=torch.long), torch.tensor(sgn, dtype=torch.float32)


def adapt_config(cfg, n_train, max_train_seconds):
    n = max(int(n_train), 1)
    if n <= 1500:
        bs = 32
    elif n <= 4000:
        bs = 64
    elif n <= 12000:
        bs = 128
    elif n <= 25000:
        bs = 256
    else:
        bs = 512
    cfg["batch_size"] = bs
    cfg["lr"] = min(2e-3, 1e-3 * math.sqrt(bs / 64.0))
    cfg["weight_decay"] = float(min(1e-2, max(2e-3, 1e-2 * math.sqrt(2000.0 / n))))
    if n > 12000:
        cfg["width"] = max(int(cfg["width"]), 1024)
        cfg["aug_views"] = 1       # full orbit costs |G|x compute; plenty of epochs remain with one view
    steps_per_epoch = max(1, math.ceil(n / min(bs, n)))
    cfg["warmup_epochs"] = max(1.0, float(cfg.get("warmup_steps", 400)) / steps_per_epoch)
    cfg["epochs"] = 100000         # the time limit ends training; the cosine follows the wall clock
    return cfg


class EnsembleLinear(nn.Module):
    """M independent Linear layers evaluated as one batched matmul: (M, R, din) -> (M, R, dout)."""

    def __init__(self, m, din, dout, w_scale=1.0, bias_init=None):
        super().__init__()
        bound = 1.0 / math.sqrt(din)
        w = torch.empty(m, din, dout).uniform_(-bound, bound) * float(w_scale)
        if bias_init is None:
            b = torch.empty(m, 1, dout).uniform_(-bound, bound)
        else:
            b = torch.full((m, 1, dout), float(bias_init))
        self.weight = nn.Parameter(w)
        self.bias = nn.Parameter(b)

    def forward(self, x):
        return torch.baddbmm(self.bias, x, self.weight)


class Block(nn.Module):
    """Pre-LN residual MLP block for all members at once; output projection zero-initialised."""

    def __init__(self, m, d, dff, dropout):
        super().__init__()
        self.fc1 = EnsembleLinear(m, d, dff)
        self.fc2 = EnsembleLinear(m, dff, d, w_scale=0.0, bias_init=0.0)
        self.dropout = float(dropout)

    def forward(self, h):
        z = F.layer_norm(h, h.shape[-1:])
        z = F.gelu(self.fc1(z))
        if self.dropout > 0:
            z = F.dropout(z, self.dropout, self.training)
        return h + self.fc2(z)


class SymEnsembleResMLP(nn.Module):
    def __init__(self, in_dim, out_dim, cfg):
        super().__init__()
        self.m = int(cfg["n_members"])
        d = int(cfg["width"])
        dff = int(round(d * float(cfg["ff_mult"])))
        self.inp = EnsembleLinear(self.m, in_dim, d)
        self.blocks = nn.ModuleList([Block(self.m, d, dff, cfg["dropout"]) for _ in range(int(cfg["depth"]))])
        self.head = EnsembleLinear(self.m, d, out_dim, w_scale=0.0, bias_init=cfg["head_bias_init"])
        idx, sgn = _view_tables(bool(cfg["use_flip"]))
        self.register_buffer("view_idx", idx, persistent=False)
        self.register_buffer("view_sgn", sgn, persistent=False)

    def members_forward(self, xr):
        """xr: (R, 14) shared by all members or (M, R, 14) per member -> (M, R, out_dim)."""
        if xr.dim() == 2:
            xr = xr.unsqueeze(0).expand(self.m, -1, -1)
        h = self.inp(xr)
        for blk in self.blocks:
            h = blk(h)
        return self.head(F.layer_norm(h, h.shape[-1:]))

    def views(self, x):
        """(B, 14) -> (B, G, 14): all symmetry views of every sample."""
        return x[:, self.view_idx] * self.view_sgn

    def forward(self, x):
        b = x.shape[0]
        g = self.view_idx.shape[0]
        out = self.members_forward(self.views(x).reshape(b * g, -1))      # (M, B*G, out)
        return out.view(self.m, b, g, -1).mean(dim=(0, 2))


def build_model(in_dim, out_dim, cfg):
    return SymEnsembleResMLP(in_dim, out_dim, cfg)


def compute_loss(model, xb, yb, cfg):
    """Mean of the members' MSEs over symmetry views of the batch (full orbit or one random view each)."""
    m, b = model.m, xb.shape[0]
    g = model.view_idx.shape[0]
    nv = int(cfg.get("aug_views", 0))
    if nv <= 0 or nv >= g:
        xv = model.views(xb).reshape(b * g, -1)
        out = model.members_forward(xv).float()                           # (M, B*G, out)
        diff = out.view(m, b, g, -1) - yb[None, :, None, :]
    else:
        sel = torch.randint(g, (m, b), device=xb.device)
        xv = torch.gather(xb.unsqueeze(0).expand(m, -1, -1), 2, model.view_idx[sel]) * model.view_sgn[sel]
        out = model.members_forward(xv).float()                           # (M, B, out)
        diff = out - yb[None]
    return diff.pow(2).mean()


def make_optimizer(model, cfg):
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if name.endswith("bias") else decay).append(p)
    groups = [{"params": decay, "weight_decay": float(cfg["weight_decay"])},
              {"params": no_decay, "weight_decay": 0.0}]
    kw = dict(lr=float(cfg["lr"]), betas=tuple(cfg.get("betas", (0.9, 0.999))))
    try:
        return torch.optim.AdamW(groups, fused=True, **kw)
    except (RuntimeError, TypeError, ValueError):
        return torch.optim.AdamW(groups, **kw)
