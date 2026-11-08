"""Sharpness-aware (SAM) training of a batched deep ensemble (K=4) of the incumbent V4-exact relational transformer
(recipe change vs r06_recipe_ens4wd: SAM, per-member radius rho = 0.1). Member = incumbent r05: 1 global (h, p)
token + 4 resonator tokens from ellipse shape-tensor invariants s = rx+ry, u, v = (rx-ry)cos/sin 2theta, |d|,
u/|d|, v/|d|, h, p (shared GELU-MLP embedding), n_layers pre-LN blocks (d_model, 8 heads, GELU FFN 4x) with
per-head attention-logit biases per resonator relation i XOR j (exact V4 invariance), mean pool, head
concat(global, mean resonator) -> 2x Linear-GELU(head_width) -> Linear(2001); members are batched matmuls, each
trained on its own MSE, output = member mean. Recipe: every step takes the member gradient g_m, moves member m
to w_m + rho g_m / ||g_m|| (worst case in a rho-ball), back-propagates the MSE there and applies that gradient
to w_m (AdamW pre-step hook restores w_m), i.e. it minimises the neighbourhood-max loss (flat minima); AdamW
betas 0.9/0.98, effective wd = 1 / (lr tau), tau = 0.35 x expected steps (k=16000: wd 0.10), global_standard
y, time-aware cosine + warmup, EMA, grad clip 1 per member; incumbent n_train table (k=16000: d 320, 4 layers,
head 1536, batch 256, lr 1.5e-3, dropout 0.02). Cost: 2 forward/backward passes per step.

Why SAM: r06_recipe_ens4wd showed that an effective weight decay (timescale 6.6k steps instead of 67k) leaves
the val curve unchanged (best 1.503e-3 at epoch 75 vs 1.510e-3 for r06_improve_ens4 with wd 0.01; afterwards the
same slow rise, also while the LR is annealed to its floor), i.e. limiting the weight norm or annealing early
does not change how the model interpolates between the sparse grid samples. SAM instead penalises the loss
increase under worst-case weight perturbations, which slows memorisation of single samples and biases each
member toward flat, more robust interpolants; it targets the shared (member-correlated) error that ensembling
cannot remove.

Earlier reasoning kept from r06_recipe_ens4wd: every incumbent run reaches its best val MSE after 5-10 % of the budget and then overfits (k=16000: best
epoch 67 of 849, train 2.3e-4 vs val 1.72e-3), and the 4-member ensemble only removes the member-variance part
(r06_improve_ens4: 1.51e-3, i.e. member errors are ~82 % correlated). The incumbent's AdamW decay was inactive:
its timescale 1 / (lr * wd) = 67k steps exceeds the whole run (48k steps) and is 17x the steps to the best epoch,
so the weight norm was never controlled. A timescale of about 1.7x the steps-to-best makes weight decay an
actual capacity control (smaller-norm, smoother interpolants between the sparse grid samples), which targets the
shared, systematic part of the error that ensembling cannot remove. Diagnostic (train split, incumbent
checkpoint): the model is already invariant to the global angle flip to 3.5e-5 MSE at unlabeled geometries,
so flip augmentation / test-time flipping cannot help (expected gain < 1 %) and is not used.

Physics prior (unchanged): shifting the periodic 2x2 supercell by one sub-cell permutes the resonators by a
double transposition i -> i XOR g (Klein four-group V4); attention that sees only the pair relations i XOR j plus
per-token content is V4-equivariant, and mean pooling makes the spectrum exactly V4-invariant. The shape-tensor
features make the two encodings (rx, ry, theta) and (ry, rx, theta -+ pi/2) of the same ellipse identical.
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
    "grad_clip": 0.5,            # = 1 / sqrt(n_members): per-member clip of about 1 (set in adapt_config)
    "x_scaling": "minmax",       # column permutations of the resonator triples stay exact under minmax
    "y_scaling": "global_standard",
    "amp": False,
    "tf32": True,
    "ema_decay": 0.996,
    "eval_every": 1,
    "patience": 0,
    # ---- ensemble
    "n_members": 4,
    # ---- weight decay from a decay timescale (adapt_config): tau = wd_tau_frac x expected optimizer steps
    "wd_tau_frac": 0.35,
    "wd_min": 0.02,
    "wd_max": 0.3,
    # ---- sharpness-aware minimisation (per-member radius in parameter space; 0 = off)
    "sam_rho": 0.1,              # about 0.1 % of a member's initial weight norm (about 95)
    # ---- model (per member)
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
    # ---- single-member timing model of the incumbent (used only for EMA horizon / warmup / epoch cap, so the
    #      per-member recipe is exactly the incumbent's)
    "est_step_base_s": 0.007,
    "est_step_per_layer_s": 0.003,
    "est_epoch_overhead_s": 0.01,
    "warmup_steps": 300,
}


def adapt_config(cfg, n_train, max_train_seconds):
    """Incumbent n_train table for the per-member recipe; ensemble-aware gradient clip."""
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
    K = int(cfg["n_members"])
    cfg["grad_clip"] = round(1.0 / math.sqrt(K), 4)   # loss is the member MEAN -> member grads are 1/K
    # effective decoupled weight decay: decay timescale tau = 1 / (lr * wd) = wd_tau_frac x expected steps
    tau = float(cfg["wd_tau_frac"]) * est_steps
    wd_eff = 1.0 / (float(cfg["lr"]) * tau)
    cfg["weight_decay"] = round(min(float(cfg["wd_max"]), max(float(cfg["wd_min"]), wd_eff)), 4)
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


def _member_mse(model, xb, yb):
    pred = model.forward_members(xb).float()            # (K, B, 2001)
    return (pred - yb.unsqueeze(0)).pow(2).mean()


def compute_loss(model, xb, yb, cfg):
    """SAM: mean over members of each member's own MSE, evaluated at the per-member worst-case perturbation
    w_m + rho * g_m / ||g_m|| (g_m = member gradient at w_m). The harness back-propagates the returned loss, so
    p.grad = gradient at the perturbed point; the optimizer pre-step hook (make_optimizer) subtracts the
    perturbation again before AdamW applies that gradient to the unperturbed weights."""
    rho = float(cfg.get("sam_rho", 0.0))
    if rho <= 0 or not model.training or not torch.is_grad_enabled():
        return _member_mse(model, xb, yb)
    params = [p for p in model.parameters() if p.requires_grad]
    grads = torch.autograd.grad(_member_mse(model, xb, yb), params, allow_unused=True)
    K = model.K
    with torch.no_grad():
        sq = torch.zeros(K, device=xb.device, dtype=torch.float32)
        for g in grads:
            if g is not None:
                sq += g.float().pow(2).reshape(K, -1).sum(1)      # every parameter has a leading member dim
        scale = rho / (sq.sqrt() + 1e-12)                          # (K,) per-member normalisation
        eps = []
        for p, g in zip(params, grads):
            if g is None:
                eps.append(None)
                continue
            e = (g * scale.view(K, *([1] * (g.dim() - 1)))).to(p.dtype)
            p.add_(e)
            eps.append(e)
    model._sam_state = (params, eps)
    return _member_mse(model, xb, yb)


def make_optimizer(model, cfg):
    decay, no_decay = [], []
    for m in model.modules():
        if isinstance(m, BLinear):
            decay.append(m.weight)
            no_decay.append(m.bias)
        elif isinstance(m, BLayerNorm):
            no_decay += [m.weight, m.bias]
        elif isinstance(m, BRelAttention):
            no_decay.append(m.rel_bias)
    n_all = sum(1 for _ in model.parameters())
    assert len(decay) + len(no_decay) == n_all, "every parameter must be in exactly one group"
    groups = [{"params": decay, "weight_decay": float(cfg["weight_decay"])},
              {"params": no_decay, "weight_decay": 0.0}]
    kw = dict(lr=float(cfg["lr"]), betas=tuple(cfg["betas"]))
    try:
        opt = torch.optim.AdamW(groups, fused=True, **kw)
    except Exception:  # noqa: BLE001 - fused kernel unavailable
        opt = torch.optim.AdamW(groups, **kw)

    def _sam_restore(optimizer, args, kwargs):
        """Undo the SAM perturbation (set in compute_loss) right before the AdamW step."""
        st = model.__dict__.pop("_sam_state", None)
        if st is None:
            return
        with torch.no_grad():
            for p, e in zip(*st):
                if e is not None:
                    p.sub_(e)

    opt.register_step_pre_hook(_sam_restore)
    return opt
