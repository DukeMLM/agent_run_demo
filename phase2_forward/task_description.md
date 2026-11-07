# AIDE task: ADM forward surrogate (14 geometry parameters -> 2001-point absorptivity spectrum)

Write one PyTorch solution file that predicts the absorptivity spectrum of an all-dielectric metasurface (ADM)
from its geometry. Goal: **validation MSE <= 2.0e-3 (strict upper bound), as close to 1.3e-3 as possible.**
This reproduces Forward_Train of Lu, Malof & Padilla, ACS Photonics 2025, 12, 6071. Full tool reference and
contract: `framework/TOOLS.md`; round policy: `phase2_forward/memory/plan.md`.

## Background

* Unit cell: a 2x2 supercell of four SiC elliptical resonators. Spectra are CST simulations of absorptivity
  at 2001 frequency points.
* Input x, shape (B, 14), fixed column order `[h, p, rx1, rx2, rx3, rx4, ry1, ry2, ry3, ry4, theta1, theta2,
  theta3, theta4]`: height h 0.3-0.6 (5 levels), period p 1.0-1.5 (5), semi-axes rx_i, ry_i 0.1-0.2
  (9 levels), rotations theta_i -0.7854..0.7854 rad (5). Resonator i (1-4) = columns (1+i, 5+i, 9+i);
  h and p are shared. Values lie on a grid, but the model must treat them as continuous.
* Output y, shape (B, 2001): absorptivity in about [-0.01, 1.0] (0.7 % of values slightly negative),
  mean 0.52, std 0.31. Spectra contain sharp resonances whose positions shift quickly with the geometry;
  capturing them is the hard part. Label noise is tiny (duplicate-pair floor about 1.3e-5).
* Physics prior (hypothesis: keep it only if the val MSE improves): shifting the periodic supercell by one
  sub-cell permutes the resonators by a double transposition, (1 2)(3 4), (1 3)(2 4) or (1 4)(2 3), whatever
  the index layout, so these permutations of the (rx_i, ry_i, theta_i) triples should leave the spectrum
  unchanged. Flipping the sign of all four angles is a mirror symmetry only if the incident polarization is
  along x or y (not documented). Use as augmentation inside `compute_loss` (column permutation is exact under
  the default minmax x scaling) or as output symmetrization (average over the 4 permutations).

## Data and evaluation protocol (all handled by the harness)

* D_k = first k rows of a fixed shuffled pool (52,812 samples); k is set per round by the Controller
  (k0 = 500, max 50,000). Split 10:1: pool position i is validation iff i % 11 == 10; data sets are nested.
  Examples: k = 500 -> 455 train / 45 val; 10,000 -> 9,091 / 909; 50,000 -> 45,455 / 4,545.
* The harness fits the x / y scalers on the train split, puts all tensors on the GPU, trains your model and
  evaluates it every `eval_every` epochs. Your file never loads data. Never read the test set
  (`data/processed/test_*`, `data/extracted/ADM/testset/`); never modify `framework/` or `data/`.
* Metric: mean of (y_pred - y_true)^2 over all validation samples and all 2001 points on raw absorptivity
  (the harness inverts your y scaling; fp32 evaluation). Reported `val_mse` = best validation checkpoint.
  Mean-spectrum baseline: 1.52e-2. Only numbers printed by the harness (`RESULT_JSON`, ledger) count.
* "Test" rounds re-run your file unchanged on a larger k: it must work well for every k from 500 to 50,000.

## Solution-file contract (essentials of framework/TOOLS.md)

One self-contained file `phase2_forward/solutions/rNN_<idea>.py`: a new file per candidate, never edit a
submitted file, imports from torch / numpy / math / the standard library only (no local imports).

    """One paragraph: architecture and training recipe (reported as the architecture details)."""
    CONFIG = {...}                                     # optional: overrides defaults; custom keys allowed
    def adapt_config(cfg, n_train, max_train_seconds): ...; return cfg    # optional, recommended
    def build_model(in_dim, out_dim, cfg) -> nn.Module    # REQUIRED: scaled (B, 14) -> scaled (B, 2001)
    def compute_loss(model, xb, yb, cfg) -> scalar        # optional; xb, yb scaled; bf16 autocast if amp
    def make_optimizer(model, cfg) -> torch.optim.Optimizer           # optional
    def make_scheduler(opt, cfg, steps_per_epoch, epochs) -> (sched, "step" | "epoch" | "plateau" | None)

* Precedence: harness defaults <- CONFIG <- adapt_config(cfg, n_train, max_train_seconds) <- CLI `--set`.
  The merged cfg goes to build_model and is logged. Keep every setting in the file (no `--set`).
* Defaults: epochs 200, batch_size 256, lr 1e-3, weight_decay 1e-4, optimizer adamw (adam, radam, nadam,
  sgd), scheduler cosine per step (warmup_epochs 3, min_lr_ratio 0.01, time_aware_schedule True; also
  onecycle, step, plateau, none), loss mse (smooth_l1 beta 0.01, l1, huber delta 0.01), grad_clip 0,
  x_scaling minmax to [-1, 1] (standard, none), y_scaling none (standard per point, global_standard,
  minmax), amp False (bf16), tf32 True, ema_decay 0, eval_every 1, patience 0, seed 0.
* Reloadable: `final_eval.py` and Phase 3 rebuild the model with `build_model(14, 2001, logged cfg)` and
  `load_state_dict(strict=True)`. Structure must follow from cfg alone; fixed random tensors (Fourier
  matrices, encodings) must be registered buffers; no lazy modules, no parameters created in forward.
* Differentiable w.r.t. the input: Phase 3 Neural-Adjoint backpropagates through the frozen model to the
  geometry, including off-grid points. No rounding / one-hot / lookup of inputs, no detach or no_grad in
  forward, no hard output clamps; smooth activations (GELU, SiLU) give cleaner input gradients.
* A custom `make_scheduler` replaces the time-aware cosine (it receives no wall clock); prefer the built-in.
* Not available: torch.compile (no Triton on this Windows machine), DataLoader workers, file / network I/O.

## Running candidates

    cd /d/Claude/metasurface_na_demo && PY=/c/Users/93107/.conda/envs/cst_inference/python.exe
    "$PY" -I framework/harness.py check --code phase2_forward/solutions/F.py --k K --time-epochs 2
    "$PY" -I framework/harness.py submit --code phase2_forward/solutions/F.py --k K --tag rNN_idea_kK \
         --max-train-seconds S --wait 0
    "$PY" -I framework/harness.py wait RUN_ID    # exit 0 DONE (RESULT_JSON), 1 FAILED (traceback), 3 RUNNING
    "$PY" -I framework/harness.py ledger --k K --best

K and S come from the Forward Modeler for the round. Run `check` before every submit, use a Bash timeout of
600000 ms for submit / wait, and repeat `wait` while it exits 3.

## Practical guidance

* Scaling: minmax x is a good default. For y, `none` or `global_standard` keep the training loss
  proportional to the metric; per-point `standard` / `minmax` up-weight low-variance frequencies and can
  raise the raw MSE. Prefer a linear output layer; a bounded output must cover [-0.01, 1.0].
* Time budget: training stops at S (best checkpoint kept). The time-aware cosine anneals the LR to its floor
  by S even when GPU sharing slows the run, which then just sees fewer epochs. Size epochs from
  `check --time-epochs 2`: epochs ~ S / (sec_per_epoch x 1.5-2), the factor covering 3 concurrent runs; or
  set a generous epoch cap and let the wall clock drive the schedule. onecycle / step are not time-aware.
* adapt_config by n_train: small batches (32-64) and thousands of epochs at k <= 2,000; 128-256 at
  10,000-20,000; 256-512 at >= 40,000 (scale lr roughly with sqrt(batch)); lower dropout / weight decay as
  n_train grows; capacity may grow with n_train. Keep the EMA horizon 1 / (1 - decay) steps well below the
  total number of steps.
* GPU sharing: up to 3 runs share 16 GB, so keep one run under about 4 GB (the k = 50,000 data take 0.4 GB).
  No full attention over the 2001 output points (2001^2 per head): use patches. bf16 `amp` speeds up large
  models (evaluation stays fp32). Small models are bound by per-step overhead: larger batches help.
* Noise and fairness: k = 500 has only 45 val samples, so do not over-tune at small k. Compare candidates at
  the same k and S launched together; change one thing per improve step; no seed shopping.

## Possible solution families (a menu, not a constraint: any differentiable nn.Module is allowed)

1. Deep residual MLPs: pre-norm (LayerNorm / RMSNorm) blocks, GELU / SiLU, width 512-2048, depth 4-16,
   optional dropout / stochastic depth, wide linear output layer.
2. Gated / SE residual MLPs: GLU / SwiGLU blocks, squeeze-and-excitation gating of hidden features,
   highway or learned-scale skips.
3. Input embeddings: per-parameter linear embeddings; Fourier features of x (random projection stored as a
   buffer) for the fast variation of the resonances with geometry.
4. Transformer encoders over geometry tokens: 14 scalar tokens, or 4 resonator tokens (rx_i, ry_i, theta_i)
   plus an (h, p) token with type embeddings; 2-8 layers, width 128-512; pooled or flattened into an MLP
   head or a convolutional spectral decoder.
5. 1-D (transposed-)convolutional spectral decoders: latent -> (C x L) -> transposed-conv or upsample + conv
   stack -> 2048 points, cropped to 2001; or an MLP output refined by a residual 1-D conv stack (resonances
   are local features); patch-token decoders (latent -> P patch tokens -> attention / MLP-Mixer -> linear).
6. Low-rank / basis spectral heads: y = b0 + sum_r c_r(x) B_r with a learned basis (r = 32-512), optionally
   plus a full-rank residual head (the file never sees the data, so learn the basis instead of PCA).
7. Mixture-of-experts: soft gating over 2-8 expert MLPs or heads.
8. Physics-inspired heads: sum of K Lorentzian / Fano line shapes (learned centres, widths, amplitudes) plus
   a smooth background and a free-form residual head (harder to optimize).
9. Conditional implicit decoders: an MLP over frequency-coordinate features, FiLM-conditioned on a geometry
   latent, evaluated at all 2001 points (memory grows as B x 2001 x hidden).
10. Small ensembles: 2-5 members averaged inside one module (each member gets a share of S).

Training-recipe options: MSE loss (equals the metric) vs smooth-L1 / Huber (set beta / delta near the
residual scale, about 0.02-0.05 in raw units, not the 0.01 default; rescale if y is scaled); an auxiliary
MSE on first differences along frequency to sharpen resonances; AdamW lr 3e-4 to 2e-3, weight decay 1e-5 to
1e-2; grad clip 0.5-1 for deep or attention models; EMA 0.999-0.9999; dropout 0-0.2; long cosine schedules
(warmup 3-10 epochs, min_lr_ratio 1e-3 to 1e-2); symmetry augmentation (see Background).

## Search procedure (each AIDE call)

Inputs from the Forward Modeler: round, k, S, the incumbent file and run (if any), `memory/aide_journal.jsonl`.
1. Read the journal and `ledger --k K --best`; do not repeat ideas that already failed.
2. Draft: new solutions from different families (a wave of 3 drafts covers at least 2 families).
3. Improve: one change to the best node (width / depth, embedding, head, loss, lr / weight decay / dropout,
   EMA, batch, epochs), so its effect is attributable.
4. Debug: fix a failed node from the `wait` traceback (max 2 attempts).
5. Log every node to the journal; return the best node: file, run_id, val_mse, n_params, one-line idea.
