# Agent tool reference (agentic metasurface framework)

Reproduction of Lu, Malof & Padilla, *An Agentic Framework for Autonomous Metamaterial Modeling and
Inverse Design*, ACS Photonics 2025, 12, 6071. The tools below are the human-written APIs of the
framework; LLM agents decide what to do with them and write the model code.

## Environment (always)

```bash
cd /d/Claude/metasurface_na_demo            # repository root (Windows path D:/Claude/metasurface_na_demo)
PY=/c/Users/93107/.conda/envs/cst_inference/python.exe
"$PY" -I framework/<tool>.py ...            # always run from the repo root, always with -I
```

* Local machine only (never ssh / DCC). GPU: NVIDIA RTX 4080 SUPER, 16 GB, shared by all agents:
  at most 3 harness training runs execute at once, extra runs queue automatically.
* Use a Bash timeout of 600000 ms for every `harness.py submit` / `harness.py wait` call.
* Never modify `framework/` or `data/`. Never read or use the independent test set
  (`data/processed/test_*.npy`, `data/extracted/ADM/testset/*`) during model development.
* Official numbers are only the ones the harness prints (`RESULT_JSON:` lines / ledger).

## Data protocol

* Training pool: 52,812 ADM geometry-spectrum pairs, shuffled once (seed 2025).
  Geometry x: 14 values `[h, p, rx1..rx4, ry1..ry4, theta1..theta4]` on a discrete grid
  (h: 0.3-0.6, 5 levels; p: 1.0-1.5, 5 levels; semi-axes: 0.1-0.2, 9 levels; angles: -0.7854..0.7854 rad,
  5 levels).
  Spectrum y: 2001 absorptivity values in about [-0.01, 1.0], with sharp resonances.
* D_k = first k pool rows; train:val = 10:1 (pool position i is validation iff i % 11 == 10).
  Data sets are nested; growing the data only adds new samples.
* k0 = 500, data budget = 50,000 (max k), max rounds = 50.
* Metric: **validation MSE** = mean of (y_pred - y_true)^2 over all validation samples and all
  2001 points, on raw absorptivity (the harness inverts any target scaling you use).
  Strict upper bound 2.0e-3, stretch goal 1.3e-3. Predicting the mean spectrum gives 1.5e-2.

## Tools

### File_Check
```bash
"$PY" -I framework/file_check.py PATH [PATH ...]       # JSON {all_exist, files}; exit 2 if any missing
```

### Prepare Dataset
```bash
"$PY" -I framework/prepare_dataset.py --k K [--note "..."]   # grow acquired data to K (idempotent)
"$PY" -I framework/prepare_dataset.py --status
```
Only acquire the size the Controller decided. Training on any k <= k_current is allowed.

### Harness (Evaluate / Code Generation execution)
```bash
# validate a solution file, count parameters, time 2 epochs on D_k (nothing is logged)
"$PY" -I framework/harness.py check --code phase2_forward/solutions/FILE.py --k K --time-epochs 2

# train + evaluate on D_k (snapshots the code, detached run, waits up to 540 s)
"$PY" -I framework/harness.py submit --code phase2_forward/solutions/FILE.py --k K --tag TAG \
     --max-train-seconds S
# exit 0 = DONE (prints RESULT_JSON), 1 = FAILED (prints traceback), 3 = still RUNNING:
"$PY" -I framework/harness.py wait RUN_ID              # repeat until DONE/FAILED
"$PY" -I framework/harness.py submit ... --wait 0      # launch without waiting (then `wait` each)

"$PY" -I framework/harness.py ledger --last 20         # recent runs
"$PY" -I framework/harness.py ledger --k K --best      # best runs at size K
```
`--max-train-seconds` caps the training wall time (time limit -> training stops, the best
checkpoint so far is kept). Max 2400. Use `check --time-epochs 2` to measure seconds/epoch (solo);
up to 3 concurrent runs slow each other down (expect 1.5-3x). With the default time-aware cosine
schedule the learning rate is annealed by the time limit anyway, so a run that is slowed down still
ends with a fully annealed model (it simply sees fewer epochs).
The reported `val_mse` is that of the best validation checkpoint.

## Solution file contract (what AIDE writes)

One self-contained Python file in `phase2_forward/solutions/` (no imports of other local files).
Store any fixed random tensors as registered buffers. Required:

```python
def build_model(in_dim: int, out_dim: int, cfg: dict) -> torch.nn.Module:
    """Maps SCALED geometry (B, 14) to the (scaled) spectrum (B, 2001)."""
```

Optional:

```python
CONFIG = {...}   # overrides the harness defaults below
def adapt_config(cfg: dict, n_train: int, max_train_seconds: float) -> dict:
    """Recommended: scale epochs / batch size with the data set size so the SAME code stays
    appropriate when the Controller grows the data (a 'test' action re-runs the code unchanged)."""
def compute_loss(model, xb, yb, cfg) -> torch.Tensor          # custom loss (xb, yb are scaled)
def make_optimizer(model, cfg) -> torch.optim.Optimizer
def make_scheduler(optimizer, cfg, steps_per_epoch, epochs) -> (scheduler, "step"|"epoch"|"plateau"|None)
```

The module docstring must describe the architecture and training recipe in one paragraph.

Harness defaults (override in CONFIG):

| key | default | options |
|---|---|---|
| epochs | 200 | |
| batch_size | 256 | |
| lr | 1e-3 | |
| weight_decay | 1e-4 | |
| optimizer | adamw | adamw, adam, radam, nadam, sgd |
| scheduler | cosine (per step, linear warmup `warmup_epochs`=3, floor `min_lr_ratio`=0.01) | cosine, onecycle, step, plateau, none |
| time_aware_schedule | True: cosine progress = max(step progress, elapsed / max_train_seconds), so the LR always reaches its floor by the time limit even when GPU sharing slows the run | True, False |
| loss | mse | mse, smooth_l1 (`smooth_l1_beta`=0.01), l1, huber (`huber_delta`=0.01) |
| grad_clip | 0 (off) | |
| x_scaling | minmax (to [-1, 1], fitted on train) | minmax, standard, none |
| y_scaling | none | none, standard (per point), global_standard, minmax |
| amp | False | bf16 autocast for training |
| tf32 | True | TF32 matmuls for training (evaluation is always full fp32) |
| ema_decay | 0 (off) | e.g. 0.999 |
| eval_every | 1 | epochs between validation evaluations |
| patience | 0 (off) | early-stopping patience in epochs |
| seed | 0 | |

Example:

```python
"""Residual MLP: Linear(14->1024), 6 pre-LayerNorm residual blocks (Linear-GELU-Linear, width 1024),
LayerNorm, Linear(1024->2001). AdamW lr 1e-3, cosine schedule, MSE on raw absorptivity."""
import torch.nn as nn

CONFIG = {"epochs": 300, "batch_size": 256, "lr": 1e-3}

def adapt_config(cfg, n_train, max_train_seconds):
    cfg["epochs"] = int(min(600, max(100, 300 * 20000 / max(n_train, 1))))
    return cfg

class Block(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.f = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))
    def forward(self, x):
        return x + self.f(x)

def build_model(in_dim, out_dim, cfg):
    d = 1024
    return nn.Sequential(nn.Linear(in_dim, d), *[Block(d) for _ in range(6)],
                         nn.LayerNorm(d), nn.Linear(d, out_dim))
```

## Reporting tools (Reporter / verifiers only)

```bash
"$PY" -I framework/plot_history.py --history phase2_forward/memory/history.json \
     --out phase2_forward/val_mse_vs_dataset_size.png --candidates-out phase2_forward/all_candidates.png
"$PY" -I framework/final_eval.py --run RUN_ID --test --figure phase2_forward/test_examples.png \
     --out phase2_forward/final_eval.json
```
