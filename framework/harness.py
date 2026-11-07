#!/usr/bin/env python
"""Forward_Train execution harness: trains and scores AIDE-generated forward-model code.

Every candidate forward surrogate is trained and scored through this harness so that all
validation-MSE numbers in the Forward_Train history are computed identically:

  * data set D_k    = first k rows of the fixed shuffled ADM training pool (Prepare Dataset tool),
  * train/val split = 10:1 (pool position i is validation iff i % 11 == 10),
  * validation MSE  = mean of (prediction - truth)^2 over ALL validation samples and ALL 2001
                      spectral points, on the ORIGINAL (unscaled) absorptivity values, evaluated
                      in fp32 with TF32 disabled and an fp64 reduction.
  * the reported val_mse is that of the best checkpoint (model selection on validation MSE).

Commands (run from the repository root with the cst_inference interpreter, `python -I`):
  check  --code FILE [--k K --time-epochs N] [--set key=value ...]
         import the solution, build the model, count parameters, run a forward pass; optionally
         time N training epochs on D_k (nothing is saved, nothing is logged).
  submit --code FILE --k K --tag TAG [--max-train-seconds S] [--seed N] [--set key=value ...]
         [--wait SECONDS]
         snapshot the code, launch a detached training run (max 3 run concurrently on the GPU;
         extra runs queue), then wait up to --wait seconds (default 540) and print the result.
  wait   RUN_ID [--timeout SECONDS]   block until the run finishes (or timeout) and print result.
  status RUN_ID                       print the status JSON.
  ledger [--last N] [--k K] [--json]  list finished runs.

Exit codes of submit/wait: 0 = done, 1 = failed, 3 = still running (call `wait RUN_ID` again).
"""
from __future__ import annotations

import argparse
import copy
import datetime as _dt
import importlib.util
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
import traceback
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # `python -I` does not add the script dir
import common as C  # noqa: E402

N_GPU_SLOTS = 3
MAX_TRAIN_SECONDS_CAP = 2400
DEFAULT_MAX_TRAIN_SECONDS = 900
WAIT_POLL_S = 3.0

DEFAULTS = {
    "epochs": 200,
    "batch_size": 256,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "optimizer": "adamw",        # adamw | adam | radam | nadam | sgd
    "betas": [0.9, 0.999],
    "momentum": 0.9,             # sgd only
    "scheduler": "cosine",       # cosine | onecycle | step | plateau | none
    "warmup_epochs": 3,
    "min_lr_ratio": 0.01,
    "pct_start": 0.3,            # onecycle
    "step_size": 50,             # step (epochs)
    "gamma": 0.5,                # step
    "plateau_factor": 0.5,
    "plateau_patience": 10,
    "loss": "mse",               # mse | smooth_l1 | l1 | huber
    "smooth_l1_beta": 0.01,
    "huber_delta": 0.01,
    "grad_clip": 0.0,            # 0 = off
    "x_scaling": "minmax",       # minmax ([-1, 1]) | standard | none   (fit on the TRAIN split)
    "y_scaling": "none",         # none | standard | global_standard | minmax
    "amp": False,                # bf16 autocast for training (evaluation is always fp32)
    "tf32": True,                # TF32 matmuls for training (evaluation always disables TF32)
    "ema_decay": 0.0,            # exponential moving average of weights, 0 = off
    "eval_every": 1,             # epochs between validation evaluations
    "patience": 0,               # early stopping patience in epochs, 0 = off
    "time_aware_schedule": True, # cosine: progress = max(step progress, elapsed / max_train_seconds)
    "seed": 0,
}


# ============================================================================ utilities
def _now_stamp() -> str:
    return _dt.datetime.now().strftime("%m%d-%H%M%S")


def _pid_alive(pid: int | None, create_time: float | None = None) -> bool:
    if not pid:
        return False
    try:
        import psutil
        if not psutil.pid_exists(pid):
            return False
        p = psutil.Process(pid)
        if create_time is not None and abs(p.create_time() - create_time) > 2.0:
            return False  # pid reused by another process
        return p.status() != psutil.STATUS_ZOMBIE
    except Exception:
        return False


def _parse_overrides(items: list[str] | None) -> dict:
    out = {}
    for it in items or []:
        if "=" not in it:
            raise SystemExit(f"--set expects key=value, got {it!r}")
        k, v = it.split("=", 1)
        k = k.strip()
        try:
            out[k] = json.loads(v)
        except json.JSONDecodeError:
            out[k] = v
    return out


def load_solution(path: Path):
    path = Path(path).resolve()
    name = "solution_" + re.sub(r"\W", "_", path.stem) + "_" + uuid.uuid4().hex[:6]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    if not hasattr(mod, "build_model"):
        raise AttributeError(f"{path} must define build_model(in_dim, out_dim, cfg) -> nn.Module")
    return mod


def merge_config(mod, overrides: dict | None, n_train: int | None = None,
                 max_train_seconds: float | None = None) -> dict:
    """DEFAULTS <- module CONFIG <- module adapt_config(cfg, n_train, max_train_seconds) <- CLI overrides."""
    cfg = copy.deepcopy(DEFAULTS)
    user = getattr(mod, "CONFIG", {}) or {}
    if not isinstance(user, dict):
        raise TypeError("CONFIG must be a dict")
    cfg.update(copy.deepcopy(user))
    if n_train is not None and hasattr(mod, "adapt_config"):
        new = mod.adapt_config(dict(cfg), int(n_train), float(max_train_seconds or DEFAULT_MAX_TRAIN_SECONDS))
        if not isinstance(new, dict):
            raise TypeError("adapt_config must return the (modified) config dict")
        cfg = new
    cfg.update(overrides or {})
    return cfg


def describe(mod) -> str:
    doc = (getattr(mod, "__doc__", None) or "").strip()
    return re.sub(r"\s+", " ", doc)[:600]


# ============================================================================ scaling
class Scaler:
    """Feature / target scaling fitted on the TRAIN split only."""

    def __init__(self, x_mode, y_mode, Xtr, Ytr):
        import numpy as np
        self.x_mode, self.y_mode = x_mode, y_mode
        self.x_min, self.x_max = Xtr.min(0), Xtr.max(0)
        self.x_mean, self.x_std = Xtr.mean(0), Xtr.std(0)
        if x_mode == "minmax":
            span = np.where(self.x_max - self.x_min > 0, self.x_max - self.x_min, 1.0)
            self.x_shift, self.x_scale = (self.x_max + self.x_min) / 2, span / 2
        elif x_mode == "standard":
            self.x_shift, self.x_scale = self.x_mean, np.where(self.x_std > 0, self.x_std, 1.0)
        elif x_mode == "none":
            self.x_shift, self.x_scale = np.zeros_like(self.x_min), np.ones_like(self.x_min)
        else:
            raise ValueError(f"unknown x_scaling {x_mode!r}")
        if y_mode == "none":
            self.y_shift, self.y_scale = np.zeros(Ytr.shape[1]), np.ones(Ytr.shape[1])
        elif y_mode == "standard":
            sd = Ytr.std(0)
            self.y_shift, self.y_scale = Ytr.mean(0), np.where(sd > 1e-8, sd, 1.0)
        elif y_mode == "global_standard":
            self.y_shift = np.full(Ytr.shape[1], Ytr.mean())
            self.y_scale = np.full(Ytr.shape[1], Ytr.std())
        elif y_mode == "minmax":
            lo, hi = Ytr.min(0), Ytr.max(0)
            self.y_shift, self.y_scale = (hi + lo) / 2, np.where(hi - lo > 1e-8, (hi - lo) / 2, 1.0)
        else:
            raise ValueError(f"unknown y_scaling {y_mode!r}")
        f32 = lambda a: np.asarray(a, dtype=np.float32)  # noqa: E731
        for k in ["x_min", "x_max", "x_mean", "x_std", "x_shift", "x_scale", "y_shift", "y_scale"]:
            setattr(self, k, f32(getattr(self, k)))

    def tx(self, X):
        return ((X - self.x_shift) / self.x_scale).astype("float32")

    def ty(self, Y):
        return ((Y - self.y_shift) / self.y_scale).astype("float32")

    def to_dict(self) -> dict:
        return {"x_mode": self.x_mode, "y_mode": self.y_mode,
                "formula": "x_scaled = (x_raw - x_shift) / x_scale ; y_raw = y_model * y_scale + y_shift",
                **{k: getattr(self, k).tolist() for k in
                   ["x_min", "x_max", "x_mean", "x_std", "x_shift", "x_scale", "y_shift", "y_scale"]}}


# ============================================================================ training core
def _make_optimizer(model, cfg):
    import torch
    params = [p for p in model.parameters() if p.requires_grad]
    name = str(cfg["optimizer"]).lower()
    kw = dict(lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    if name == "adamw":
        return torch.optim.AdamW(params, betas=tuple(cfg["betas"]), **kw)
    if name == "adam":
        return torch.optim.Adam(params, betas=tuple(cfg["betas"]), **kw)
    if name == "radam":
        return torch.optim.RAdam(params, betas=tuple(cfg["betas"]), **kw)
    if name == "nadam":
        return torch.optim.NAdam(params, betas=tuple(cfg["betas"]), **kw)
    if name == "sgd":
        return torch.optim.SGD(params, momentum=cfg["momentum"], nesterov=True, **kw)
    raise ValueError(f"unknown optimizer {name!r}")


def _make_scheduler(opt, cfg, steps_per_epoch, epochs, clock=None):
    import torch
    name = str(cfg["scheduler"]).lower()
    total = max(1, steps_per_epoch * epochs)
    warm = int(round(cfg["warmup_epochs"] * steps_per_epoch))
    if name == "cosine":
        mr = float(cfg["min_lr_ratio"])
        time_aware = bool(cfg.get("time_aware_schedule", True)) and clock is not None

        def f(step):
            if step < warm:
                return (step + 1) / max(1, warm)
            t = (step - warm) / max(1, total - warm)
            if time_aware and clock.get("t0") is not None:
                # progress = max(step progress, wall-clock progress): the schedule always reaches its
                # floor by the time limit, even when concurrent runs slow this one down.
                t = max(t, (time.time() - clock["t0"]) / max(1e-9, clock["budget"]))
            t = min(1.0, t)
            return mr + (1 - mr) * 0.5 * (1 + math.cos(math.pi * t))
        return torch.optim.lr_scheduler.LambdaLR(opt, f), "step"
    if name == "onecycle":
        return torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg["lr"], total_steps=total,
                                                   pct_start=cfg["pct_start"]), "step"
    if name == "step":
        return torch.optim.lr_scheduler.StepLR(opt, step_size=cfg["step_size"], gamma=cfg["gamma"]), "epoch"
    if name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, factor=cfg["plateau_factor"], patience=cfg["plateau_patience"],
            min_lr=cfg["lr"] * cfg["min_lr_ratio"]), "plateau"
    if name in ("none", "constant"):
        return None, None
    raise ValueError(f"unknown scheduler {name!r}")


def _make_loss(cfg):
    import torch.nn.functional as F
    name = str(cfg["loss"]).lower()
    if name == "mse":
        return lambda p, y: F.mse_loss(p, y)
    if name == "smooth_l1":
        return lambda p, y: F.smooth_l1_loss(p, y, beta=cfg["smooth_l1_beta"])
    if name == "l1":
        return lambda p, y: F.l1_loss(p, y)
    if name == "huber":
        return lambda p, y: F.huber_loss(p, y, delta=cfg["huber_delta"])
    raise ValueError(f"unknown loss {name!r}")


class _EMA:
    def __init__(self, model, decay):
        self.decay = float(decay)
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    def update(self, model):
        import torch
        with torch.no_grad():
            d = self.decay
            for s, p in zip(self.shadow.parameters(), model.parameters()):
                s.mul_(d).add_(p.detach(), alpha=1 - d)
            for s, b in zip(self.shadow.buffers(), model.buffers()):
                s.copy_(b)


def predict_raw(model, X_scaled, y_scale, y_shift, bs=2048):
    """Model prediction mapped back to raw absorptivity; fp32, TF32 disabled."""
    import torch
    was_training = model.training
    model.eval()
    m_tf32, c_tf32 = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    outs = []
    try:
        with torch.no_grad():
            for i in range(0, X_scaled.shape[0], bs):
                out = model(X_scaled[i:i + bs]).float()
                if out.dim() != 2 or out.shape[1] != C.SPEC_DIM:
                    raise ValueError(f"model output shape {tuple(out.shape)} != (B, {C.SPEC_DIM})")
                outs.append(out * y_scale + y_shift)
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = m_tf32, c_tf32
        model.train(was_training)
    return torch.cat(outs, 0)


def mse_raw(model, X_scaled, Y_raw, y_scale, y_shift) -> float:
    import torch
    pred = predict_raw(model, X_scaled, y_scale, y_shift)
    return float(((pred.double() - Y_raw.double()) ** 2).mean().item())


def fit(mod, cfg, Xtr_s, Ytr_s, Xva_s, Yva_raw, y_scale, y_shift, *, epochs, max_seconds,
        on_eval=None, log_rows=None):
    """Generic training loop shared by `check --time-epochs` and real runs."""
    import torch
    dev = Xtr_s.device
    torch.manual_seed(int(cfg["seed"]))
    model = mod.build_model(C.GEOM_DIM, C.SPEC_DIM, cfg).to(dev)
    n_params = int(sum(p.numel() for p in model.parameters()))
    n = Xtr_s.shape[0]
    bs = int(min(cfg["batch_size"], n))
    steps = math.ceil(n / bs)
    opt = mod.make_optimizer(model, cfg) if hasattr(mod, "make_optimizer") else _make_optimizer(model, cfg)
    clock = {"t0": None, "budget": float(max_seconds)}
    if hasattr(mod, "make_scheduler"):
        sched, smode = mod.make_scheduler(opt, cfg, steps, epochs)
    else:
        sched, smode = _make_scheduler(opt, cfg, steps, epochs, clock)
    custom = getattr(mod, "compute_loss", None)
    loss_fn = _make_loss(cfg)
    ema = _EMA(model, cfg["ema_decay"]) if float(cfg["ema_decay"]) > 0 else None
    use_amp = bool(cfg["amp"])
    torch.backends.cuda.matmul.allow_tf32 = bool(cfg["tf32"])
    torch.backends.cudnn.allow_tf32 = bool(cfg["tf32"])
    eval_every = max(1, int(cfg["eval_every"]))
    patience = int(cfg["patience"])
    clip = float(cfg["grad_clip"])

    best, best_state, best_epoch, since_best = math.inf, None, -1, 0
    stop_reason, ep_done = "max_epochs", 0
    t0 = time.time()
    clock["t0"] = t0
    epoch_times = []
    for ep in range(epochs):
        te = time.time()
        model.train()
        perm = torch.randperm(n, device=dev)
        acc = torch.zeros((), device=dev)
        nb = 0
        for s in range(steps):
            idx = perm[s * bs:(s + 1) * bs]
            if idx.numel() < 2:
                continue
            xb, yb = Xtr_s[idx], Ytr_s[idx]
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                if custom is not None:
                    loss = custom(model, xb, yb, cfg)
                else:
                    loss = loss_fn(model(xb).float(), yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step()
            if ema is not None:
                ema.update(model)
            if smode == "step":
                sched.step()
            acc += loss.detach().float()
            nb += 1
        ep_done = ep + 1
        train_loss = float((acc / max(1, nb)).item())
        epoch_times.append(time.time() - te)
        if not math.isfinite(train_loss):
            stop_reason = "non_finite_loss"
            break
        do_eval = (ep_done % eval_every == 0) or ep_done == epochs
        vm = None
        if do_eval:
            em = ema.shadow if ema is not None else model
            vm = mse_raw(em, Xva_s, Yva_raw, y_scale, y_shift)
            if vm < best:
                best, best_epoch, since_best = vm, ep_done, 0
                best_state = {k: v.detach().clone() for k, v in em.state_dict().items()}
            else:
                since_best += eval_every
            if smode == "plateau":
                sched.step(vm)
        if smode == "epoch":
            sched.step()
        if log_rows is not None:
            log_rows.append({"epoch": ep_done, "train_loss": train_loss, "val_mse": vm,
                             "lr": opt.param_groups[0]["lr"], "elapsed_s": round(time.time() - t0, 2)})
        if on_eval is not None:
            on_eval(ep_done, epochs, train_loss, vm, best, time.time() - t0)
        if patience > 0 and since_best >= patience:
            stop_reason = "early_stopping"
            break
        if time.time() - t0 > max_seconds:
            stop_reason = "time_limit"
            break
    final_model = ema.shadow if ema is not None else model
    if best_state is not None:
        final_model.load_state_dict(best_state)
    return {"model": final_model, "n_params": n_params, "best_val_mse": best, "best_epoch": best_epoch,
            "epochs_run": ep_done, "stop_reason": stop_reason, "train_time_s": time.time() - t0,
            "sec_per_epoch": (sum(epoch_times) / len(epoch_times)) if epoch_times else None,
            "batch_size": bs, "steps_per_epoch": steps}


# ============================================================================ run (detached)
class _Status:
    def __init__(self, rd: Path):
        self.path = rd / "status.json"
        self.data = C.read_json(self.path, default={}) or {}
        self._last = 0.0

    def set(self, **kw):
        self.data.update(kw)
        self.data["heartbeat"] = time.time()
        self._last = time.time()
        C.write_json_atomic(self.path, self.data)

    def beat(self, min_interval=3.0, **kw):
        self.data.update(kw)
        if time.time() - self._last >= min_interval:
            self.set()


def _acquire_gpu_slot(st: _Status) -> C.FileLock:
    slots = C.RUNS / ".gpu_slots"
    while True:
        for i in range(N_GPU_SLOTS):
            lk = C.FileLock(slots / f"slot{i}.lock", timeout=None)
            if lk.try_acquire():
                lk.index = i
                return lk
        st.beat(state="queued")
        time.sleep(2.0)


def cmd_run(run_id: str) -> int:
    import numpy as np
    import psutil
    rd = C.RUNS / run_id
    req = C.read_json(rd / "request.json")
    st = _Status(rd)
    st.set(state="queued", pid=os.getpid(), create_time=psutil.Process().create_time(), queued_at=C.now_iso())
    t_q = time.time()
    slot = _acquire_gpu_slot(st)
    queue_time = time.time() - t_q
    started_at = C.now_iso()
    st.set(state="running", slot=slot.index, started_at=started_at)
    t_setup = time.time()
    rows: list[dict] = []
    try:
        import torch
        mod = load_solution(rd / "solution.py")
        torch.backends.cudnn.benchmark = True
        dev = torch.device("cuda")
        Xtr, Ytr, Xva, Yva = C.load_dataset(req["k"])
        cfg = merge_config(mod, req.get("overrides"), n_train=Xtr.shape[0],
                           max_train_seconds=req["max_train_seconds"])
        sc = Scaler(cfg["x_scaling"], cfg["y_scaling"], Xtr, Ytr)
        T = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(dev)  # noqa: E731
        Xtr_s, Ytr_s, Xva_s, Yva_raw = T(sc.tx(Xtr)), T(sc.ty(Ytr)), T(sc.tx(Xva)), T(Yva)
        y_scale, y_shift = T(sc.y_scale), T(sc.y_shift)
        setup_time = time.time() - t_setup

        def on_eval(ep, E, tl, vm, best, el):
            st.beat(epoch=ep, epochs=E, train_loss=tl, last_val_mse=vm, best_val_mse=best, elapsed_s=round(el, 1))

        out = fit(mod, cfg, Xtr_s, Ytr_s, Xva_s, Yva_raw, y_scale, y_shift, epochs=int(cfg["epochs"]),
                  max_seconds=float(req["max_train_seconds"]), on_eval=on_eval, log_rows=rows)
        model = out["model"]
        if not math.isfinite(out["best_val_mse"]):
            raise RuntimeError(f"training produced no finite validation MSE (stop: {out['stop_reason']})")
        val_check = mse_raw(model, Xva_s, Yva_raw, y_scale, y_shift)
        torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, rd / "best_model.pt")
        C.write_json_atomic(rd / "scaler.json", sc.to_dict())
        result = {
            "run_id": run_id, "tag": req["tag"], "status": "done",
            "k": req["k"], "n_train": int(Xtr.shape[0]), "n_val": int(Xva.shape[0]),
            "val_mse": val_check, "val_mse_during_training": out["best_val_mse"],
            "best_epoch": out["best_epoch"], "epochs_run": out["epochs_run"], "epochs_planned": int(cfg["epochs"]),
            "stop_reason": out["stop_reason"], "n_params": out["n_params"],
            "train_time_s": round(out["train_time_s"], 2), "setup_time_s": round(setup_time, 2),
            "queue_time_s": round(queue_time, 2), "sec_per_epoch": out["sec_per_epoch"],
            "batch_size": out["batch_size"], "steps_per_epoch": out["steps_per_epoch"],
            "max_train_seconds": req["max_train_seconds"],
            "source_code": req["source_code"], "code_sha256": req["code_sha256"],
            "solution_snapshot": C.rel(rd / "solution.py"), "checkpoint": C.rel(rd / "best_model.pt"),
            "description": describe(mod), "model_repr": str(model)[:4000],
            "config": cfg, "overrides": req.get("overrides") or {},
            "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
            "submitted_at": req["submitted_at"], "started_at": started_at, "finished_at": C.now_iso(),
        }
        C.write_json_atomic(rd / "result.json", result)
        _write_log(rd, rows)
        ledger_rec = {k: v for k, v in result.items() if k not in ("model_repr",)}
        C.append_jsonl(C.LEDGER, ledger_rec)
        st.set(state="done", val_mse=val_check, finished_at=result["finished_at"])
        return 0
    except BaseException:  # noqa: BLE001 - report every failure to the agent
        tb = traceback.format_exc()
        _write_log(rd, rows)
        fail = {"run_id": run_id, "tag": req.get("tag"), "status": "failed", "k": req.get("k"),
                "source_code": req.get("source_code"), "code_sha256": req.get("code_sha256"),
                "error": tb[-6000:], "submitted_at": req.get("submitted_at"), "started_at": started_at,
                "finished_at": C.now_iso(), "train_time_s": round(time.time() - t_setup, 2),
                "queue_time_s": round(queue_time, 2), "overrides": req.get("overrides") or {}}
        C.write_json_atomic(rd / "result.json", fail)
        C.append_jsonl(C.LEDGER, {k: v for k, v in fail.items() if k != "error"} | {"error_tail": tb[-800:]})
        st.set(state="failed", error=tb[-3000:], finished_at=fail["finished_at"])
        return 1
    finally:
        slot.release()


def _write_log(rd: Path, rows: list[dict]) -> None:
    if not rows:
        return
    import csv
    with open(rd / "train_log.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


# ============================================================================ CLI commands
def _print_result(rd: Path, st: dict) -> int:
    res = C.read_json(rd / "result.json", default=None)
    if res is None:
        res = {"run_id": rd.name, "status": st.get("state"), "error": st.get("error")}
    if res.get("status") == "done":
        short = {k: res.get(k) for k in ["run_id", "status", "k", "n_train", "n_val", "val_mse", "best_epoch",
                                         "epochs_run", "epochs_planned", "stop_reason", "n_params",
                                         "train_time_s", "sec_per_epoch", "queue_time_s", "source_code",
                                         "solution_snapshot", "checkpoint"]}
        print(f"DONE {res['run_id']}: val_mse={res['val_mse']:.6e} k={res['k']} params={res['n_params']:,} "
              f"epochs={res['epochs_run']}/{res['epochs_planned']} ({res['stop_reason']}) "
              f"train_time={res['train_time_s']:.1f}s")
        print("RESULT_JSON: " + json.dumps(short))
        return 0
    print(f"FAILED {rd.name}")
    print((res.get("error") or st.get("error") or "unknown error")[-4000:])
    print("RESULT_JSON: " + json.dumps({"run_id": rd.name, "status": "failed", "k": res.get("k")}))
    return 1


def cmd_wait(run_id: str, timeout: float) -> int:
    rd = C.RUNS / run_id
    if not rd.exists():
        print(f"unknown run {run_id}")
        return 1
    t0 = time.time()
    while True:
        st = C.read_json(rd / "status.json", default={}) or {}
        state = st.get("state")
        if state in ("done", "failed"):
            return _print_result(rd, st)
        pid = st.get("pid")
        dead = (pid is not None and not _pid_alive(pid, st.get("create_time"))) or \
               (pid is None and time.time() - st.get("submitted_ts", time.time()) > 180)
        if dead:
            st = C.read_json(rd / "status.json", default={}) or {}  # re-read: may have finished meanwhile
            if st.get("state") in ("done", "failed"):
                return _print_result(rd, st)
            st["state"] = "failed"
            st["error"] = "runner process died without reporting (see stdout.log):\n" + \
                          _tail(rd / "stdout.log", 3000)
            C.write_json_atomic(rd / "status.json", st)
            return _print_result(rd, st)
        if time.time() - t0 >= timeout:
            prog = {k: st.get(k) for k in ["state", "epoch", "epochs", "best_val_mse", "last_val_mse", "elapsed_s"]}
            print(f"RUNNING {run_id}: {json.dumps(prog)}")
            print(f"Still running. Call: python -I framework/harness.py wait {run_id}")
            return 3
        time.sleep(WAIT_POLL_S)


def _tail(path: Path, n: int) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")[-n:]
    except FileNotFoundError:
        return ""


def cmd_submit(a) -> int:
    code = Path(a.code).resolve()
    if not code.is_file():
        print(f"code file not found: {code}")
        return 1
    k = int(a.k)
    granted = C.granted_size()
    if not 1 <= k <= granted:
        print(f"k={k} not allowed: data acquired so far k_current={granted}. Acquire data first with "
              f"`python -I framework/prepare_dataset.py --k K` (only as instructed by the Controller).")
        return 1
    mts = float(min(max(30.0, a.max_train_seconds), MAX_TRAIN_SECONDS_CAP))
    tag = re.sub(r"[^A-Za-z0-9_.-]", "_", a.tag)[:60]
    run_id = f"{tag}-{_now_stamp()}-{uuid.uuid4().hex[:4]}"
    rd = C.RUNS / run_id
    rd.mkdir(parents=True)
    shutil.copy2(code, rd / "solution.py")
    overrides = _parse_overrides(a.set)
    if a.seed is not None:
        overrides["seed"] = a.seed
    req = {"run_id": run_id, "tag": tag, "source_code": C.rel(code), "code_sha256": C.sha256_file(code),
           "k": k, "max_train_seconds": mts, "overrides": overrides, "submitted_at": C.now_iso()}
    C.write_json_atomic(rd / "request.json", req)
    C.write_json_atomic(rd / "status.json", {"state": "submitted", "submitted_ts": time.time()})
    log = open(rd / "stdout.log", "ab")
    kw = {}
    if os.name == "nt":
        kw["creationflags"] = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kw["start_new_session"] = True
    subprocess.Popen([sys.executable, "-I", str(Path(__file__).resolve()), "_run", run_id],
                     stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, close_fds=True,
                     cwd=str(C.ROOT), **kw)
    log.close()
    print(f"RUN_ID: {run_id}  (k={k}, max_train_seconds={mts:.0f}, overrides={overrides})", flush=True)
    return cmd_wait(run_id, a.wait)


def cmd_check(a) -> int:
    import numpy as np
    import torch
    mod = load_solution(Path(a.code))
    n_train = None
    if a.k:
        n_train = int(C.split_positions(int(a.k))[0].size)
    cfg = merge_config(mod, _parse_overrides(a.set), n_train=n_train, max_train_seconds=a.max_train_seconds)
    dev = torch.device("cuda")
    torch.manual_seed(int(cfg["seed"]))
    model = mod.build_model(C.GEOM_DIM, C.SPEC_DIM, cfg).to(dev)
    n_params = int(sum(p.numel() for p in model.parameters()))
    with torch.no_grad():
        out = model.eval()(torch.rand(8, C.GEOM_DIM, device=dev) * 2 - 1)
    info = {"ok": tuple(out.shape) == (8, C.SPEC_DIM) and bool(torch.isfinite(out).all()),
            "output_shape": list(out.shape), "n_params": n_params, "config": cfg,
            "description": describe(mod)}
    del model
    if a.time_epochs and a.k:
        k = int(a.k)
        if k > C.granted_size():
            print(f"k={k} exceeds acquired data (k_current={C.granted_size()})")
            return 1
        Xtr, Ytr, Xva, Yva = C.load_dataset(k)
        sc = Scaler(cfg["x_scaling"], cfg["y_scaling"], Xtr, Ytr)
        T = lambda x: torch.from_numpy(np.ascontiguousarray(x)).to(dev)  # noqa: E731
        res = fit(mod, cfg, T(sc.tx(Xtr)), T(sc.ty(Ytr)), T(sc.tx(Xva)), T(Yva), T(sc.y_scale), T(sc.y_shift),
                  epochs=int(a.time_epochs), max_seconds=300)
        info.update({"timed_k": k, "timed_epochs": res["epochs_run"], "sec_per_epoch": res["sec_per_epoch"],
                     "val_mse_after_timed_epochs": res["best_val_mse"],
                     "estimated_seconds_for_configured_epochs": (res["sec_per_epoch"] or 0) * int(cfg["epochs"])})
    spe = info.get("sec_per_epoch")
    print(f"CHECK {'OK' if info['ok'] else 'FAILED'}: output_shape={info['output_shape']} "
          f"params={n_params:,} epochs={cfg['epochs']} batch={cfg['batch_size']}"
          + (f" | k={info['timed_k']}: {spe:.3f} s/epoch -> ~{info['estimated_seconds_for_configured_epochs']:.0f} s "
             f"for {cfg['epochs']} epochs (solo GPU; concurrent runs are slower)" if spe else ""))
    print("CHECK_JSON: " + json.dumps(info, default=str))
    return 0 if info["ok"] else 1


def cmd_status(run_id: str) -> int:
    st = C.read_json(C.RUNS / run_id / "status.json", default=None)
    print(json.dumps(st, indent=1))
    return 0 if st else 1


def cmd_ledger(a) -> int:
    rows = C.read_jsonl(C.LEDGER)
    if a.k is not None:
        rows = [r for r in rows if r.get("k") == a.k]
    if a.best:
        rows = sorted([r for r in rows if r.get("status") == "done"], key=lambda r: r["val_mse"])
    rows = rows[-a.last:] if (a.last and not a.best) else rows[: a.last or None]
    if a.json:
        print(json.dumps(rows, indent=1))
        return 0
    print(f"{'run_id':<44} {'k':>6} {'val_mse':>11} {'params':>11} {'train_s':>8} {'ep':>9}  description")
    for r in rows:
        vm = f"{r['val_mse']:.4e}" if r.get("val_mse") is not None else "FAILED"
        ep = f"{r.get('epochs_run', '-')}/{r.get('epochs_planned', '-')}"
        print(f"{r['run_id']:<44} {r.get('k', 0):>6} {vm:>11} {r.get('n_params', 0) or 0:>11,} "
              f"{r.get('train_time_s', 0) or 0:>8.1f} {ep:>9}  {(r.get('description') or '')[:70]}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("check")
    p.add_argument("--code", required=True)
    p.add_argument("--k", type=int)
    p.add_argument("--time-epochs", type=int, default=0)
    p.add_argument("--max-train-seconds", type=float, default=DEFAULT_MAX_TRAIN_SECONDS)
    p.add_argument("--set", action="append")
    p = sub.add_parser("submit")
    p.add_argument("--code", required=True)
    p.add_argument("--k", type=int, required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--max-train-seconds", type=float, default=DEFAULT_MAX_TRAIN_SECONDS)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--set", action="append")
    p.add_argument("--wait", type=float, default=540)
    p = sub.add_parser("wait")
    p.add_argument("run_id")
    p.add_argument("--timeout", type=float, default=540)
    p = sub.add_parser("status")
    p.add_argument("run_id")
    p = sub.add_parser("ledger")
    p.add_argument("--last", type=int, default=30)
    p.add_argument("--k", type=int, default=None)
    p.add_argument("--best", action="store_true")
    p.add_argument("--json", action="store_true")
    p = sub.add_parser("_run")
    p.add_argument("run_id")
    a = ap.parse_args(argv)
    if a.cmd == "check":
        return cmd_check(a)
    if a.cmd == "submit":
        return cmd_submit(a)
    if a.cmd == "wait":
        return cmd_wait(a.run_id, a.timeout)
    if a.cmd == "status":
        return cmd_status(a.run_id)
    if a.cmd == "ledger":
        return cmd_ledger(a)
    if a.cmd == "_run":
        return cmd_run(a.run_id)
    return 2


if __name__ == "__main__":
    sys.exit(main())
