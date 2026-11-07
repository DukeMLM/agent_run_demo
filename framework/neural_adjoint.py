#!/usr/bin/env python
"""Neural_Adjoint tool (manuscript Sec. 2.4, Figure 3; method of Ren, Padilla & Malof, NeurIPS 2020).

Repurposes a trained forward surrogate f(g) -> s for inverse design WITHOUT any retraining:
the network weights are frozen and gradients of the spectral loss are taken with respect to the
INPUT geometry, which is updated iteratively from many random initialisations:

    z in [-1, 1]^14  (training-range normalised geometry, g = g_mid + z * g_half)
    L(z) = mean_j (f(g(z))_j - s*_j)^2  +  w_bdy * sum_d ReLU(|z_d| - 1)        (boundary loss)
    z <- Adam(dL/dz)  for `steps` iterations,  n_init candidates in parallel
    after optimisation every candidate is projected into the training domain (|z| <= 1),
    re-scored with the forward model, ranked, and the lowest-error geometry is returned.

    Stage 1 (global search) runs under bf16 autocast for speed; stage 2 (local polish) continues the
    polish_top best candidates for polish_steps exact-fp32 Adam steps (lr 1e-3 -> 1e-6). All candidates
    of both stages are re-scored in full fp32 (TF32 off) before ranking. On 3 validation spectra of the
    Phase 2 model (never the Phase 3 target), bf16 stage 1 matched an all-fp32 run within 0.4% in best MSE
    at 1.9x the speed (scratchpad study na_bench/conv.py, 2026-11-06). The polish was lengthened from 200
    steps (lr 1e-3 -> 1e-5) to 2000 steps (lr 1e-3 -> 1e-6) after a first Phase 3 run returned a design 4.9%
    above its local basin floor: on the same 3 validation spectra the gap of the best design to a long
    Adam + L-BFGS reference fell from 0.83 / 0.00 / 8.21% (200 steps) to 0.30 / 0.00 / 0.29% (2000 steps), at
    14.5 s more per design (scratchpad study na_bench/polish_study.py, 2026-11-07).

Expected model directory (produced by the Code_Modify / Modifier agent):
    forward_model.py   defines  load_forward_model(model_dir: str, device) -> torch.nn.Module
                       mapping RAW geometry (B, 14) -> RAW absorptivity (B, 2001), differentiable
                       w.r.t. its input, scaler(s) embedded, weights loaded, eval mode.
    scaler.json        must contain "x_min" and "x_max" (raw 14-D training range).
    (weights file(s) referenced by forward_model.py, e.g. forward_model.pt)

Usage (from the repo root):
    python -I framework/neural_adjoint.py --model-dir phase3_inverse/na_model \
        --target phase3_inverse/target_spectrum.csv --out-dir phase3_inverse/na_result \
        [--gt-geometry phase3_inverse/target_info.json] [--check-model-dir DIR]
    python -I framework/neural_adjoint.py --model-dir ... --targets-idx data/processed/inverse_test_idx.npy \
        --out-dir phase3_inverse/na_batch [--check-model-dir DIR]          # statistics over test targets
Fixed hyperparameters (precoded tool): n_init 2048, steps 500, Adam lr 0.01 cosine-decayed to 1e-3,
boundary weight 1.0, seed 0; polish: top 64 candidates, 2000 fp32 steps, lr 1e-3 cosine-decayed to 1e-6.
Batch mode: --n-targets N uses the first N indices of --targets-idx; it is resumable (rows are appended to
OUT/na_batch_rows.jsonl) and --max-seconds S stops starting new targets after S seconds (exit code 3 =
partial, rerun the same command; exit 0 + "BATCH COMPLETE" = done, na_batch.json and histogram written).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import time
import uuid
from pathlib import Path

sys.dont_write_bytecode = True  # leave no __pycache__ in framework/ or in the model directories
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

import numpy as np  # noqa: E402

HP = {"n_init": 2048, "steps": 500, "lr": 0.01, "lr_final": 1e-3, "bdy_weight": 1.0, "seed": 0, "top": 10,
      "amp_bf16": True, "polish_top": 64, "polish_steps": 2000, "polish_lr": 1e-3, "polish_lr_final": 1e-6}
WARMUP_HP = {**HP, "n_init": 64, "steps": 5, "polish_top": 8, "polish_steps": 3}


def load_model_dir(model_dir: Path, device):
    import torch
    model_dir = Path(model_dir).resolve()
    spec = importlib.util.spec_from_file_location(f"na_fwd_{uuid.uuid4().hex[:6]}", model_dir / "forward_model.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    model = mod.load_forward_model(str(model_dir), device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    sc = json.loads((model_dir / "scaler.json").read_text(encoding="utf-8"))
    x_min = torch.tensor(sc["x_min"], dtype=torch.float32, device=device)
    x_max = torch.tensor(sc["x_max"], dtype=torch.float32, device=device)
    with torch.no_grad():
        out = model(((x_min + x_max) / 2)[None])
    if tuple(out.shape) != (1, C.SPEC_DIM):
        raise ValueError(f"forward model output {tuple(out.shape)} != (1, {C.SPEC_DIM})")
    return model, x_min, x_max


def _fp32_eval(fn):
    import torch
    m, c = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        with torch.no_grad():
            return fn()
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = m, c


def _adam_on_inputs(model, z0, target, mid, half, steps, lr, lr_final, bdy_weight, amp):
    """NA optimizer: Adam on the normalised input z (weights frozen), MSE + boundary loss, cosine lr."""
    import torch
    z = z0.detach().clone().requires_grad_(True)
    opt = torch.optim.Adam([z], lr=lr)
    ratio = lr_final / lr
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: ratio + (1 - ratio) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / max(1, steps)))))
    for _ in range(int(steps)):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bool(amp)):
            pred = model(mid + z * half)
        mse = ((pred.float() - target) ** 2).mean(1)
        bdy = torch.relu(z.abs() - 1).sum(1)
        loss = (mse + bdy_weight * bdy).sum()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
    return z.detach()


def _rescore(model, z, target, mid, half):
    """Project into the training domain and score every candidate with the full-fp32 forward model."""
    geo = mid + z.clamp(-1, 1) * half
    pred = _fp32_eval(lambda: model(geo))
    return geo, pred, ((pred.double() - target.double()) ** 2).mean(1)


def neural_adjoint(model, x_min, x_max, target, hp=HP):
    """Return dict with the best geometry, its predicted spectrum, ranked candidates and timing.

    Stage 1 (global search): n_init random starts, `steps` Adam steps on the input (bf16 autocast for
    speed when amp_bf16). Stage 2 (local polish): the polish_top best candidates (scored in full fp32)
    get polish_steps more Adam steps in exact fp32 (TF32 off) with a small, decaying lr. Every
    candidate of both stages, projected into the domain, is re-scored in fp32 and ranked."""
    import torch
    dev = target.device
    torch.backends.cuda.matmul.allow_tf32 = False  # fp32 parts of the optimisation and all scoring exact
    torch.backends.cudnn.allow_tf32 = False
    g = torch.Generator(device=dev).manual_seed(int(hp["seed"]))
    mid, half = (x_max + x_min) / 2, (x_max - x_min) / 2
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    z_init = torch.rand(int(hp["n_init"]), C.GEOM_DIM, device=dev, generator=g) * 2 - 1
    z1 = _adam_on_inputs(model, z_init, target, mid, half, hp["steps"], hp["lr"], hp["lr_final"],
                         hp["bdy_weight"], hp.get("amp_bf16", False))
    outside = int((z1.abs() > 1).any(1).sum())
    geo1, pred1, mse1 = _rescore(model, z1, target, mid, half)
    torch.cuda.synchronize()
    t_stage1 = time.perf_counter() - t0
    best_stage1 = float(mse1.min())
    k = min(int(hp.get("polish_top", 0)), len(z1))
    if k > 0 and int(hp.get("polish_steps", 0)) > 0:
        sel = torch.argsort(mse1)[:k]
        z2 = _adam_on_inputs(model, z1.clamp(-1, 1)[sel], target, mid, half, hp["polish_steps"],
                             hp["polish_lr"], hp["polish_lr_final"], hp["bdy_weight"], False)
        geo2, pred2, mse2 = _rescore(model, z2, target, mid, half)
        geo, pred, mse = torch.cat([geo1, geo2]), torch.cat([pred1, pred2]), torch.cat([mse1, mse2])
    else:
        geo, pred, mse = geo1, pred1, mse1
    order = torch.argsort(mse)
    torch.cuda.synchronize()
    t_total = time.perf_counter() - t0
    top = order[: int(hp["top"])]
    best = int(order[0])
    return {"best_geometry": geo[best].cpu().numpy().astype(np.float64),
            "best_pred": pred[best].cpu().numpy().astype(np.float64),
            "best_mse": float(mse[best]),
            "best_mse_stage1": best_stage1,
            "best_is_polished": bool(best >= len(geo1)),
            "top_geometries": geo[top].cpu().numpy().astype(np.float64),
            "top_mse": mse[top].cpu().numpy().astype(np.float64),
            "candidates_outside_domain_before_projection": outside,
            "median_candidate_mse": float(mse1.median()),
            "stage1_time_s": t_stage1,
            "inference_time_s": t_total}


def _predict(model, geo_np, device):
    import torch
    x = torch.tensor(np.atleast_2d(geo_np), dtype=torch.float32, device=device)
    return _fp32_eval(lambda: model(x)).double().cpu().numpy()


def _gt_geometry(path: str | None):
    if not path:
        return None
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    g = d.get("ground_truth_geometry", d)
    return np.array([g[n] for n in C.GEOM_NAMES], dtype=np.float64)


def run_single(a, device):
    import torch
    t_load = time.perf_counter()
    model, x_min, x_max = load_model_dir(Path(a.model_dir), device)
    torch.cuda.synchronize()
    load_s = time.perf_counter() - t_load
    s_star = np.loadtxt(a.target, delimiter=",", dtype=np.float64).reshape(-1)
    if s_star.size != C.SPEC_DIM:
        raise ValueError(f"target has {s_star.size} values, expected {C.SPEC_DIM}")
    target = torch.tensor(s_star, dtype=torch.float32, device=device)[None]
    _ = neural_adjoint(model, x_min, x_max, target, WARMUP_HP)  # CUDA warm-up
    res = neural_adjoint(model, x_min, x_max, target)
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    g_best = res["best_geometry"]
    gt = _gt_geometry(a.gt_geometry)
    summary = {
        "method": "Neural-Adjoint (Ren, Padilla & Malof 2020): gradient descent on the input geometry of the "
                  "frozen forward surrogate from random initialisations, boundary loss, projection, ranking",
        "hyperparameters": HP, "model_dir": C.rel(a.model_dir), "target_file": C.rel(a.target),
        "optimal_geometry": {n: float(v) for n, v in zip(C.GEOM_NAMES, g_best)},
        "optimal_geometry_vector": [float(v) for v in g_best],
        "predicted_mse_vs_target": res["best_mse"],
        "inference_time_s": res["inference_time_s"],
        "inference_time_note": "wall time of the NA optimisation (initialisation, all gradient steps of both stages, "
                               "projection, re-scoring and ranking of all candidates), CUDA-synchronised, model "
                               "already loaded",
        "stage1_time_s": res["stage1_time_s"],
        "best_mse_stage1": res["best_mse_stage1"],
        "best_is_polished": res["best_is_polished"],
        "model_load_time_s": load_s,
        "median_candidate_mse": res["median_candidate_mse"],
        "candidates_outside_domain_before_projection": res["candidates_outside_domain_before_projection"],
        "top_candidates": [{"mse": float(m), "geometry": [float(v) for v in g]}
                           for m, g in zip(res["top_mse"], res["top_geometries"])],
        "training_range": {"x_min": x_min.cpu().tolist(), "x_max": x_max.cpu().tolist()},
    }
    if gt is not None:
        pred_gt = _predict(model, gt, device)[0]
        summary["ground_truth_geometry"] = {n: float(v) for n, v in zip(C.GEOM_NAMES, gt)}
        summary["surrogate_mse_of_ground_truth_geometry"] = float(((pred_gt - s_star) ** 2).mean())
        span = (x_max - x_min).cpu().numpy()
        summary["normalised_geometry_distance_to_ground_truth"] = float(np.abs((g_best - gt) / span).mean())
    if a.check_model_dir:
        cm, _, _ = load_model_dir(Path(a.check_model_dir), device)
        p2 = _predict(cm, g_best, device)[0]
        summary["independent_surrogate_check"] = {
            "model_dir": C.rel(a.check_model_dir),
            "mse_vs_target": float(((p2 - s_star) ** 2).mean()),
            "note": "spectrum of the NA design predicted by a second, independently trained forward model "
                    "(proxy for re-simulation; no CST model of the ADM cell is available)"}
        if gt is not None:
            summary["independent_surrogate_check"]["check_model_mse_of_ground_truth_geometry"] = float(
                ((_predict(cm, gt, device)[0] - s_star) ** 2).mean())
            summary["independent_surrogate_check"]["floor_note"] = (
                "error of the check model itself on the geometry that produced the target: a design scored by the "
                "check model cannot be expected to do much better than this floor")
    np.savetxt(out_dir / "optimal_geometry.csv", g_best[None], delimiter=",", fmt="%.9e",
               header=",".join(C.GEOM_NAMES), comments="")
    np.savetxt(out_dir / "predicted_spectrum.csv", res["best_pred"][None], delimiter=",", fmt="%.9e")
    (out_dir / "na_result.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    _plot_single(s_star, res["best_pred"], res["best_mse"], res["inference_time_s"], g_best, gt, x_min, x_max,
                 out_dir, summary.get("independent_surrogate_check"))
    print(json.dumps({k: summary[k] for k in ["optimal_geometry", "predicted_mse_vs_target", "inference_time_s",
                                              "model_load_time_s"]}, indent=1))
    return 0


def _plot_single(s_star, pred, mse, t_s, g_best, gt, x_min, x_max, out_dir, check):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x = np.arange(C.SPEC_DIM)
    fig, ax = plt.subplots(figsize=(10, 5.2))
    ax.plot(x, s_star, color="#D62728", lw=2.0, label="target spectrum (ground truth, CST)")
    ax.plot(x, pred, color="#1F77B4", lw=1.4, ls="--", label="forward-model prediction of NA-optimal geometry")
    ax.set_xlabel("spectral point index (2001 points)")
    ax.set_ylabel("absorptivity")
    txt = f"MSE = {mse:.3e}\nNA inference time = {t_s:.2f} s"
    if check:
        txt += f"\nindependent surrogate MSE = {check['mse_vs_target']:.3e}"
    ax.text(0.02, 0.97, txt, transform=ax.transAxes, va="top", fontsize=10,
            bbox=dict(boxstyle="round", fc="white", ec="0.7"))
    ax.set_title("Neural-Adjoint inverse design: predicted vs. target spectrum")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "inverse_design_spectrum.png", dpi=150)
    plt.close(fig)

    lo, hi = x_min.cpu().numpy(), x_max.cpu().numpy()
    fig, ax = plt.subplots(figsize=(10, 3.8))
    idx = np.arange(C.GEOM_DIM)
    ax.bar(idx - 0.2, 2 * (g_best - lo) / (hi - lo) - 1, width=0.4, color="#1F77B4", label="NA-optimal geometry")
    if gt is not None:
        ax.bar(idx + 0.2, 2 * (gt - lo) / (hi - lo) - 1, width=0.4, color="#D62728",
               label="geometry that produced the target (reference only)")
    ax.set_xticks(idx, C.GEOM_NAMES, rotation=0)
    ax.set_ylim(-1.1, 1.1)
    ax.set_ylabel("normalised value (training range -> [-1, 1])")
    ax.set_title("Inverse-designed geometry (inverse problem is one-to-many: matching spectra need not share geometry)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(out_dir / "inverse_design_geometry.png", dpi=150)
    plt.close(fig)


def run_batch(a, device):
    """Resumable: finished targets are appended to OUT/na_batch_rows.jsonl; with --max-seconds the call stops
    starting new targets after that many seconds and exits with code 3 (run the same command again)."""
    import torch
    t_call = time.perf_counter()
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_path = out_dir / "na_batch_rows.jsonl"
    idx = np.load(a.targets_idx).astype(int)
    n_avail = len(idx)
    if a.n_targets:
        idx = idx[: int(a.n_targets)]
    tag = {"model_dir": C.rel(a.model_dir), "check_model_dir": C.rel(a.check_model_dir) if a.check_model_dir else None,
           "hyperparameters": HP}
    done = {}
    if rows_path.exists():
        for ln in rows_path.read_text(encoding="utf-8").splitlines():
            if ln.strip():
                r = json.loads(ln)
                if {k: r.get(k) for k in tag} != json.loads(json.dumps(tag)):
                    raise SystemExit(f"{rows_path} holds rows from a different model/check model/HP; use a new --out-dir")
                done[int(r["test_index"])] = r
    todo = [int(i) for i in idx if int(i) not in done]
    if todo:
        model, x_min, x_max = load_model_dir(Path(a.model_dir), device)
        check = load_model_dir(Path(a.check_model_dir), device)[0] if a.check_model_dir else None
        _, Yte = C.load_test()
        Xte = np.load(C.PROCESSED / "test_g.npy").astype(np.float64)
        warm = torch.tensor(Yte[todo[0]], dtype=torch.float32, device=device)[None]
        _ = neural_adjoint(model, x_min, x_max, warm, WARMUP_HP)
        for n_new, i in enumerate(todo):
            if a.max_seconds and n_new > 0 and time.perf_counter() - t_call > a.max_seconds:
                break  # at least one new target per call, so repeated calls always make progress
            s_star = Yte[i].astype(np.float64)
            res = neural_adjoint(model, x_min, x_max, torch.tensor(s_star, dtype=torch.float32, device=device)[None])
            row = {"test_index": int(i), "na_mse": res["best_mse"], "na_mse_stage1": res["best_mse_stage1"],
                   "time_s": res["inference_time_s"],
                   "geometry": [float(v) for v in res["best_geometry"]],
                   "surrogate_mse_of_true_geometry": float(((_predict(model, Xte[i], device)[0] - s_star) ** 2).mean())}
            if check is not None:
                row["independent_surrogate_mse"] = float(
                    ((_predict(check, res["best_geometry"], device)[0] - s_star) ** 2).mean())
                row["check_model_mse_of_true_geometry"] = float(((_predict(check, Xte[i], device)[0] - s_star) ** 2).mean())
            row.update(json.loads(json.dumps(tag)))
            with open(rows_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(row) + "\n")
            done[i] = row
            print(f"[{len(done)}/{len(idx)}] test #{i}: NA MSE {res['best_mse']:.3e} in {res['inference_time_s']:.1f} s",
                  flush=True)
    remaining = [int(i) for i in idx if int(i) not in done]
    if remaining:
        print(f"BATCH PARTIAL: {len(idx) - len(remaining)}/{len(idx)} targets done; "
              f"run the same command again to continue", flush=True)
        return 3
    rows = [{k: v for k, v in done[int(i)].items() if k not in tag} for i in idx]
    check = a.check_model_dir
    na = np.array([r["na_mse"] for r in rows])
    ts = np.array([r["time_s"] for r in rows])
    stats = {"n_targets": len(rows), "n_targets_available": n_avail, "targets_idx_file": C.rel(a.targets_idx),
             "hyperparameters": HP, "model_dir": C.rel(a.model_dir),
             "na_mse_mean": float(na.mean()), "na_mse_median": float(np.median(na)),
             "na_mse_p95": float(np.percentile(na, 95)),
             "na_mse_stage1_mean": float(np.mean([r["na_mse_stage1"] for r in rows])),
             "time_per_design_mean_s": float(ts.mean()), "time_per_design_std_s": float(ts.std()),
             "surrogate_mse_of_true_geometry_mean": float(np.mean([r["surrogate_mse_of_true_geometry"] for r in rows]))}
    if check:
        ind = np.array([r["independent_surrogate_mse"] for r in rows])
        stats.update({"check_model_dir": C.rel(a.check_model_dir), "independent_surrogate_mse_mean": float(ind.mean()),
                      "independent_surrogate_mse_median": float(np.median(ind)),
                      "independent_surrogate_mse_p95": float(np.percentile(ind, 95)),
                      "check_model_mse_of_true_geometry_mean": float(
                          np.mean([r["check_model_mse_of_true_geometry"] for r in rows]))})
    (out_dir / "na_batch.json").write_text(json.dumps({"stats": stats, "rows": rows}, indent=1), encoding="utf-8")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4.2))
    allv = np.concatenate([na, ind]) if check else na
    bins = np.logspace(np.log10(max(allv.min(), 1e-7)), np.log10(allv.max() * 1.01), 30)
    ax.hist(na, bins=bins, color="#1F77B4", alpha=0.75, label="NA design vs target (inverse surrogate)")
    if check:
        ax.hist(ind, bins=bins, color="#FF7F0E", alpha=0.55, label="independent surrogate check")
    ax.axvline(np.percentile(na, 95), color="k", ls="--", lw=1, label="95% of targets (NA)")
    ax.set_xscale("log")
    ax.set_xlabel("MSE")
    ax.set_ylabel("targets")
    ax.set_title(f"Neural-Adjoint over {len(rows)} pre-registered test targets")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "na_batch_hist.png", dpi=150)
    plt.close(fig)
    print(json.dumps(stats, indent=1))
    print("BATCH COMPLETE", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--target", default=None)
    ap.add_argument("--targets-idx", default=None)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--gt-geometry", default=None)
    ap.add_argument("--check-model-dir", default=None)
    ap.add_argument("--n-targets", type=int, default=None, help="batch mode: use the first N indices only")
    ap.add_argument("--max-seconds", type=float, default=None,
                    help="batch mode: stop starting new targets after this many seconds (exit 3; rerun to resume)")
    a = ap.parse_args()
    import torch
    device = torch.device("cuda")
    if a.target:
        return run_single(a, device)
    if a.targets_idx:
        return run_batch(a, device)
    ap.error("give --target FILE or --targets-idx FILE")
    return 2


if __name__ == "__main__":
    sys.exit(main())
