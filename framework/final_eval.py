#!/usr/bin/env python
"""Independent re-evaluation of a trained harness run (used by the Reporter / verifiers only).

    python -I framework/final_eval.py --run RUN_ID [--test] [--figure PATH] [--out JSON]

Rebuilds the model from the run's code snapshot and checkpoint, recomputes the validation MSE on
the canonical 10:1 split of D_k and, with --test, the MSE on the independent test set (never used
for model selection), with per-sample statistics and a best / median / worst figure (Figure 6 style).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
import harness as H  # noqa: E402


def load_run(run_id: str, device):
    import torch
    rd = C.RUNS / run_id
    res = C.read_json(rd / "result.json")
    if not res or res.get("status") != "done":
        raise SystemExit(f"run {run_id} has no successful result")
    mod = H.load_solution(rd / "solution.py")
    cfg = res["config"]
    torch.manual_seed(int(cfg["seed"]))
    model = mod.build_model(C.GEOM_DIM, C.SPEC_DIM, cfg).to(device)
    sd = torch.load(rd / "best_model.pt", map_location=device, weights_only=True)
    model.load_state_dict(sd, strict=True)
    model.eval()
    sc = C.read_json(rd / "scaler.json")
    return rd, res, model, sc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--figure", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    import numpy as np
    import torch
    dev = torch.device("cuda")
    rd, res, model, sc = load_run(a.run, dev)
    T = lambda x: torch.from_numpy(np.ascontiguousarray(np.asarray(x, dtype=np.float32))).to(dev)  # noqa: E731
    x_shift, x_scale = np.asarray(sc["x_shift"], np.float32), np.asarray(sc["x_scale"], np.float32)
    y_scale, y_shift = T(sc["y_scale"]), T(sc["y_shift"])

    _, _, Xva, Yva = C.load_dataset(res["k"], enforce_grant=False)
    pred_va = H.predict_raw(model, T((Xva - x_shift) / x_scale), y_scale, y_shift)
    val_mse = float(((pred_va.double() - T(Yva).double()) ** 2).mean())
    out = {"run_id": a.run, "k": res["k"], "n_val": int(Xva.shape[0]), "val_mse_recomputed": val_mse,
           "val_mse_reported": res["val_mse"],
           "val_rel_diff": abs(val_mse - res["val_mse"]) / res["val_mse"], "n_params": res["n_params"]}

    if a.test:
        Xte, Yte = C.load_test()
        pred = H.predict_raw(model, T((Xte - x_shift) / x_scale), y_scale, y_shift).double()
        per = ((pred - T(Yte).double()) ** 2).mean(1).cpu().numpy()
        order = np.argsort(per)
        idx = {"best": int(order[0]), "median": int(order[len(order) // 2]), "worst": int(order[-1])}
        out.update({"n_test": int(Xte.shape[0]), "test_mse": float(per.mean()),
                    "test_mse_median": float(np.median(per)), "test_mse_p95": float(np.percentile(per, 95)),
                    "test_mse_max": float(per.max()), "test_mse_min": float(per.min()), "examples": idx})
        if a.figure:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            P = pred.cpu().numpy()
            fig, axs = plt.subplots(1, 4, figsize=(18, 4.2))
            x = np.arange(C.SPEC_DIM)
            for ax, (name, i) in zip(axs[:3], idx.items()):
                ax.plot(x, Yte[i], color="#D62728", lw=1.6, label="ground truth (CST)")
                ax.plot(x, P[i], color="#1F77B4", lw=1.2, ls="--", label="forward model")
                ax.set_title(f"{name}: test #{i}, MSE = {per[i]:.2e}")
                ax.set_xlabel("spectral point index")
                ax.set_ylabel("absorptivity")
                ax.legend(fontsize=8)
            axs[3].hist(per, bins=np.logspace(np.log10(max(per.min(), 1e-7)), np.log10(per.max()), 50),
                        color="0.4")
            axs[3].set_xscale("log")
            axs[3].axvline(per.mean(), color="#D62728", ls="--", label=f"mean {per.mean():.2e}")
            axs[3].set_title("Per-sample test MSE distribution")
            axs[3].legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(a.figure, dpi=140)
            plt.close(fig)
            out["figure"] = C.rel(a.figure)
    js = json.dumps(out, indent=1)
    print(js)
    if a.out:
        Path(a.out).write_text(js, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
