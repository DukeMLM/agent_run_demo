#!/usr/bin/env python
"""Diagnostic plot of the Forward_Train history: validation MSE vs data set size (manuscript Figure 5).

    python -I framework/plot_history.py --history phase2_forward/memory/history.json \
        --out phase2_forward/val_mse_vs_dataset_size.png [--candidates-out PATH]

history.json is a list of events {round, k, val_mse, action ('generate'|'test'), reason, run_id, ...};
the first event is the Initialization (reason 'Initialization', action 'generate').
Blue = Tested (latest code re-evaluated on the grown data set), red = Generated (new code from AIDE).
--candidates-out additionally plots every candidate trained by the harness (from the run ledger).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--history", default=str(C.MEMORY / "history.json"))
    ap.add_argument("--out", default=str(C.PHASE2 / "val_mse_vs_dataset_size.png"))
    ap.add_argument("--candidates-out", default=None)
    ap.add_argument("--title", default="Validation MSE vs. Dataset Size")
    a = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    hist = json.loads(Path(a.history).read_text(encoding="utf-8"))
    hist = [h for h in hist if h.get("val_mse") is not None]
    ks = [h["k"] for h in hist]
    ms = [h["val_mse"] for h in hist]

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.plot(ks, ms, "-", color="0.75", lw=1.2, zorder=1)
    tested = [(h["k"], h["val_mse"]) for h in hist if h["action"] == "test"]
    gen = [(h["k"], h["val_mse"]) for h in hist if h["action"] == "generate"]
    if tested:
        ax.scatter(*zip(*tested), s=46, color="#3B4BA8", label="Tested", zorder=3)
    if gen:
        ax.scatter(*zip(*gen), s=46, color="#D62728", label="Generated", zorder=3)
    ax.axhline(C.TARGET_MSE, ls="--", lw=2.2, color="#D62728", label=f"Target MSE = {C.TARGET_MSE:g}")
    ax.axhline(C.STRETCH_MSE, ls="-.", lw=1.6, color="#2CA02C", label=f"Stretch goal = {C.STRETCH_MSE:g}")
    last = hist[-1]
    ax.annotate(f"final: {last['val_mse']:.3e}\n(k = {last['k']:,})", xy=(last["k"], last["val_mse"]),
                xytext=(-150, 6), textcoords="offset points", fontsize=9, va="center",
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.7", alpha=0.95),
                arrowprops=dict(arrowstyle="->", color="0.3"), zorder=4)
    ax.set_xlabel("Dataset Size", fontsize=12)
    ax.set_ylabel("Validation MSE", fontsize=12)
    ax.set_title(a.title, fontsize=13)
    ax.legend(title="Action", fontsize=10)
    ymax = max(ms + [C.TARGET_MSE]) * 1.08
    ymin = min(ms + [C.STRETCH_MSE]) * 0.85
    ax.set_ylim(ymin, ymax)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=150)
    plt.close(fig)
    print(f"wrote {a.out}")

    if a.candidates_out:
        rows = [r for r in C.read_jsonl(C.LEDGER) if r.get("status") == "done"]
        chosen = {h.get("run_id") for h in hist}
        fig, ax = plt.subplots(figsize=(8.5, 5.5))
        other = [(r["k"], r["val_mse"]) for r in rows if r["run_id"] not in chosen]
        sel = [(r["k"], r["val_mse"]) for r in rows if r["run_id"] in chosen]
        if other:
            ax.scatter(*zip(*other), s=14, color="0.6", alpha=0.7, label=f"other candidates ({len(other)})")
        if sel:
            ax.scatter(*zip(*sel), s=40, color="#D62728", label="history entries")
        ax.plot(ks, ms, "-", color="#D62728", lw=0.8, alpha=0.6)
        ax.axhline(C.TARGET_MSE, ls="--", color="#D62728", lw=1.5)
        ax.axhline(C.STRETCH_MSE, ls="-.", color="#2CA02C", lw=1.2)
        ax.set_yscale("log")
        ax.set_xlabel("Dataset Size")
        ax.set_ylabel("Validation MSE (log)")
        ax.set_title(f"All harness runs ({len(rows)} trained candidates)")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.25, which="both")
        fig.tight_layout()
        fig.savefig(a.candidates_out, dpi=150)
        plt.close(fig)
        print(f"wrote {a.candidates_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
