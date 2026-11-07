#!/usr/bin/env python
"""Phase 1 - data preparation for the ADM benchmark (Lu, Malof & Padilla, ACS Photonics 2025).

Run from the repository root:
    python -I phase1_data/prepare_data.py

Steps
 1. Verify the downloaded archive data/raw/ADM.zip (size, SHA-256, CRC of every member).
 2. Extract it safely (zip-slip guarded) into data/extracted/.
 3. Load the four header-less CSV files
        ADM/data_g.csv, ADM/data_s.csv                 -> training pool (90 % of the simulations)
        ADM/testset/test_g.csv, ADM/testset/test_s.csv -> independent test set (10 %)
 4. Verify: geometry vectors are 14-D, spectra are 2001-D, row counts of g/s match, every value
    is finite; report value ranges, discreteness of the geometry grid and duplicate geometries.
 5. Partition
      * pool -> shuffled once (seed 2025) -> data/processed/pool_{g,s}.npy (+ pool_perm.npy).
                Forward_Train draws D_k = first k rows and splits it train:val = 10:1
                (position i is validation iff i % 11 == 10), see framework/common.py.
      * test -> data/processed/test_{g,s}.npy (held out from all model selection).
      * inverse-design targets, pre-registered BEFORE any model exists (seed 0):
                100 test indices (data/processed/inverse_test_idx.npy); the first one is the
                Phase 3 target spectrum (phase3_inverse/target_spectrum.csv).
 6. Write phase1_data/data_summary.json, phase1_data/summary.md, phase1_data/data_overview.png.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "framework"))
import common as C  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

MEMBERS = {
    "data_g": "ADM/data_g.csv",
    "data_s": "ADM/data_s.csv",
    "test_g": "ADM/testset/test_g.csv",
    "test_s": "ADM/testset/test_s.csv",
}
EXPECTED_ZIP_BYTES = 1_006_554_462  # Content-Length reported by the Duke repository


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def safe_extract(zip_path: Path, dest: Path) -> list[dict]:
    dest = dest.resolve()
    listing = []
    with zipfile.ZipFile(zip_path) as z:
        bad = z.testzip()
        if bad is not None:
            raise RuntimeError(f"CRC check failed for member {bad}")
        for info in z.infolist():
            target = (dest / info.filename).resolve()
            if dest not in target.parents and target != dest:
                raise RuntimeError(f"unsafe path in archive: {info.filename}")
            listing.append({"member": info.filename, "bytes": info.file_size,
                            "compressed_bytes": info.compress_size, "is_dir": info.is_dir()})
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if target.exists() and target.stat().st_size == info.file_size:
                continue  # idempotent re-run
            target.parent.mkdir(parents=True, exist_ok=True)
            with z.open(info) as src, open(target, "wb") as out:
                while True:
                    buf = src.read(1 << 22)
                    if not buf:
                        break
                    out.write(buf)
    return listing


def load_csv(path: Path, expected_cols: int) -> np.ndarray:
    t0 = time.time()
    df = pd.read_csv(path, header=None, dtype=np.float64, engine="c")
    arr = df.to_numpy()
    log(f"loaded {C.rel(path)} shape={arr.shape} in {time.time() - t0:.1f}s")
    if arr.ndim != 2 or arr.shape[1] != expected_cols:
        raise ValueError(f"{path}: expected {expected_cols} columns, got {arr.shape}")
    return arr


def row_keys(a: np.ndarray) -> np.ndarray:
    a32 = np.ascontiguousarray(a.astype(np.float32))
    return a32.view(np.dtype((np.void, a32.dtype.itemsize * a32.shape[1]))).ravel()


def dup_stats(pool_g: np.ndarray, test_g: np.ndarray) -> dict:
    kp, kt = row_keys(pool_g), row_keys(test_g)
    up, cp = np.unique(kp, return_counts=True)
    ut, ct = np.unique(kt, return_counts=True)
    cross = np.intersect1d(up, ut)
    return {
        "pool_unique_geometries": int(up.size),
        "pool_rows_in_duplicate_groups": int(cp[cp > 1].sum()),
        "pool_extra_duplicate_rows": int((cp[cp > 1] - 1).sum()),
        "test_unique_geometries": int(ut.size),
        "test_extra_duplicate_rows": int((ct[ct > 1] - 1).sum()),
        "geometries_shared_by_pool_and_test": int(cross.size),
        "test_rows_whose_geometry_appears_in_pool": int(np.isin(kt, cross).sum()),
    }


def geom_stats(g: np.ndarray) -> list[dict]:
    out = []
    for j, name in enumerate(C.GEOM_NAMES):
        col = g[:, j]
        uniq = np.unique(np.round(col, 6))
        d = {"index": j, "name": name, "min": float(col.min()), "max": float(col.max()),
             "mean": float(col.mean()), "std": float(col.std()), "n_unique": int(uniq.size)}
        if uniq.size <= 25:
            d["unique_values"] = [float(v) for v in uniq]
        out.append(d)
    return out


def spec_stats(s: np.ndarray) -> dict:
    return {"min": float(s.min()), "max": float(s.max()), "mean": float(s.mean()),
            "std": float(s.std()), "frac_below_0": float((s < 0).mean()),
            "frac_above_1": float((s > 1).mean()),
            "per_sample_mean_range": [float(s.mean(1).min()), float(s.mean(1).max())],
            "mean_squared_value": float((s ** 2).mean()),
            "variance_around_mean_spectrum": float(((s - s.mean(0)) ** 2).mean())}


def make_figure(pool_g, pool_s, test_s, target, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(15, 9), constrained_layout=True)
    gs = fig.add_gridspec(3, 7)
    ax = fig.add_subplot(gs[0, :4])
    x = np.arange(C.SPEC_DIM)
    for i in range(6):
        ax.plot(x, pool_s[i], lw=1, label=f"pool sample {i}")
    ax.set_title("Example ADM absorptivity spectra (first 6 shuffled pool samples)")
    ax.set_xlabel("spectral point index (0-2000)")
    ax.set_ylabel("absorptivity")
    ax.legend(fontsize=7, ncol=3)
    ax2 = fig.add_subplot(gs[0, 4:])
    m, sd = pool_s.mean(0), pool_s.std(0)
    ax2.plot(x, m, color="k", lw=1.5, label="pool mean")
    ax2.fill_between(x, m - sd, m + sd, color="0.7", label="+/- 1 std")
    ax2.plot(x, test_s.mean(0), "--", color="tab:red", lw=1, label="test mean")
    ax2.plot(x, target, color="tab:blue", lw=1, label="Phase 3 target")
    ax2.set_title("Spectral statistics")
    ax2.set_xlabel("spectral point index")
    ax2.legend(fontsize=7)
    for j, name in enumerate(C.GEOM_NAMES):
        a = fig.add_subplot(gs[1 + j // 7, j % 7])
        a.hist(pool_g[:, j], bins=40, color="tab:blue", alpha=0.8)
        a.set_title(name, fontsize=9)
        a.tick_params(labelsize=7)
    fig.suptitle("ADM dataset overview (training pool histograms of the 14 geometry parameters)")
    fig.savefig(out, dpi=130)
    plt.close(fig)


def write_markdown(summary: dict, out: Path) -> None:
    s = summary
    v = s["verification"]
    lines = [
        "# Phase 1 summary: ADM data preparation",
        "",
        f"Generated {s['generated_at']} by `phase1_data/prepare_data.py` in {s['runtime_s']:.1f} s.",
        "",
        "## Verification",
        "",
        "| Check | Result |",
        "|---|---|",
        f"| Archive bytes (expected {EXPECTED_ZIP_BYTES:,}) | {s['archive']['bytes']:,} |",
        f"| Archive SHA-256 | `{s['archive']['sha256']}` |",
        f"| CRC check of all members | {s['archive']['crc_ok']} |",
        f"| Geometry dimension (independent variables) | **{v['geometry_dim']}** (expected 14): {v['geometry_dim_ok']} |",
        f"| Spectrum dimension (dependent variables) | **{v['spectrum_dim']}** (expected 2001): {v['spectrum_dim_ok']} |",
        f"| Row counts g/s match (pool, test) | {v['pool_rows_match']}, {v['test_rows_match']} |",
        f"| All values finite | {v['all_finite']} |",
        f"| float64 -> float32 max abs error (g, s) | {v['float32_max_abs_err_g']:.2e}, {v['float32_max_abs_err_s']:.2e} |",
        "",
        "## Extracted files",
        "",
        "| Member | Bytes |",
        "|---|---|",
    ]
    for m in s["archive"]["members"]:
        if not m["is_dir"]:
            lines.append(f"| `{m['member']}` | {m['bytes']:,} |")
    p = s["partition"]
    lines += [
        "",
        "## Partition",
        "",
        "| Subset | Samples | Geometry shape | Spectrum shape | File |",
        "|---|---|---|---|---|",
        f"| Training pool (shuffled, seed {C.POOL_SEED}) | {p['pool_rows']:,} | ({p['pool_rows']}, 14) | ({p['pool_rows']}, 2001) | `data/processed/pool_{{g,s}}.npy` |",
        f"| Independent test set | {p['test_rows']:,} | ({p['test_rows']}, 14) | ({p['test_rows']}, 2001) | `data/processed/test_{{g,s}}.npy` |",
        f"| Total | {p['pool_rows'] + p['test_rows']:,} | | | |",
        "",
        f"Forward_Train protocol: D_k = first k pool rows; train:val = 10:1 (position i is validation iff "
        f"i % 11 == 10); k0 = {C.K0}; data budget = {p['data_budget']:,} (max usable k = {p['max_k']:,}).",
        "",
        "| k | train | val |",
        "|---|---|---|",
    ]
    for row in p["example_splits"]:
        lines.append(f"| {row['k']:,} | {row['n_train']:,} | {row['n_val']:,} |")
    t = s["inverse_design_targets"]
    lines += [
        "",
        "Inverse-design targets were pre-registered before any model was trained (seed 0): 100 test "
        f"indices in `data/processed/inverse_test_idx.npy`; the Phase 3 target is test row {t['target_test_index']} "
        f"(`phase3_inverse/target_spectrum.csv`).",
        "",
        "## Geometry parameters (training pool)",
        "",
        "Column order follows the manuscript's listing (the CSV has no header): h, p, rx1-rx4, ry1-ry4, "
        "theta1-theta4. Angles are in radians.",
        "",
        "| # | Name | Min | Max | Mean | Std | Unique values |",
        "|---|---|---|---|---|---|---|",
    ]
    for gq in s["geometry_stats_pool"]:
        lines.append(f"| {gq['index']} | {gq['name']} | {gq['min']:.4f} | {gq['max']:.4f} | {gq['mean']:.4f} | "
                     f"{gq['std']:.4f} | {gq['n_unique']} |")
    sp = s["spectrum_stats_pool"]
    d = s["duplicates"]
    lines += [
        "",
        "## Spectra (training pool)",
        "",
        f"Absorptivity values span [{sp['min']:.4g}, {sp['max']:.4g}], mean {sp['mean']:.4f}, std {sp['std']:.4f}. "
        f"Predicting the pool-mean spectrum for every sample gives MSE = {sp['variance_around_mean_spectrum']:.4g} "
        "(a trivial baseline for the 2e-3 target).",
        "",
        "## Duplicates",
        "",
        f"* Pool: {d['pool_unique_geometries']:,} unique geometries, {d['pool_extra_duplicate_rows']:,} extra duplicate rows.",
        f"* Test: {d['test_unique_geometries']:,} unique geometries, {d['test_extra_duplicate_rows']:,} extra duplicate rows.",
        f"* Geometries shared by pool and test: {d['geometries_shared_by_pool_and_test']:,} "
        f"({d['test_rows_whose_geometry_appears_in_pool']:,} test rows).",
        "",
        "![overview](data_overview.png)",
        "",
    ]
    out.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zip", default=str(C.RAW / "ADM.zip"))
    ap.add_argument("--seed", type=int, default=C.POOL_SEED)
    args = ap.parse_args()
    t_start = time.time()
    zip_path = Path(args.zip)

    log("1/6 verifying archive")
    nbytes = zip_path.stat().st_size
    sha = C.sha256_file(zip_path)
    if nbytes != EXPECTED_ZIP_BYTES:
        raise RuntimeError(f"archive size {nbytes} != expected {EXPECTED_ZIP_BYTES}")

    log("2/6 extracting (CRC-checked, zip-slip guarded)")
    members = safe_extract(zip_path, C.EXTRACTED)

    log("3/6 loading CSV files")
    arrays = {key: load_csv(C.EXTRACTED / m, C.GEOM_DIM if key.endswith("_g") else C.SPEC_DIM)
              for key, m in MEMBERS.items()}
    dg, ds, tg, ts = arrays["data_g"], arrays["data_s"], arrays["test_g"], arrays["test_s"]

    log("4/6 verifying")
    verification = {
        "geometry_dim": int(dg.shape[1]), "geometry_dim_ok": dg.shape[1] == 14 and tg.shape[1] == 14,
        "spectrum_dim": int(ds.shape[1]), "spectrum_dim_ok": ds.shape[1] == 2001 and ts.shape[1] == 2001,
        "pool_rows_match": dg.shape[0] == ds.shape[0], "test_rows_match": tg.shape[0] == ts.shape[0],
        "all_finite": bool(np.isfinite(dg).all() and np.isfinite(ds).all()
                           and np.isfinite(tg).all() and np.isfinite(ts).all()),
        "float32_max_abs_err_g": float(np.abs(dg.astype(np.float32).astype(np.float64) - dg).max()),
        "float32_max_abs_err_s": float(np.abs(ds.astype(np.float32).astype(np.float64) - ds).max()),
    }
    ok = all(verification[k] for k in ["geometry_dim_ok", "spectrum_dim_ok", "pool_rows_match",
                                       "test_rows_match", "all_finite"])
    if not ok:
        raise RuntimeError(f"verification failed: {verification}")

    log("5/6 partitioning")
    C.PROCESSED.mkdir(parents=True, exist_ok=True)
    perm = np.random.default_rng(args.seed).permutation(dg.shape[0])
    pool_g = dg[perm].astype(np.float32)
    pool_s = ds[perm].astype(np.float32)
    np.save(C.PROCESSED / "pool_g.npy", pool_g)
    np.save(C.PROCESSED / "pool_s.npy", pool_s)
    np.save(C.PROCESSED / "pool_perm.npy", perm.astype(np.int64))
    np.save(C.PROCESSED / "test_g.npy", tg.astype(np.float32))
    np.save(C.PROCESSED / "test_s.npy", ts.astype(np.float32))

    inv_idx = np.random.default_rng(0).choice(tg.shape[0], size=100, replace=False).astype(np.int64)
    np.save(C.PROCESSED / "inverse_test_idx.npy", inv_idx)
    t_idx = int(inv_idx[0])
    C.PHASE3.mkdir(parents=True, exist_ok=True)
    target = ts[t_idx].astype(np.float32)
    np.savetxt(C.PHASE3 / "target_spectrum.csv", target[None, :], delimiter=",", fmt="%.9e")
    C.write_json_atomic(C.PHASE3 / "target_info.json", {
        "description": "Pre-registered inverse-design target (Phase 1, seed 0): a held-out test spectrum "
                       "simulated with CST. The geometry below is for reference only and is never given "
                       "to the Neural-Adjoint optimizer.",
        "source": f"{MEMBERS['test_s']} row {t_idx} (0-based, no header)",
        "test_index": t_idx,
        "ground_truth_geometry": {n: float(v) for n, v in zip(C.GEOM_NAMES, tg[t_idx])},
    })

    max_k = min(C.DATA_BUDGET, pool_g.shape[0])
    example_splits = []
    for k in [C.K0, 1000, 5000, 10000, 20000, 42250, max_k]:
        if k <= pool_g.shape[0]:
            tr, va = C.split_positions(k)
            example_splits.append({"k": k, "n_train": int(tr.size), "n_val": int(va.size)})

    log("6/6 summarising")
    summary = {
        "generated_at": C.now_iso(),
        "archive": {"path": C.rel(zip_path), "bytes": nbytes, "sha256": sha, "crc_ok": True,
                    "source_url": "https://research.repository.duke.edu/record/176/files/ADM.zip",
                    "members": members},
        "verification": verification,
        "raw_shapes": {k: list(v.shape) for k, v in arrays.items()},
        "partition": {"pool_rows": int(pool_g.shape[0]), "test_rows": int(tg.shape[0]),
                      "pool_shuffle_seed": args.seed, "val_rule": "position i is validation iff i % 11 == 10",
                      "k0": C.K0, "data_budget": C.DATA_BUDGET, "max_k": int(max_k),
                      "example_splits": example_splits,
                      "files": {n: C.rel(C.PROCESSED / n) for n in
                                ["pool_g.npy", "pool_s.npy", "pool_perm.npy", "test_g.npy", "test_s.npy",
                                 "inverse_test_idx.npy"]}},
        "geometry_names": C.GEOM_NAMES,
        "geometry_stats_pool": geom_stats(dg),
        "geometry_stats_test": geom_stats(tg),
        "spectrum_stats_pool": spec_stats(ds),
        "spectrum_stats_test": spec_stats(ts),
        "duplicates": dup_stats(dg, tg),
        "inverse_design_targets": {"seed": 0, "n": 100, "target_test_index": t_idx,
                                   "target_file": C.rel(C.PHASE3 / "target_spectrum.csv")},
    }
    make_figure(pool_g, pool_s, ts, target, C.PHASE1 / "data_overview.png")
    summary["runtime_s"] = time.time() - t_start
    C.write_json_atomic(C.PHASE1 / "data_summary.json", summary)
    write_markdown(summary, C.PHASE1 / "summary.md")
    log(f"done in {summary['runtime_s']:.1f}s")
    print(json.dumps({"verification": verification, "partition": {k: summary["partition"][k] for k in
                                                                    ["pool_rows", "test_rows", "max_k"]},
                      "duplicates": summary["duplicates"], "target_test_index": t_idx}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
