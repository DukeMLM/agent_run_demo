#!/usr/bin/env python
"""Prepare Dataset tool of Forward_Train (manuscript Sec. 2.3, Algorithm 1 lines 2 and 7).

    python -I framework/prepare_dataset.py --k K        # grow the acquired data set to size K
    python -I framework/prepare_dataset.py --status     # show the current data state

The tool accepts a target size k' and acquires only the additional (k' - k_prev) geometry-spectrum
pairs needed to grow the existing data set to k'. Here the "simulator" is the pre-computed ADM pool
(manuscript Sec. 3.1: drawing a subset from the existing ADM data set is functionally identical to
running new simulations). New samples are split train:val = 10:1 (pool position i is validation iff
i % 11 == 10). Data sets are nested, sizes may only grow, and k' may not exceed the data budget.
The call is idempotent: requesting the current size again is a no-op.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--k", type=int)
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--note", default="")
    a = ap.parse_args()
    kmax = C.max_dataset_size()
    with C.FileLock(C.DATA_STATE.with_name("data_state.lock"), timeout=120):
        st = C.read_json(C.DATA_STATE, default=None) or {"k_current": 0, "acquisitions": []}
        if a.status or a.k is None:
            print(json.dumps({"k_current": st["k_current"], "data_budget": C.DATA_BUDGET, "max_k": kmax,
                              "n_acquisitions": len(st["acquisitions"])}, indent=1))
            return 0
        k_new, k_prev = int(a.k), int(st["k_current"])
        if k_new > kmax:
            print(json.dumps({"ok": False, "error": f"k={k_new} exceeds the data budget / pool (max {kmax})"}))
            return 2
        if k_new < k_prev:
            print(json.dumps({"ok": False, "error": f"data sets only grow: k={k_new} < k_current={k_prev}. "
                                                    f"Train on a smaller k directly if needed (k <= k_current)."}))
            return 2
        tr, va = C.split_positions(k_new)
        tr0, va0 = C.split_positions(k_prev)
        rec = {"k_prev": k_prev, "k_new": k_new, "added": k_new - k_prev,
               "added_train": int(tr.size - tr0.size), "added_val": int(va.size - va0.size),
               "n_train": int(tr.size), "n_val": int(va.size), "at": C.now_iso(), "note": a.note}
        if k_new > k_prev:
            st["k_current"] = k_new
            st["acquisitions"].append(rec)
            C.write_json_atomic(C.DATA_STATE, st)
            status = "acquired"
        else:
            status = "no-op (already at this size)"
    print(json.dumps({"ok": True, "status": status, **rec, "data_budget": C.DATA_BUDGET, "max_k": kmax}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
