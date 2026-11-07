"""Shared constants, paths and dataset access for the agentic metasurface framework.

The framework mirrors the tools of Lu, Malof & Padilla, "An Agentic Framework for Autonomous
Metamaterial Modeling and Inverse Design", ACS Photonics 2025, 12, 6071-6080:

    File_Check      -> framework/file_check.py
    Forward_Train   -> Prepare Dataset (framework/prepare_dataset.py)
                       Code Generation / Evaluate execution (framework/harness.py)
    Code_Modify     -> performed by the Modifier agent (Phase 3), output consumed by
    Neural_Adjoint  -> framework/neural_adjoint.py

Dataset protocol (manuscript Sec. 2.3 and 3.1)
---------------------------------------------
* The ADM training pool (data_g.csv / data_s.csv) is shuffled ONCE with a fixed seed in Phase 1
  and stored as data/processed/pool_{g,s}.npy.
* "Prepare Dataset(k)" returns D_k = the first k rows of that order, so data sets are nested
  (D_k is a subset of D_k' for k < k'): growing from k_prev to k' only adds the (k' - k_prev) new
  geometry-spectrum pairs, exactly as the manuscript's tool does.
* Each D_k is split into training / validation subsets with a 10:1 ratio: position i of the
  shuffled pool belongs to the validation subset iff i % 11 == 10.  The split of the newly added
  samples is therefore also 10:1, and samples never move between train and validation.
* The independent test set (testset/test_{g,s}.csv) is never used for model selection.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = DATA / "raw"
EXTRACTED = DATA / "extracted"
PROCESSED = DATA / "processed"
PHASE1 = ROOT / "phase1_data"
PHASE2 = ROOT / "phase2_forward"
PHASE3 = ROOT / "phase3_inverse"
SOLUTIONS = PHASE2 / "solutions"
MEMORY = PHASE2 / "memory"
RUNS = PHASE2 / "runs"
LEDGER = RUNS / "ledger.jsonl"
DATA_STATE = MEMORY / "data_state.json"

GEOM_DIM = 14
SPEC_DIM = 2001
# Column order of data_g.csv (the CSV has no header). Order follows the manuscript's listing:
# height h, periodicity p, ellipse semi-axes rx1-rx4, ry1-ry4, rotation angles theta1-theta4 (rad).
GEOM_NAMES = ["h", "p", "rx1", "rx2", "rx3", "rx4", "ry1", "ry2", "ry3", "ry4",
              "theta1", "theta2", "theta3", "theta4"]

VAL_PERIOD = 11          # train:val = 10:1
K0 = 500                 # initial data set size (Figure 5 starts at ~500 samples)
DATA_BUDGET = 50_000     # manuscript: "data budget = 50,000 samples"
MAX_ROUNDS = 50          # manuscript: "max rounds = 50"
TARGET_MSE = 2e-3        # strict upper bound on validation MSE
STRETCH_MSE = 1.3e-3     # aspiration (best agent result in the manuscript)
POOL_SEED = 2025         # seed of the one-time pool shuffle (Phase 1)


def now_iso() -> str:
    return _dt.datetime.now().astimezone().isoformat(timespec="seconds")


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def rel(path: Path | str) -> str:
    """Repository-relative POSIX path (for logs and reports)."""
    p = Path(path).resolve()
    try:
        return p.relative_to(ROOT).as_posix()
    except ValueError:
        return p.as_posix()


# ----------------------------------------------------------------------------- JSON helpers
def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return o.as_posix()
    return str(o)


def write_json_atomic(path: Path, obj, indent: int | None = 1) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(obj, indent=indent, default=_json_default), encoding="utf-8")
    for _ in range(200):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:  # Windows: reader holds the file open for a moment
            time.sleep(0.05)
    os.replace(tmp, path)


def read_json(path: Path, default=None, tries: int = 100):
    path = Path(path)
    for _ in range(tries):
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return default
        except (PermissionError, json.JSONDecodeError):
            time.sleep(0.05)
    return default


# ----------------------------------------------------------------------------- file locks
class FileLock:
    """Inter-process lock on a lock file (msvcrt on Windows, fcntl elsewhere).

    The OS releases the lock automatically if the holding process dies.
    """

    def __init__(self, path: Path, timeout: float | None = 60.0, poll: float = 0.05):
        self.path = Path(path)
        self.timeout = timeout
        self.poll = poll
        self.fd = None
        self.index = None

    def try_acquire(self) -> bool:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.path, os.O_RDWR | os.O_CREAT)
        try:
            if os.name == "nt":
                import msvcrt
                os.lseek(fd, 0, 0)
                msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            os.close(fd)
            return False
        self.fd = fd
        return True

    def acquire(self) -> "FileLock":
        t0 = time.time()
        while not self.try_acquire():
            if self.timeout is not None and time.time() - t0 > self.timeout:
                raise TimeoutError(f"could not lock {self.path}")
            time.sleep(self.poll)
        return self

    def release(self) -> None:
        if self.fd is None:
            return
        try:
            if os.name == "nt":
                import msvcrt
                os.lseek(self.fd, 0, 0)
                try:
                    msvcrt.locking(self.fd, msvcrt.LK_UNLCK, 1)
                except OSError:
                    pass
            else:
                import fcntl
                fcntl.flock(self.fd, fcntl.LOCK_UN)
        finally:
            os.close(self.fd)
            self.fd = None

    def __enter__(self):
        return self.acquire()

    def __exit__(self, *exc):
        self.release()


def append_jsonl(path: Path, record: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(path.with_name(path.name + ".lock"), timeout=120):
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=_json_default) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


# ----------------------------------------------------------------------------- data access
def pool_size() -> int:
    return int(np.load(PROCESSED / "pool_g.npy", mmap_mode="r").shape[0])


def max_dataset_size() -> int:
    return min(DATA_BUDGET, pool_size())


def split_positions(k: int) -> tuple[np.ndarray, np.ndarray]:
    """Positions (into the shuffled pool) of the train / validation subsets of D_k."""
    pos = np.arange(int(k))
    is_val = (pos % VAL_PERIOD) == (VAL_PERIOD - 1)
    return pos[~is_val], pos[is_val]


def granted_size() -> int:
    """Largest data set size acquired so far through the Prepare Dataset tool (0 if none)."""
    st = read_json(DATA_STATE, default={}) or {}
    return int(st.get("k_current", 0))


def load_dataset(k: int, enforce_grant: bool = True):
    """Return (Xtr, Ytr, Xva, Yva) as float32 arrays for data set D_k."""
    k = int(k)
    kmax = max_dataset_size()
    if not 1 <= k <= kmax:
        raise ValueError(f"k={k} outside [1, {kmax}] (data budget / pool size)")
    if enforce_grant:
        g = granted_size()
        if k > g:
            raise PermissionError(
                f"k={k} exceeds the data acquired so far (k_current={g}). Data can only be acquired "
                f"through the Prepare Dataset tool (framework/prepare_dataset.py) after a Controller decision.")
    G = np.load(PROCESSED / "pool_g.npy", mmap_mode="r")
    S = np.load(PROCESSED / "pool_s.npy", mmap_mode="r")
    tr, va = split_positions(k)
    Xtr = np.ascontiguousarray(G[tr], dtype=np.float32)
    Ytr = np.ascontiguousarray(S[tr], dtype=np.float32)
    Xva = np.ascontiguousarray(G[va], dtype=np.float32)
    Yva = np.ascontiguousarray(S[va], dtype=np.float32)
    return Xtr, Ytr, Xva, Yva


def load_test():
    """Independent test set (never used for model selection)."""
    Xte = np.load(PROCESSED / "test_g.npy").astype(np.float32)
    Yte = np.load(PROCESSED / "test_s.npy").astype(np.float32)
    return Xte, Yte
