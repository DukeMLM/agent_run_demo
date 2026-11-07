# Phase 2 plan: forward surrogate modeling (Forward_Train)

Planner agent (manuscript Sec. 2.1), 2026-11-06. Sources: `phase1_data/summary.md`,
`phase1_data/data_summary.json`, `framework/TOOLS.md`, `framework/harness.py`.
AIDE works from `phase2_forward/task_description.md`.

## Objective

* Learn f: 14 ADM geometry parameters -> 2001-point absorptivity spectrum.
* Metric: validation MSE on raw absorptivity of the best checkpoint, as printed by the harness
  (`RESULT_JSON` lines, `phase2_forward/runs/ledger.jsonl`). No other number counts.
* Strict upper bound **2.0e-3**; drive toward the stretch goal **1.3e-3**. References: mean-spectrum baseline
  1.52e-2; label-noise floor about 1.3e-5 (Phase 1 duplicate pairs), so errors are model / data limited.
* Budget: k0 = 500; data budget 50,000 (max k); round 0 plus at most 50 Controller rounds;
  train:val = 10:1 (pool position i is validation iff i % 11 == 10); nested D_k.
* The independent test set is not used in this phase (only exception: Reporting step 4).
* Deliverables: final architecture details, phase runtime, plot of validation MSE vs dataset size.

## Roles

| Agent / tool | Responsibility |
|---|---|
| Planner | This plan and `phase2_forward/task_description.md`. |
| Input Verifier -> File_Check | Before round 0: `file_check.py` on the inputs below. Exit 2: stop and report the missing paths. |
| Forward Modeler -> Forward_Train | Runs the loop; sole writer of `memory/history.json`; calls Controller, Prepare Dataset, AIDE, Evaluate. |
| Controller (in Forward_Train) | After each round reads history + ledger, returns `{action: test or generate, k_next, max_train_seconds, reason}`. Only it decides data growth. |
| Prepare Dataset | `prepare_dataset.py --k K --note "round N: <reason>"`, only after a Controller decision. |
| AIDE (code generation) | Tree search (draft / improve / debug) over solution files per the task description; returns the best node (file, run_id, val_mse). |
| Evaluate | `harness.py check / submit / wait`: the only source of numbers. |
| Reporter | Plots, runtime and final-architecture write-up in `phase2_forward/summary.md`. |

File_Check inputs: `data/processed/pool_g.npy`, `data/processed/pool_s.npy`, `phase1_data/data_summary.json`,
`phase2_forward/task_description.md`, `phase2_forward/memory/plan.md`, `phase2_forward/memory/phase2_start.json`,
`phase2_forward/solutions`, `framework/harness.py`, `framework/prepare_dataset.py`, `framework/plot_history.py`.

## Forward_Train loop

1. Round 0 (Initialization): Prepare Dataset(500). AIDE drafts 3 diverse candidates in parallel on D_500, then
   up to 3 improve / debug nodes. Best node = incumbent. History event 0: `generate`, reason `Initialization`.
2. Rounds 1-50, one Controller action each:
   * **test**: Prepare Dataset(k_next), then submit the incumbent file unchanged on D_k_next.
   * **generate**: Prepare Dataset(k_next) if k_next > k, then an AIDE search on D_k_next with the incumbent
     (re-scored on D_k_next alongside wave 1 if needed) as root node, so a generate round never ends worse
     than the incumbent. Best node = new incumbent.
3. Exactly one history event per round (the round's incumbent result). All other candidates stay in the
   ledger and in `memory/aide_journal.jsonl`.
4. Check the termination rules after every round.

## Balancing data growth and model generation (Controller policy)

Default ladder (sizes only grow; the Controller may skip a rung with a stated reason):
500, 1,000, 2,000, 4,000, 8,000, 12,000, 16,000, 20,000, 25,000, 30,000, 35,000, 42,250, 50,000.

Training budget S = `--max-train-seconds` per k (the same for every history candidate at that k):

| k | <= 2,000 | 4,000-8,000 | 12,000-20,000 | 25,000-35,000 | >= 42,250 |
|---|---|---|---|---|---|
| S (s) | 180 | 300 | 600 | 900 | 2,400 |

Rules (m = validation MSE):
1. After a **test**: slope a = ln(m_prev / m_new) / ln(k_new / k_prev) for the same code, clipped to [0, 0.6];
   projection m50 = m_new * (50,000 / k_new)^(-a). **Generate** if m got worse, a < 0.15 (saturating) or
   m50 > 2.0e-3 (this code cannot meet the bound even with all data). Otherwise **test** the next rung.
2. After a **generate**: if the incumbent improved by >= 5 %, **test** next. Otherwise generate once more,
   then grow anyway (at most 2 consecutive generate rounds below k = 50,000).
3. Small-k noise: k <= 2,000 means <= 182 validation samples. Treat differences < 15 % as noise there and
   limit generate effort to round 0 plus at most one round.
4. If no generate round has happened by k = 20,000, issue one there, so the architecture is re-chosen with
   >= 1,818 validation samples before the expensive large-k rungs.
5. At k = 50,000 only generate rounds remain: architecture and hyperparameter search until termination.
6. Generate rounds below k = 8,000: one wave of 3 parallel candidates (+ debug). From k = 8,000: two waves,
   wave 1 = diverse families or bold changes, wave 2 = single-change refinements of the best node. AIDE may
   screen at a smaller k or at S/2 (ledger only, e.g. 1,200 s at k >= 42,250) and promote the best to the
   round's k and S; only results at the round's k and S enter history.

Rough cost (estimate): about 3 h of test rounds plus 1-1.5 h per generate round at k >= 42,250; expect
6-10 h wall clock in total. The Forward Modeler logs the elapsed time in every history event.

## Termination

Stop at the first of:
1. Incumbent val MSE <= 1.3e-3: stretch goal met.
2. k = 50,000, incumbent <= 2.0e-3, and 3 consecutive generate rounds improved the best val MSE by < 2 % in
   total: converged.
3. Round 50. Success only if the incumbent is <= 2.0e-3; otherwise report the failure with the best result.

The 2.0e-3 bound is strict: never stop early above it while rounds remain. Final model = the incumbent; if a
smaller model is within 1 % of it at the same k and S, prefer the smaller one (Phase 3 Neural-Adjoint reuses it).

## Compute and bookkeeping

* GPU: one RTX 4080 SUPER (16 GB), at most 3 concurrent harness runs (extra runs queue). Submit with
  `--wait 0`, then `harness.py wait RUN_ID` (Bash timeout 600000 ms) until exit 0 (DONE) or 1 (FAILED).
* Compare candidates only at the same k and S, launched in the same wave (the time-aware schedule depends
  on GPU contention). No seed shopping.
* Every setting lives in the solution file (CONFIG / adapt_config); no `--set` overrides, so a test round
  reproduces the candidate exactly.
* One new file per candidate, `phase2_forward/solutions/rNN_<idea>.py`; never edit a submitted file.
  Tag `rNN_<idea>_k<k>`.
* Failed runs: AIDE debugs them (max 2 attempts per node). Failures do not consume rounds.

## Memory (`phase2_forward/memory/`)

* `history.json`: list of round events `{round, k, action, val_mse, run_id, code, n_params,
  max_train_seconds, n_candidates, reason, elapsed_s}` (elapsed_s since `phase2_start.json`). First event:
  round 0, `generate`, reason `Initialization`. Input of `plot_history.py`.
* `aide_journal.jsonl`: one line per AIDE node `{round, node, parent, operator, idea, code, run_id, k, S,
  val_mse or error}`.
* `data_state.json` (owned by `prepare_dataset.py`); `phase2_start.json` (phase start time).

## Reporting (after termination)

1. `plot_history.py --history phase2_forward/memory/history.json --out phase2_forward/val_mse_vs_dataset_size.png
   --candidates-out phase2_forward/all_candidates.png`.
2. Runtime: wall clock = finish time minus the `phase2_start.json` start; GPU training time = sum of
   `train_time_s` over the ledger; numbers of rounds and trained candidates.
3. Final architecture: docstring, `model_repr`, n_params and config from the final run's `result.json`;
   `final_eval.py --run RUN_ID` recomputes the val MSE.
4. Test MSE (`final_eval.py --test`) only after the model is frozen, by the Reporter; never used for selection.
5. Write `phase2_forward/summary.md`: history table, both plots, architecture details, runtime.
