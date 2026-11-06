# Metasurface Neural-Adjoint Demo

Forward modeling and inverse design of an all-dielectric metasurface (ADM dataset),
following the methodology in *ACS Photonics* 12, 6071 (2025).

Work runs on the student's machine. Code and results are pushed here after each phase.
Questions and approvals go through this repo (see **Q&A protocol** below).

## Status

| Phase | Folder | Status |
|---|---|---|
| 1. Data preparation | `phase1_data/` | not started |
| 2. Forward surrogate model | `phase2_forward/` | waiting on Phase 1 approval |
| 3. Inverse design (Neural-Adjoint) | `phase3_inverse/` | waiting on Phase 2 approval |

Each phase folder holds its code, a `summary.md` with the outputs requested for that
phase, and any plots.

## Q&A protocol

This section is written for both the professor and any agent the professor uses.

### Asking a question

1. `git pull`
2. Copy `questions/TEMPLATE.md` to `questions/NNN_short-topic.md`
   - `NNN` = next unused number, zero-padded (`001`, `002`, ...)
3. Fill it in, then commit and push.

### Approving the next phase

Create a question file with `Type: proceed` and the phase number.
The next phase starts only after a `proceed` file for it is pushed.

### Answers

Each question gets a matching file in `answers/` with the same name
(`questions/003_loss.md` -> `answers/003_loss.md`).
Answers link to the code, commits, or plots they refer to.

### Rules

- Only add or edit files in `questions/`. Code and results are changed by the student.
- One question topic per file. Follow-ups go in a new file that references the earlier number.
- Questions that ask for new code or new runs are reviewed by the student before they are run.

## Data

The ADM dataset is not stored in this repo. It is downloaded from the Duke Research
Data Repository: https://research.repository.duke.edu/record/176
