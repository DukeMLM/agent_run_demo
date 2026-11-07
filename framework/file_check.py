#!/usr/bin/env python
"""File_Check tool (manuscript Sec. 2.2, Input Verifier).

    python -I framework/file_check.py PATH [PATH ...]

Accepts one or more file paths, checks their existence and returns whether each exists (JSON).
Exit code 0 if all exist, 2 otherwise.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    res = {}
    for p in argv:
        path = Path(p)
        res[p] = {"exists": path.exists(),
                  "type": "dir" if path.is_dir() else ("file" if path.is_file() else None),
                  "bytes": path.stat().st_size if path.is_file() else None}
    all_ok = all(v["exists"] for v in res.values())
    print(json.dumps({"all_exist": all_ok, "files": res}, indent=1))
    return 0 if all_ok else 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
