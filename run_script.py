#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

from nearest_neighbors import BruteForceNN
from script_loader import BuildOp, QueryOp, load_script


def run_script(path: Path) -> None:
    index = BruteForceNN()

    for operation in load_script(path):
        if isinstance(operation, BuildOp):
            index.build(operation.points)
            print("build ok")
            continue

        results = index.query(operation.point, operation.k)
        print(f"query k {operation.k}")
        for neighbor in results:
            print(neighbor.index)


def main() -> None:
    default_path = Path(__file__).resolve().parent / "script.cases"
    path = Path(sys.argv[1]) if len(sys.argv) == 2 else default_path
    run_script(path)


if __name__ == "__main__":
    main()
