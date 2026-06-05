#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from nearest_neighbors import NearestNeighbors, create_index
from script_loader import BuildOp, QueryOp, load_script

DEFAULT_ALGO = "brute_force"


def run_script(path: Path, algo: str = DEFAULT_ALGO) -> None:
    index: NearestNeighbors = create_index(algo)

    for operation in load_script(path):
        if isinstance(operation, BuildOp):
            index.build(operation.points)
            print("build ok")
            continue

        results = index.query(operation.point, operation.k)
        print(f"query k {operation.k}")
        for neighbor in results:
            print(neighbor.index)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ANN test script")
    parser.add_argument(
        "script",
        nargs="?",
        default="script.cases",
        help="path to script.cases (default: script.cases)",
    )
    parser.add_argument(
        "--algo",
        default=DEFAULT_ALGO,
        help="ANN algorithm to run (default: brute_force)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent
    script_path = Path(args.script)
    if not script_path.is_absolute():
        script_path = root / script_path

    run_script(script_path, args.algo)


if __name__ == "__main__":
    main()
