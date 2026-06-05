#!/usr/bin/env python3
from __future__ import annotations

import argparse
import difflib
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SCRIPT_FILE = ROOT / "script.cases"
CPP_RUNNER = ROOT / "build" / "run_script"
DEFAULT_ALGO = "brute_force"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify Python and C++ ANN implementations")
    parser.add_argument(
        "--algo",
        default=DEFAULT_ALGO,
        help="ANN algorithm both sides must run (default: brute_force)",
    )
    parser.add_argument(
        "--script",
        default=str(SCRIPT_FILE),
        help="path to script.cases",
    )
    return parser.parse_args()


def capture_python_output(script_file: Path, algo: str) -> str:
    return subprocess.check_output(
        [sys.executable, str(ROOT / "run_script.py"), str(script_file), "--algo", algo],
        text=True,
        cwd=ROOT,
    )


def capture_cpp_output(script_file: Path, algo: str) -> str:
    if not CPP_RUNNER.exists():
        raise FileNotFoundError(
            f"C++ runner not found at {CPP_RUNNER}. "
            "Build it with: cmake -B build && cmake --build build"
        )

    return subprocess.check_output(
        [str(CPP_RUNNER), str(script_file), "--algo", algo],
        text=True,
        cwd=ROOT,
    )


def main() -> int:
    args = parse_args()
    script_file = Path(args.script)
    if not script_file.is_absolute():
        script_file = ROOT / script_file

    try:
        python_output = capture_python_output(script_file, args.algo)
        cpp_output = capture_cpp_output(script_file, args.algo)
    except (FileNotFoundError, subprocess.CalledProcessError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1

    if python_output == cpp_output:
        query_count = python_output.count("query k ")
        print(
            f"PASS: Python and C++ agree on all {query_count} query result(s) "
            f"using algo={args.algo!r}."
        )
        return 0

    print(
        f"FAIL: Python and C++ outputs differ for algo={args.algo!r}\n",
        file=sys.stderr,
    )
    diff = difflib.unified_diff(
        python_output.splitlines(keepends=True),
        cpp_output.splitlines(keepends=True),
        fromfile="python",
        tofile="c++",
    )
    sys.stderr.writelines(diff)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
