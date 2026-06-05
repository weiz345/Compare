#!/usr/bin/env python3
from __future__ import annotations

import difflib
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SCRIPT_FILE = ROOT / "script.cases"
CPP_RUNNER = ROOT / "build" / "run_script"


def capture_python_output() -> str:
    return subprocess.check_output(
        [sys.executable, str(ROOT / "run_script.py"), str(SCRIPT_FILE)],
        text=True,
        cwd=ROOT,
    )


def capture_cpp_output() -> str:
    if not CPP_RUNNER.exists():
        raise FileNotFoundError(
            f"C++ runner not found at {CPP_RUNNER}. "
            "Build it with: cmake -B build && cmake --build build"
        )

    return subprocess.check_output(
        [str(CPP_RUNNER), str(SCRIPT_FILE)],
        text=True,
        cwd=ROOT,
    )


def main() -> int:
    try:
        python_output = capture_python_output()
        cpp_output = capture_cpp_output()
    except (FileNotFoundError, subprocess.CalledProcessError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1

    if python_output == cpp_output:
        query_count = python_output.count("query k ")
        print(f"PASS: Python and C++ agree on all {query_count} query result(s).")
        return 0

    print("FAIL: Python and C++ outputs differ\n", file=sys.stderr)
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
