#!/usr/bin/env python3
from __future__ import annotations

import argparse
import difflib
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SCRIPT_FILE = ROOT / "script.cases"
CPP_RUNNER = ROOT / "build" / "run_script"
BUILD_DIR = ROOT / "build"
DEFAULT_ALGO = "brute_force"


@dataclass(frozen=True)
class Side:
    lang: str
    algo: str

    def label(self) -> str:
        return f"{self.lang}:{self.algo}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare ANN implementations by running script.cases on two sides",
    )
    parser.add_argument(
        "--left",
        default=None,
        help="left side as LANG:ALGO (e.g. python:brute_force, cpp:brute_force)",
    )
    parser.add_argument(
        "--right",
        default=None,
        help="right side as LANG:ALGO; if only --left or --right is set, compare to itself",
    )
    parser.add_argument(
        "--algo",
        default=None,
        help=f"shorthand for cross-language compare python:ALGO vs cpp:ALGO (default: {DEFAULT_ALGO})",
    )
    parser.add_argument(
        "--python-algo",
        default=None,
        help="shorthand for python:ALGO vs python:ALGO (same language)",
    )
    parser.add_argument(
        "--cpp-algo",
        default=None,
        help="shorthand for cpp:ALGO vs cpp:ALGO (same language)",
    )
    parser.add_argument(
        "--script",
        default=str(SCRIPT_FILE),
        help="path to script.cases",
    )
    parser.add_argument(
        "--no-build",
        action="store_true",
        help="skip rebuilding C++ before running (default: rebuild if any side uses cpp)",
    )
    return parser.parse_args()


def parse_side(spec: str) -> Side:
    if ":" not in spec:
        raise ValueError(f"side must be LANG:ALGO, got {spec!r}")

    lang, algo = spec.split(":", 1)
    if lang not in {"python", "cpp"}:
        raise ValueError(f"lang must be 'python' or 'cpp', got {lang!r}")
    if not algo:
        raise ValueError("algo cannot be empty")

    return Side(lang=lang, algo=algo)


def resolve_sides(args: argparse.Namespace) -> tuple[Side, Side]:
    if args.left or args.right:
        if args.left and args.right:
            return parse_side(args.left), parse_side(args.right)
        side = parse_side(args.left or args.right)
        return side, side

    if args.python_algo and args.cpp_algo:
        return Side("python", args.python_algo), Side("cpp", args.cpp_algo)
    if args.python_algo:
        return Side("python", args.python_algo), Side("python", args.python_algo)
    if args.cpp_algo:
        return Side("cpp", args.cpp_algo), Side("cpp", args.cpp_algo)

    shared = args.algo or DEFAULT_ALGO
    return Side("python", shared), Side("cpp", shared)


def build_cpp() -> None:
    if not (BUILD_DIR / "CMakeCache.txt").exists():
        subprocess.run(["cmake", "-B", "build"], cwd=ROOT, check=True)
    subprocess.run(["cmake", "--build", "build"], cwd=ROOT, check=True)


def capture_python_output(script_file: Path, algo: str) -> str:
    return subprocess.check_output(
        [sys.executable, str(ROOT / "run_script.py"), str(script_file), "--algo", algo],
        text=True,
        cwd=ROOT,
    )


def capture_cpp_output(script_file: Path, algo: str) -> str:
    return subprocess.check_output(
        [str(CPP_RUNNER), str(script_file), "--algo", algo],
        text=True,
        cwd=ROOT,
    )


def capture_side(script_file: Path, side: Side) -> str:
    if side.lang == "python":
        return capture_python_output(script_file, side.algo)
    return capture_cpp_output(script_file, side.algo)


def main() -> int:
    args = parse_args()
    script_file = Path(args.script)
    if not script_file.is_absolute():
        script_file = ROOT / script_file

    try:
        left, right = resolve_sides(args)
    except ValueError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1

    uses_cpp = left.lang == "cpp" or right.lang == "cpp"
    if uses_cpp and not args.no_build:
        try:
            build_cpp()
        except subprocess.CalledProcessError as error:
            print(f"ERROR: C++ build failed: {error}", file=sys.stderr)
            return 1

    if uses_cpp and not CPP_RUNNER.exists():
        print(
            f"ERROR: C++ runner not found at {CPP_RUNNER} after build.",
            file=sys.stderr,
        )
        return 1

    try:
        left_output = capture_side(script_file, left)
        right_output = capture_side(script_file, right)
    except subprocess.CalledProcessError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1

    if left_output == right_output:
        query_count = left_output.count("query k ")
        print(
            f"PASS: {left.label()} and {right.label()} agree on all "
            f"{query_count} query result(s)."
        )
        return 0

    print(
        f"FAIL: outputs differ ({left.label()} vs {right.label()})\n",
        file=sys.stderr,
    )
    diff = difflib.unified_diff(
        left_output.splitlines(keepends=True),
        right_output.splitlines(keepends=True),
        fromfile=left.label(),
        tofile=right.label(),
    )
    sys.stderr.writelines(diff)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
