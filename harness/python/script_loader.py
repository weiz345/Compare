from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BuildOp:
    points: list[list[float]]


@dataclass(frozen=True)
class QueryOp:
    point: list[float]
    k: int


Operation = BuildOp | QueryOp


def _parse_floats(text: str) -> list[float]:
    return [float(value) for value in text.split()]


def load_script(path: Path | str) -> list[Operation]:
    source = Path(path).read_text(encoding="utf-8")
    operations: list[Operation] = []
    pending_points: list[list[float]] = []
    mode: str | None = None

    def flush_build() -> None:
        nonlocal pending_points, mode
        operations.append(BuildOp(points=pending_points))
        pending_points = []
        mode = None

    for raw_line in source.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        if line == "build":
            if mode == "build":
                flush_build()
            mode = "build"
            continue

        if line.startswith("query "):
            if mode == "build":
                flush_build()

            parts = line.split()
            if "k" not in parts:
                raise ValueError(f"query line missing k: {line}")

            split_at = parts.index("k")
            point = [float(value) for value in parts[1:split_at]]
            k = int(parts[split_at + 1])
            operations.append(QueryOp(point=point, k=k))
            mode = None
            continue

        if mode == "build":
            pending_points.append(_parse_floats(line))
            continue

        raise ValueError(f"unexpected line: {line}")

    if mode == "build":
        flush_build()

    return operations
