from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Sequence


@dataclass(frozen=True)
class Neighbor:
    index: int
    distance: float


class BruteForceNN:
    def __init__(self) -> None:
        self._points: list[list[float]] = []

    def build(self, points: Sequence[Sequence[float]]) -> None:
        self._points = [list(point) for point in points]

    def query(self, point: Sequence[float], k: int) -> list[Neighbor]:
        if not self._points or k <= 0:
            return []

        candidates = [
            Neighbor(i, _l2_distance(point, stored))
            for i, stored in enumerate(self._points)
        ]
        candidates.sort(key=lambda neighbor: neighbor.distance)
        return candidates[: min(k, len(candidates))]


def _l2_distance(a: Sequence[float], b: Sequence[float]) -> float:
    dims = min(len(a), len(b))
    sum_sq = 0.0

    for i in range(dims):
        diff = a[i] - b[i]
        sum_sq += diff * diff

    return sqrt(sum_sq)
