from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from math import sqrt
from typing import Sequence


@dataclass(frozen=True)
class Neighbor:
    index: int
    distance: float


class NearestNeighbors(ABC):
    @abstractmethod
    def build(self, points: Sequence[Sequence[float]]) -> None:
        raise NotImplementedError

    @abstractmethod
    def query(self, point: Sequence[float], k: int) -> list[Neighbor]:
        raise NotImplementedError


def l2_distance(a: Sequence[float], b: Sequence[float]) -> float:
    dims = min(len(a), len(b))
    sum_sq = 0.0

    for i in range(dims):
        diff = a[i] - b[i]
        sum_sq += diff * diff

    return sqrt(sum_sq)
