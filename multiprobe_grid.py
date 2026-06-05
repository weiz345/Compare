from __future__ import annotations

from collections import deque
from typing import Callable, Iterable


class CellListND:
    """N-D grid mapping each cell to a list of point indices."""

    def __init__(self, dims: int, bounds: list[tuple[float, float]], splits: tuple[int, ...]) -> None:
        self.dims = dims
        self.splits = splits
        self.mins = [bound[0] for bound in bounds]
        self.maxs = [bound[1] for bound in bounds]
        self.cell_size = [
            size if (size := (self.maxs[dim] - self.mins[dim]) / splits[dim]) > 0 else 1.0
            for dim in range(dims)
        ]
        self.cells: dict[tuple[int, ...], list[int]] = {}

    def _cell_index(self, coord: tuple[float, ...]) -> tuple[int, ...]:
        idx: list[int] = []
        for dim in range(self.dims):
            ratio = (coord[dim] - self.mins[dim]) / self.cell_size[dim]
            cell_index = max(0, min(self.splits[dim] - 1, int(ratio)))
            idx.append(cell_index)
        return tuple(idx)

    def insert(self, coord: tuple[float, ...], index_id: int) -> None:
        cell_id = self._cell_index(coord)
        self.cells.setdefault(cell_id, []).append(index_id)

    def _lin(self, cell_id: tuple[int, ...]) -> int:
        result = 0
        multiplier = 1
        for dim in range(self.dims - 1, -1, -1):
            result += cell_id[dim] * multiplier
            multiplier *= self.splits[dim]
        return result

    def _unlin(self, value: int) -> tuple[int, ...]:
        result: list[int] = []
        for dim in range(self.dims):
            multiplier = 1
            for dim2 in range(dim + 1, self.dims):
                multiplier *= self.splits[dim2]
            result.append((value // multiplier) % self.splits[dim])
        return tuple(result)

    def neighbors(self, value: int) -> Iterable[int]:
        cell_id = self._unlin(value)

        for dim in range(self.dims):
            plus_neighbor = list(cell_id)
            plus_neighbor[dim] += 1
            if 0 <= plus_neighbor[dim] < self.splits[dim]:
                yield self._lin(tuple(plus_neighbor))

            minus_neighbor = list(cell_id)
            minus_neighbor[dim] -= 1
            if 0 <= minus_neighbor[dim] < self.splits[dim]:
                yield self._lin(tuple(minus_neighbor))


def bfs_nearest_nonempty(
    total_nodes: int,
    neighbors_fn: Callable[[int], Iterable[int]],
    occupied_nodes: list[int],
) -> tuple[list[int], list[int]]:
    inf = 10**9
    dist = [inf] * total_nodes
    nearest = [-1] * total_nodes
    queue: deque[int] = deque()

    for node in occupied_nodes:
        dist[node] = 0
        nearest[node] = node
        queue.append(node)

    while queue:
        node = queue.popleft()
        for neighbor in neighbors_fn(node):
            if dist[neighbor] == inf:
                dist[neighbor] = dist[node] + 1
                nearest[neighbor] = nearest[node]
                queue.append(neighbor)

    return dist, nearest
