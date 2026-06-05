from __future__ import annotations

import itertools
from collections import deque
from typing import Callable, Iterable, Sequence

import numpy as np
from sklearn.decomposition import PCA

from ann_interface import Neighbor, NearestNeighbors, l2_distance


class BruteForceNN(NearestNeighbors):
    def __init__(self) -> None:
        self._points: list[list[float]] = []

    def build(self, points: Sequence[Sequence[float]]) -> None:
        self._points = [list(point) for point in points]

    def query(self, point: Sequence[float], k: int) -> list[Neighbor]:
        if not self._points or k <= 0:
            return []

        candidates = [
            Neighbor(i, l2_distance(point, stored))
            for i, stored in enumerate(self._points)
        ]
        candidates.sort(key=lambda neighbor: neighbor.distance)
        return candidates[: min(k, len(candidates))]


class CellListND:
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
            idx.append(max(0, min(self.splits[dim] - 1, int(ratio))))
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


class MultiProbeNN(NearestNeighbors):
    def __init__(self, grid_dims: int = 3, grid_splits: int = 6, n_probe: int = 4) -> None:
        self._grid_dims = grid_dims
        self._grid_splits = grid_splits
        self._n_probe = n_probe
        self._points: list[list[float]] = []
        self._embeddings: np.ndarray | None = None
        self._pca_comp: np.ndarray | None = None
        self._pca_mean: np.ndarray | None = None
        self._grid: CellListND | None = None
        self._grid_nearest: list[int] | None = None

    def build(self, points: Sequence[Sequence[float]]) -> None:
        self._points = [list(point) for point in points]
        self._embeddings = None
        self._pca_comp = None
        self._pca_mean = None
        self._grid = None
        self._grid_nearest = None

        if not self._points:
            return

        embeddings = np.asarray(self._points, dtype=np.float32)
        self._embeddings = embeddings

        n_points, n_dims = embeddings.shape
        dims = max(1, min(self._grid_dims, n_points, n_dims))
        splits = tuple([self._grid_splits] * dims)

        pca = PCA(n_components=dims)
        projected = pca.fit_transform(embeddings)
        self._pca_comp = pca.components_.T.astype(np.float32)
        self._pca_mean = pca.mean_.astype(np.float32)

        bounds = [
            (float(projected[:, d].min()), float(projected[:, d].max()))
            for d in range(dims)
        ]
        self._grid = CellListND(dims, bounds, splits)

        for idx, point in enumerate(projected):
            self._grid.insert(tuple(point), idx)

        total = 1
        for split in splits:
            total *= split
        occupied = [self._grid._lin(cell_id) for cell_id in self._grid.cells]
        _, self._grid_nearest = bfs_nearest_nonempty(total, self._grid.neighbors, occupied)

    def query(self, point: Sequence[float], k: int) -> list[Neighbor]:
        if not self._points or k <= 0 or self._embeddings is None or self._grid is None:
            return []

        vec = np.asarray(point, dtype=np.float32)
        n_dims = self._embeddings.shape[1]
        if vec.shape[0] < n_dims:
            vec = np.pad(vec, (0, n_dims - vec.shape[0]))
        elif vec.shape[0] > n_dims:
            vec = vec[:n_dims]

        indices = self._grid_query_vector_multiprobe(vec, k, self._n_probe)
        return [Neighbor(index, l2_distance(point, self._points[index])) for index in indices]

    def _grid_query_vector_multiprobe(self, vec: np.ndarray, k: int, n_probe: int) -> list[int]:
        assert self._pca_comp is not None
        assert self._pca_mean is not None
        assert self._grid is not None
        assert self._grid_nearest is not None
        assert self._embeddings is not None

        pt = (vec - self._pca_mean) @ self._pca_comp
        primary_cid = self._grid._cell_index(tuple(pt))
        probed_cells = self._get_nearest_cells(pt, primary_cid, n_probe)

        candidate_lists = [
            self._grid.cells[cid] for cid in probed_cells if cid in self._grid.cells
        ]

        if not candidate_lists:
            nearest_cid = self._grid._unlin(self._grid_nearest[self._grid._lin(primary_cid)])
            if nearest_cid in self._grid.cells:
                candidate_lists = [self._grid.cells[nearest_cid]]

        if not candidate_lists:
            return []

        candidate_indices = np.concatenate(candidate_lists).astype(np.int32)
        candidate_vectors = self._embeddings[candidate_indices]
        dists = np.linalg.norm(candidate_vectors - vec, axis=1)

        if len(dists) <= k:
            top_k_local = np.argsort(dists)
        else:
            idx = np.argpartition(dists, k)[:k]
            top_k_local = idx[np.argsort(dists[idx])]

        return candidate_indices[top_k_local].tolist()

    def _get_nearest_cells(
        self,
        pt: np.ndarray,
        center_cid: tuple[int, ...],
        n_probe: int,
    ) -> list[tuple[int, ...]]:
        assert self._grid is not None

        if n_probe <= 1:
            return [center_cid]

        center_arr = np.array(center_cid)
        pt_arr = np.array(pt)
        grid_splits = np.array(self._grid.splits)
        cell_size = np.array(self._grid.cell_size)
        grid_mins = np.array(self._grid.mins)

        cell_origins = grid_mins + center_arr * cell_size
        cell_midpoints = cell_origins + cell_size * 0.5

        search_directions: list[tuple[int, int]] = []
        for dim in range(self._grid.dims):
            if pt_arr[dim] >= cell_midpoints[dim]:
                search_directions.append((0, 1))
            else:
                search_directions.append((-1, 0))

        offsets = np.array(list(itertools.product(*search_directions)), dtype=np.int32)
        neighbor_coords = center_arr + offsets

        in_bounds = np.all((neighbor_coords >= 0) & (neighbor_coords < grid_splits), axis=1)
        is_center = np.all(offsets == 0, axis=1)
        valid_mask = in_bounds & ~is_center

        valid_offsets = offsets[valid_mask]
        valid_neighbors = neighbor_coords[valid_mask]

        if valid_neighbors.shape[0] == 0:
            return [center_cid]

        cell_mins_arr = cell_origins
        cell_maxs_arr = cell_origins + cell_size
        dist_sq = np.zeros(valid_neighbors.shape[0], dtype=np.float32)

        for dim in range(self._grid.dims):
            moves_right = valid_offsets[:, dim] > 0
            moves_left = valid_offsets[:, dim] < 0

            dr = float(pt_arr[dim] - cell_maxs_arr[dim])
            dist_sq[moves_right] += dr * dr

            dl = float(cell_mins_arr[dim] - pt_arr[dim])
            dist_sq[moves_left] += dl * dl

        sorted_idx = np.argsort(dist_sq)[: n_probe - 1]
        top_neighbors = [tuple(valid_neighbors[i]) for i in sorted_idx]
        return [center_cid] + top_neighbors


def create_index(algo: str) -> NearestNeighbors:
    if algo == "brute_force":
        return BruteForceNN()
    if algo == "multiprobe":
        return MultiProbeNN()

    raise ValueError(f"unknown algorithm: {algo}")
