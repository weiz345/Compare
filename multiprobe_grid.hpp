#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <deque>
#include <functional>
#include <map>
#include <tuple>
#include <utility>
#include <vector>

class CellListND {
public:
    CellListND(int dims, const std::vector<std::pair<float, float>>& bounds, const std::vector<int>& splits)
        : dims_(dims), splits_(splits), mins_(dims), maxs_(dims), cell_size_(dims) {
        for (int dim = 0; dim < dims; ++dim) {
            mins_[dim] = bounds[static_cast<std::size_t>(dim)].first;
            maxs_[dim] = bounds[static_cast<std::size_t>(dim)].second;
            const float size = (maxs_[dim] - mins_[dim]) / static_cast<float>(splits_[dim]);
            cell_size_[dim] = size > 0.0f ? size : 1.0f;
        }
    }

    int dims() const { return dims_; }
    const std::vector<int>& splits() const { return splits_; }
    const std::vector<float>& mins() const { return mins_; }
    const std::vector<float>& cell_size() const { return cell_size_; }

    std::vector<int> cell_index(const std::vector<float>& coord) const {
        std::vector<int> idx(static_cast<std::size_t>(dims_));
        for (int dim = 0; dim < dims_; ++dim) {
            const float ratio = (coord[static_cast<std::size_t>(dim)] - mins_[dim]) / cell_size_[dim];
            int cell = static_cast<int>(ratio);
            cell = std::max(0, std::min(splits_[dim] - 1, cell));
            idx[static_cast<std::size_t>(dim)] = cell;
        }
        return idx;
    }

    void insert(const std::vector<float>& coord, int index_id) {
        const std::vector<int> cell_id = cell_index(coord);
        cells_[cell_id].push_back(index_id);
    }

    int lin(const std::vector<int>& cell_id) const {
        int result = 0;
        int multiplier = 1;
        for (int dim = dims_ - 1; dim >= 0; --dim) {
            result += cell_id[static_cast<std::size_t>(dim)] * multiplier;
            multiplier *= splits_[dim];
        }
        return result;
    }

    std::vector<int> unlin(int value) const {
        std::vector<int> result(static_cast<std::size_t>(dims_));
        for (int dim = 0; dim < dims_; ++dim) {
            int multiplier = 1;
            for (int dim2 = dim + 1; dim2 < dims_; ++dim2) {
                multiplier *= splits_[dim2];
            }
            result[static_cast<std::size_t>(dim)] = (value / multiplier) % splits_[dim];
        }
        return result;
    }

    std::vector<int> neighbors(int value) const {
        const std::vector<int> cell_id = unlin(value);
        std::vector<int> result;

        for (int dim = 0; dim < dims_; ++dim) {
            std::vector<int> plus = cell_id;
            plus[static_cast<std::size_t>(dim)] += 1;
            if (plus[static_cast<std::size_t>(dim)] >= 0 && plus[static_cast<std::size_t>(dim)] < splits_[dim]) {
                result.push_back(lin(plus));
            }

            std::vector<int> minus = cell_id;
            minus[static_cast<std::size_t>(dim)] -= 1;
            if (minus[static_cast<std::size_t>(dim)] >= 0 && minus[static_cast<std::size_t>(dim)] < splits_[dim]) {
                result.push_back(lin(minus));
            }
        }

        return result;
    }

    const std::map<std::vector<int>, std::vector<int>>& cells() const { return cells_; }

private:
    int dims_;
    std::vector<int> splits_;
    std::vector<float> mins_;
    std::vector<float> maxs_;
    std::vector<float> cell_size_;
    std::map<std::vector<int>, std::vector<int>> cells_;
};

inline std::pair<std::vector<int>, std::vector<int>> bfs_nearest_nonempty(
    int total_nodes,
    const std::function<std::vector<int>(int)>& neighbors_fn,
    const std::vector<int>& occupied_nodes) {
    const int inf = 1'000'000'000;
    std::vector<int> dist(static_cast<std::size_t>(total_nodes), inf);
    std::vector<int> nearest(static_cast<std::size_t>(total_nodes), -1);
    std::deque<int> queue;

    for (int node : occupied_nodes) {
        dist[static_cast<std::size_t>(node)] = 0;
        nearest[static_cast<std::size_t>(node)] = node;
        queue.push_back(node);
    }

    while (!queue.empty()) {
        const int node = queue.front();
        queue.pop_front();
        for (int neighbor : neighbors_fn(node)) {
            if (dist[static_cast<std::size_t>(neighbor)] == inf) {
                dist[static_cast<std::size_t>(neighbor)] = dist[static_cast<std::size_t>(node)] + 1;
                nearest[static_cast<std::size_t>(neighbor)] = nearest[static_cast<std::size_t>(node)];
                queue.push_back(neighbor);
            }
        }
    }

    return {dist, nearest};
}
