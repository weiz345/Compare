#pragma once

#include "ann_interface.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <deque>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ann {
namespace detail {

struct PCAModel {
    std::vector<float> mean;
    std::vector<std::vector<float>> comp;
    int components = 0;
};

inline std::vector<float> mat_vec_mul(
    const std::vector<std::vector<float>>& matrix,
    const std::vector<float>& vec) {
    std::vector<float> out(matrix.size(), 0.0f);
    for (std::size_t row = 0; row < matrix.size(); ++row) {
        for (std::size_t col = 0; col < vec.size(); ++col) {
            out[row] += matrix[row][col] * vec[col];
        }
    }
    return out;
}

inline float dot(const std::vector<float>& a, const std::vector<float>& b) {
    float sum = 0.0f;
    for (std::size_t i = 0; i < a.size(); ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

inline std::vector<std::vector<float>> covariance(
    const std::vector<std::vector<float>>& centered) {
    const std::size_t n_samples = centered.size();
    const std::size_t n_features = centered.empty() ? 0 : centered[0].size();
    std::vector<std::vector<float>> cov(
        n_features, std::vector<float>(n_features, 0.0f));

    if (n_samples <= 1) {
        return cov;
    }

    const float denom = static_cast<float>(n_samples - 1);
    for (std::size_t i = 0; i < n_features; ++i) {
        for (std::size_t j = 0; j < n_features; ++j) {
            float sum = 0.0f;
            for (std::size_t s = 0; s < n_samples; ++s) {
                sum += centered[s][i] * centered[s][j];
            }
            cov[i][j] = sum / denom;
        }
    }

    return cov;
}

inline std::vector<float> power_eigenvector(std::vector<std::vector<float>> matrix) {
    const std::size_t n = matrix.size();
    std::vector<float> vec(n, 1.0f / std::sqrt(static_cast<float>(n)));

    for (int iter = 0; iter < 200; ++iter) {
        std::vector<float> next = mat_vec_mul(matrix, vec);
        const float norm = std::sqrt(dot(next, next));
        if (norm <= 1e-12f) {
            break;
        }
        for (float& value : next) {
            value /= norm;
        }
        vec = next;
    }

    return vec;
}

inline void deflate(std::vector<std::vector<float>>& matrix, const std::vector<float>& eigenvector) {
    const float eigenvalue = dot(eigenvector, mat_vec_mul(matrix, eigenvector));
    for (std::size_t i = 0; i < matrix.size(); ++i) {
        for (std::size_t j = 0; j < matrix.size(); ++j) {
            matrix[i][j] -= eigenvalue * eigenvector[i] * eigenvector[j];
        }
    }
}

inline PCAModel fit_pca(const std::vector<std::vector<float>>& points, int n_components) {
    PCAModel model;
    if (points.empty()) {
        return model;
    }

    const std::size_t n_samples = points.size();
    const std::size_t n_features = points[0].size();
    n_components = std::max(1, std::min(n_components, static_cast<int>(std::min(n_samples, n_features))));

    model.mean.assign(n_features, 0.0f);
    for (const auto& point : points) {
        for (std::size_t f = 0; f < n_features; ++f) {
            model.mean[f] += point[f];
        }
    }
    for (float& value : model.mean) {
        value /= static_cast<float>(n_samples);
    }

    std::vector<std::vector<float>> centered(n_samples, std::vector<float>(n_features, 0.0f));
    for (std::size_t s = 0; s < n_samples; ++s) {
        for (std::size_t f = 0; f < n_features; ++f) {
            centered[s][f] = points[s][f] - model.mean[f];
        }
    }

    std::vector<std::vector<float>> cov = covariance(centered);
    model.components = n_components;
    model.comp.assign(n_features, std::vector<float>(static_cast<std::size_t>(n_components), 0.0f));

    for (int c = 0; c < n_components; ++c) {
        const std::vector<float> eigenvector = power_eigenvector(cov);
        for (std::size_t f = 0; f < n_features; ++f) {
            model.comp[f][static_cast<std::size_t>(c)] = eigenvector[f];
        }
        deflate(cov, eigenvector);
    }

    return model;
}

inline std::vector<float> pca_project(const PCAModel& model, const std::vector<float>& point) {
    std::vector<float> centered(point.size(), 0.0f);
    for (std::size_t f = 0; f < point.size(); ++f) {
        centered[f] = point[f] - model.mean[f];
    }

    std::vector<float> projected(static_cast<std::size_t>(model.components), 0.0f);
    for (int c = 0; c < model.components; ++c) {
        for (std::size_t f = 0; f < centered.size(); ++f) {
            projected[static_cast<std::size_t>(c)] += centered[f] * model.comp[f][static_cast<std::size_t>(c)];
        }
    }

    return projected;
}

inline std::vector<std::vector<float>> pca_project_all(
    const PCAModel& model,
    const std::vector<std::vector<float>>& points) {
    std::vector<std::vector<float>> projected;
    projected.reserve(points.size());
    for (const auto& point : points) {
        projected.push_back(pca_project(model, point));
    }
    return projected;
}

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
        cells_[cell_index(coord)].push_back(index_id);
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

inline std::vector<int> bfs_nearest_nonempty(
    int total_nodes,
    const std::function<std::vector<int>(int)>& neighbors_fn,
    const std::vector<int>& occupied_nodes) {
    const int inf = 1'000'000'000;
    std::vector<int> nearest(static_cast<std::size_t>(total_nodes), -1);
    std::vector<int> dist(static_cast<std::size_t>(total_nodes), inf);
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

    return nearest;
}

}  // namespace detail

class BruteForceNN : public NearestNeighbors {
public:
    void build(const std::vector<std::vector<float>>& points) override {
        points_ = points;
    }

    std::vector<Neighbor> query(const std::vector<float>& point, int k) const override {
        if (points_.empty() || k <= 0) {
            return {};
        }

        std::vector<Neighbor> candidates;
        candidates.reserve(points_.size());

        for (std::size_t i = 0; i < points_.size(); ++i) {
            candidates.push_back({i, l2_distance(point, points_[i])});
        }

        const int count = std::min(k, static_cast<int>(candidates.size()));
        std::partial_sort(
            candidates.begin(),
            candidates.begin() + count,
            candidates.end(),
            [](const Neighbor& a, const Neighbor& b) { return a.distance < b.distance; });

        candidates.resize(static_cast<std::size_t>(count));
        return candidates;
    }

private:
    std::vector<std::vector<float>> points_;
};

class MultiProbeNN : public NearestNeighbors {
public:
    MultiProbeNN(int grid_dims = 3, int grid_splits = 6, int n_probe = 4)
        : grid_dims_(grid_dims), grid_splits_(grid_splits), n_probe_(n_probe) {}

    void build(const std::vector<std::vector<float>>& points) override {
        points_ = points;
        pca_ = detail::PCAModel{};
        grid_.reset();
        grid_nearest_.clear();
        ready_ = false;

        if (points_.empty()) {
            return;
        }

        const std::size_t n_points = points_.size();
        const std::size_t n_dims = points_[0].size();
        const int dims = static_cast<int>(std::max<std::size_t>(
            1, std::min({static_cast<std::size_t>(grid_dims_), n_points, n_dims})));

        pca_ = detail::fit_pca(points_, dims);
        const std::vector<std::vector<float>> projected = detail::pca_project_all(pca_, points_);

        std::vector<std::pair<float, float>> bounds(static_cast<std::size_t>(dims));
        for (int d = 0; d < dims; ++d) {
            float min_v = std::numeric_limits<float>::infinity();
            float max_v = -std::numeric_limits<float>::infinity();
            for (const auto& point : projected) {
                min_v = std::min(min_v, point[static_cast<std::size_t>(d)]);
                max_v = std::max(max_v, point[static_cast<std::size_t>(d)]);
            }
            bounds[static_cast<std::size_t>(d)] = {min_v, max_v};
        }

        const std::vector<int> splits(static_cast<std::size_t>(dims), grid_splits_);
        grid_ = std::make_unique<detail::CellListND>(dims, bounds, splits);

        for (std::size_t idx = 0; idx < projected.size(); ++idx) {
            grid_->insert(projected[idx], static_cast<int>(idx));
        }

        int total = 1;
        for (int split : splits) {
            total *= split;
        }

        std::vector<int> occupied;
        for (const auto& entry : grid_->cells()) {
            occupied.push_back(grid_->lin(entry.first));
        }

        grid_nearest_ = detail::bfs_nearest_nonempty(
            total,
            [this](int value) { return grid_->neighbors(value); },
            occupied);
        ready_ = true;
    }

    std::vector<Neighbor> query(const std::vector<float>& point, int k) const override {
        if (!ready_ || k <= 0) {
            return {};
        }

        const std::vector<float> vec = normalize_query(point, points_[0].size());
        const std::vector<int> indices = grid_query_vector_multiprobe(vec, k, n_probe_);

        std::vector<Neighbor> results;
        results.reserve(indices.size());
        for (int index : indices) {
            results.push_back({static_cast<std::size_t>(index), l2_distance(point, points_[index])});
        }
        return results;
    }

private:
    std::vector<int> grid_query_vector_multiprobe(
        const std::vector<float>& vec,
        int k,
        int n_probe) const {
        const std::vector<float> pt = detail::pca_project(pca_, vec);
        const std::vector<int> primary_cid = grid_->cell_index(pt);
        const std::vector<std::vector<int>> probed_cells = get_nearest_cells(pt, primary_cid, n_probe);

        std::vector<int> candidate_indices;
        for (const auto& cid : probed_cells) {
            const auto it = grid_->cells().find(cid);
            if (it != grid_->cells().end()) {
                candidate_indices.insert(candidate_indices.end(), it->second.begin(), it->second.end());
            }
        }

        if (candidate_indices.empty()) {
            const int nearest = grid_nearest_[static_cast<std::size_t>(grid_->lin(primary_cid))];
            const std::vector<int> nearest_cid = grid_->unlin(nearest);
            const auto it = grid_->cells().find(nearest_cid);
            if (it != grid_->cells().end()) {
                candidate_indices = it->second;
            }
        }

        if (candidate_indices.empty()) {
            return {};
        }

        std::vector<int> order(candidate_indices.size());
        for (std::size_t i = 0; i < order.size(); ++i) {
            order[i] = static_cast<int>(i);
        }

        const auto dist_at = [&](int local_idx) {
            return l2_distance(vec, points_[candidate_indices[static_cast<std::size_t>(local_idx)]]);
        };

        const int count = std::min(k, static_cast<int>(order.size()));
        std::partial_sort(
            order.begin(),
            order.begin() + count,
            order.end(),
            [&](int lhs, int rhs) { return dist_at(lhs) < dist_at(rhs); });

        std::vector<int> results;
        results.reserve(static_cast<std::size_t>(count));
        for (int i = 0; i < count; ++i) {
            results.push_back(candidate_indices[static_cast<std::size_t>(order[static_cast<std::size_t>(i)])]);
        }
        return results;
    }

    std::vector<std::vector<int>> get_nearest_cells(
        const std::vector<float>& pt,
        const std::vector<int>& center_cid,
        int n_probe) const {
        if (n_probe <= 1) {
            return {center_cid};
        }

        const int dims = grid_->dims();
        std::vector<float> cell_origins(static_cast<std::size_t>(dims));
        std::vector<float> cell_midpoints(static_cast<std::size_t>(dims));
        for (int dim = 0; dim < dims; ++dim) {
            cell_origins[static_cast<std::size_t>(dim)] =
                grid_->mins()[static_cast<std::size_t>(dim)] +
                static_cast<float>(center_cid[static_cast<std::size_t>(dim)]) *
                    grid_->cell_size()[static_cast<std::size_t>(dim)];
            cell_midpoints[static_cast<std::size_t>(dim)] =
                cell_origins[static_cast<std::size_t>(dim)] +
                grid_->cell_size()[static_cast<std::size_t>(dim)] * 0.5f;
        }

        std::vector<int> left_offsets(static_cast<std::size_t>(dims), 0);
        std::vector<int> right_offsets(static_cast<std::size_t>(dims), 0);
        for (int dim = 0; dim < dims; ++dim) {
            if (pt[static_cast<std::size_t>(dim)] >= cell_midpoints[static_cast<std::size_t>(dim)]) {
                right_offsets[static_cast<std::size_t>(dim)] = 1;
            } else {
                left_offsets[static_cast<std::size_t>(dim)] = -1;
            }
        }

        struct NeighborCell {
            std::vector<int> coords;
            float dist_sq;
        };
        std::vector<NeighborCell> candidates;

        const int combinations = 1 << dims;
        for (int mask = 0; mask < combinations; ++mask) {
            std::vector<int> offset(static_cast<std::size_t>(dims), 0);
            bool is_center = true;
            for (int dim = 0; dim < dims; ++dim) {
                const bool use_right = (mask >> dim) & 1;
                offset[static_cast<std::size_t>(dim)] =
                    use_right ? right_offsets[static_cast<std::size_t>(dim)]
                              : left_offsets[static_cast<std::size_t>(dim)];
                if (offset[static_cast<std::size_t>(dim)] != 0) {
                    is_center = false;
                }
            }
            if (is_center) {
                continue;
            }

            std::vector<int> coords(static_cast<std::size_t>(dims));
            bool in_bounds = true;
            for (int dim = 0; dim < dims; ++dim) {
                coords[static_cast<std::size_t>(dim)] =
                    center_cid[static_cast<std::size_t>(dim)] + offset[static_cast<std::size_t>(dim)];
                if (coords[static_cast<std::size_t>(dim)] < 0 ||
                    coords[static_cast<std::size_t>(dim)] >= grid_->splits()[dim]) {
                    in_bounds = false;
                    break;
                }
            }
            if (!in_bounds) {
                continue;
            }

            float dist_sq = 0.0f;
            for (int dim = 0; dim < dims; ++dim) {
                if (offset[static_cast<std::size_t>(dim)] > 0) {
                    const float dr = pt[static_cast<std::size_t>(dim)] -
                                     (cell_origins[static_cast<std::size_t>(dim)] +
                                      grid_->cell_size()[static_cast<std::size_t>(dim)]);
                    dist_sq += dr * dr;
                } else if (offset[static_cast<std::size_t>(dim)] < 0) {
                    const float dl =
                        cell_origins[static_cast<std::size_t>(dim)] - pt[static_cast<std::size_t>(dim)];
                    dist_sq += dl * dl;
                }
            }

            candidates.push_back({coords, dist_sq});
        }

        if (candidates.empty()) {
            return {center_cid};
        }

        const int neighbor_count = std::min(n_probe - 1, static_cast<int>(candidates.size()));
        std::partial_sort(
            candidates.begin(),
            candidates.begin() + neighbor_count,
            candidates.end(),
            [](const NeighborCell& a, const NeighborCell& b) { return a.dist_sq < b.dist_sq; });

        std::vector<std::vector<int>> result;
        result.push_back(center_cid);
        for (int i = 0; i < neighbor_count; ++i) {
            result.push_back(candidates[static_cast<std::size_t>(i)].coords);
        }
        return result;
    }

    int grid_dims_;
    int grid_splits_;
    int n_probe_;
    std::vector<std::vector<float>> points_;
    detail::PCAModel pca_;
    std::unique_ptr<detail::CellListND> grid_;
    std::vector<int> grid_nearest_;
    bool ready_ = false;
};

inline std::unique_ptr<NearestNeighbors> create_index(const std::string& algo) {
    if (algo == "brute_force") {
        return std::make_unique<BruteForceNN>();
    }
    if (algo == "multiprobe") {
        return std::make_unique<MultiProbeNN>();
    }

    throw std::runtime_error("unknown algorithm: " + algo);
}

}  // namespace ann

using ann::BruteForceNN;
using ann::MultiProbeNN;
using ann::create_index;
