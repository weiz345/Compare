#pragma once

#include "ann_interface.hpp"
#include "multiprobe_grid.hpp"
#include "pca.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

class MultiProbeNN : public NearestNeighbors {
public:
    MultiProbeNN(int grid_dims = 3, int grid_splits = 6, int n_probe = 4)
        : grid_dims_(grid_dims), grid_splits_(grid_splits), n_probe_(n_probe) {}

    void build(const std::vector<std::vector<float>>& points) override {
        points_ = points;
        pca_ = PCAModel{};
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

        pca_ = fit_pca(points_, dims);
        const std::vector<std::vector<float>> projected = pca_project_all(pca_, points_);

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
        grid_ = std::make_unique<CellListND>(dims, bounds, splits);

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

        grid_nearest_ = bfs_nearest_nonempty(
                           total,
                           [this](int value) { return grid_->neighbors(value); },
                           occupied)
                            .second;
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
        const std::vector<float> pt = pca_project(pca_, vec);
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
    PCAModel pca_;
    std::unique_ptr<CellListND> grid_;
    std::vector<int> grid_nearest_;
    bool ready_ = false;
};
