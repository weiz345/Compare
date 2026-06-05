#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

struct Neighbor {
    std::size_t index;
    float distance;
};

class BruteForceNN {
public:
    void build(const std::vector<std::vector<float>>& points) {
        points_ = points;
    }

    // Return up to k nearest neighbors, sorted by ascending distance.
    std::vector<Neighbor> query(const std::vector<float>& point, int k) const {
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
    static float l2_distance(const std::vector<float>& a, const std::vector<float>& b) {
        const std::size_t dims = std::min(a.size(), b.size());
        float sum_sq = 0.0f;

        for (std::size_t i = 0; i < dims; ++i) {
            const float diff = a[i] - b[i];
            sum_sq += diff * diff;
        }

        return std::sqrt(sum_sq);
    }

    std::vector<std::vector<float>> points_;
};
