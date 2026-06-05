#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

struct Neighbor {
    std::size_t index;
    float distance;
};

class NearestNeighbors {
public:
    virtual ~NearestNeighbors() = default;

    virtual void build(const std::vector<std::vector<float>>& points) = 0;
    virtual std::vector<Neighbor> query(const std::vector<float>& point, int k) const = 0;
};

inline float l2_distance(const std::vector<float>& a, const std::vector<float>& b) {
    const std::size_t dims = std::min(a.size(), b.size());
    float sum_sq = 0.0f;

    for (std::size_t i = 0; i < dims; ++i) {
        const float diff = a[i] - b[i];
        sum_sq += diff * diff;
    }

    return std::sqrt(sum_sq);
}

inline std::vector<float> normalize_query(
    const std::vector<float>& point,
    std::size_t n_dims) {
    std::vector<float> vec = point;
    if (vec.size() < n_dims) {
        vec.resize(n_dims, 0.0f);
    } else if (vec.size() > n_dims) {
        vec.resize(n_dims);
    }
    return vec;
}
