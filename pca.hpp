#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

struct PCAModel {
    std::vector<float> mean;
    std::vector<std::vector<float>> comp;  // comp[feature][component]
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
