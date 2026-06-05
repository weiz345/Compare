#pragma once

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

struct BuildOp {
    std::vector<std::vector<float>> points;
};

struct QueryOp {
    std::vector<float> point;
    int k = 0;
};

using Operation = std::variant<BuildOp, QueryOp>;

inline std::vector<float> parse_floats(const std::string& text) {
    std::vector<float> values;
    std::istringstream stream(text);

    float value = 0.0f;
    while (stream >> value) {
        values.push_back(value);
    }

    return values;
}

inline std::vector<std::string> split_tokens(const std::string& text) {
    std::vector<std::string> tokens;
    std::istringstream stream(text);
    std::string token;
    while (stream >> token) {
        tokens.push_back(token);
    }
    return tokens;
}

inline std::vector<Operation> load_script(const std::string& path) {
    std::ifstream file(path);
    if (!file) {
        throw std::runtime_error("failed to open script file: " + path);
    }

    std::vector<Operation> operations;
    std::vector<std::vector<float>> pending_points;
    bool building = false;

    auto flush_build = [&]() {
        operations.push_back(BuildOp{pending_points});
        pending_points.clear();
        building = false;
    };

    std::string line;
    while (std::getline(file, line)) {
        const std::size_t start = line.find_first_not_of(" \t");
        if (start == std::string::npos || line[start] == '#') {
            continue;
        }

        const std::string trimmed = line.substr(start);
        if (trimmed == "build") {
            if (building) {
                flush_build();
            }
            building = true;
            continue;
        }

        if (trimmed.rfind("query ", 0) == 0) {
            if (building) {
                flush_build();
            }

            const std::vector<std::string> parts = split_tokens(trimmed);
            std::size_t split_at = parts.size();
            for (std::size_t i = 1; i < parts.size(); ++i) {
                if (parts[i] == "k") {
                    split_at = i;
                    break;
                }
            }

            if (split_at >= parts.size() - 1) {
                throw std::runtime_error("query line missing k: " + trimmed);
            }

            QueryOp query;
            for (std::size_t i = 1; i < split_at; ++i) {
                query.point.push_back(std::stof(parts[i]));
            }
            query.k = std::stoi(parts[split_at + 1]);
            operations.push_back(query);
            continue;
        }

        if (building) {
            pending_points.push_back(parse_floats(trimmed));
            continue;
        }

        throw std::runtime_error("unexpected line: " + trimmed);
    }

    if (building) {
        flush_build();
    }

    return operations;
}
