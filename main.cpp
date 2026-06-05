#include "nearest_neighbors.hpp"

#include <iostream>

int main() {
    BruteForceNN index;
    index.build({
        {0.0f, 0.0f},
        {1.0f, 0.0f},
        {0.0f, 1.0f},
        {5.0f, 5.0f},
    });

    const std::vector<float> query = {0.9f, 0.1f};

    for (const Neighbor& neighbor : index.query(query, 2)) {
        std::cout << "index=" << neighbor.index << " distance=" << neighbor.distance << '\n';
    }

    return 0;
}
