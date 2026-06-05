#include "nearest_neighbors.hpp"
#include "script_loader.hpp"

#include <iostream>
#include <string>

namespace {

void run_script(const std::string& path) {
    BruteForceNN index;

    for (const Operation& operation : load_script(path)) {
        if (std::holds_alternative<BuildOp>(operation)) {
            index.build(std::get<BuildOp>(operation).points);
            std::cout << "build ok\n";
            continue;
        }

        const QueryOp& query = std::get<QueryOp>(operation);
        const std::vector<Neighbor> results = index.query(query.point, query.k);
        std::cout << "query k " << query.k << '\n';
        for (const Neighbor& neighbor : results) {
            std::cout << neighbor.index << '\n';
        }
    }
}

}  // namespace

int main(int argc, char* argv[]) {
    const std::string path = (argc == 2) ? argv[1] : "script.cases";

    try {
        run_script(path);
    } catch (const std::exception& ex) {
        std::cerr << ex.what() << '\n';
        return 1;
    }

    return 0;
}
