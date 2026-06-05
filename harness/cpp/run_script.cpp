#include "algorithms.hpp"
#include "script_loader.hpp"

#include <iostream>
#include <string>

namespace {

constexpr const char* kDefaultAlgo = "brute_force";

void run_script(const std::string& path, const std::string& algo) {
    std::unique_ptr<NearestNeighbors> index = create_index(algo);

    for (const Operation& operation : load_script(path)) {
        if (std::holds_alternative<BuildOp>(operation)) {
            index->build(std::get<BuildOp>(operation).points);
            std::cout << "build ok\n";
            continue;
        }

        const QueryOp& query = std::get<QueryOp>(operation);
        const std::vector<Neighbor> results = index->query(query.point, query.k);
        std::cout << "query k " << query.k << '\n';
        for (const Neighbor& neighbor : results) {
            std::cout << neighbor.index << '\n';
        }
    }
}

struct RunnerOptions {
    std::string script_path = "harness/script.cases";
    std::string algo = kDefaultAlgo;
};

RunnerOptions parse_args(int argc, char* argv[]) {
    RunnerOptions options;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--algo") {
            if (i + 1 >= argc) {
                throw std::runtime_error("--algo requires a value");
            }
            options.algo = argv[++i];
            continue;
        }

        if (arg.rfind("--", 0) == 0) {
            throw std::runtime_error("unknown flag: " + arg);
        }

        options.script_path = arg;
    }

    return options;
}

}  // namespace

int main(int argc, char* argv[]) {
    try {
        const RunnerOptions options = parse_args(argc, argv);
        run_script(options.script_path, options.algo);
    } catch (const std::exception& ex) {
        std::cerr << ex.what() << '\n';
        return 1;
    }

    return 0;
}
