#include "runner.h"
#include "mpi_ops.h"
#include <iostream>

int main(int argc, char* argv[]) {
    if (argc < 6) {
        std::cerr << "Usage: " << argv[0] << " <begin_size> <end_size> <factor> <algorithm> <mode>" << std::endl;
        return 1;
    }

    std::string begin_str = argv[1];
    std::string end_str = argv[2];
    int factor = std::stoi(argv[3]);
    std::string algo = argv[4];
    std::string mode = argv[5];

    MPIOps mpiOps;
    Runner runner(mpiOps, begin_str, end_str, factor, algo, mode);
    runner.run();

    return 0;
}

