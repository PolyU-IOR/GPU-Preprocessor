#include <gpu_preprocessor/lp.hpp>
#include <gpupresolve_qp.h>
#include <cstdio>
int main() {
    auto volatile lp_entry = &gpu_preprocessor::lp::preprocess;
    Config qp = default_config();
    if (!lp_entry || !(qp.rules & PROPAGATION)) return 1;
    std::puts("LP and QP public targets link together");
    return 0;
}
