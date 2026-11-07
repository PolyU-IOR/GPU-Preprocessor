#include "gpupresolve_qp.h"
#include <stdio.h>

/* Run with CUDA_VISIBLE_DEVICES empty: presolve must never select a CPU path. */
int main(void) {
    Problem problem = {0};
    DeviceProblemView device_problem = {0};
    Config config = default_config();
    Result result = {0};
    int failed = config.backend != CUDA || cuda_available(0);
    if (problem_alloc(&problem, 0, 0) != OK)
        return 1;
    const Backend backends[] = {CUDA, AUTO, CPU};
    for (size_t i = 0; i < sizeof(backends) / sizeof(*backends); ++i) {
        config.backend = backends[i];
        Status expected = config.backend == CPU ? INVALID_ARGUMENT : CUDA_ERROR;
        Status status = presolve(&problem, &config, &result);
        failed |= status != expected || result.state != NULL;
        result_free(&result);
        status = presolve_resident(&problem, &config, NULL, &result);
        failed |= status != expected || result.state != NULL;
        result_free(&result);
        status = presolve_device(&device_problem, &config, NULL, &result);
        failed |= status != expected || result.state != NULL;
        result_free(&result);
    }
    problem_free(&problem);
    puts(failed ? "CUDA requirement failed" : "CUDA required; CPU presolve rejected");
    return failed ? 1 : 0;
}
