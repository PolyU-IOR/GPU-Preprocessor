#ifdef __APPLE__
#define _DARWIN_C_SOURCE
#endif
#define _POSIX_C_SOURCE 200809L
#include "gpupresolve_qp.h"
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#ifdef __APPLE__
#include <mach-o/dyld.h>
#endif
#ifndef PATH_MAX
#define PATH_MAX 4096
#endif

static const char *default_input(char *path, size_t capacity) {
    char executable[PATH_MAX];
    int found = 0;
#if defined(__linux__)
    ssize_t length = readlink("/proc/self/exe", executable, sizeof(executable) - 1);
    if (length > 0 && (size_t)length < sizeof(executable) - 1) {
        executable[length] = 0;
        found = 1;
    }
#elif defined(__APPLE__)
    uint32_t size = sizeof(executable);
    found = _NSGetExecutablePath(executable, &size) == 0;
#endif
    if (found) {
        char *slash = strrchr(executable, '/');
        if (slash) {
            *slash = 0;
            int length =
                snprintf(path, capacity, "%s/%s", executable, INSTALL_EXAMPLE_RELATIVE);
            if (length > 0 && (size_t)length < capacity && access(path, R_OK) == 0)
                return path;
        }
    }
    return EXAMPLE_MPS;
}

int main(int argc, char **argv) {
    Config config = default_config();
    config.rules = FIXED_COLS;
    char input_buffer[PATH_MAX];
    const char *input = default_input(input_buffer, sizeof(input_buffer));
    int explicit_input = 0;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--gpu"))
            config.backend = CUDA;
        else if (!strcmp(argv[i], "--help")) {
            printf("Usage: presolve_example [--gpu] [INPUT.mps]\n");
            return 0;
        } else if (argv[i][0] == '-' || explicit_input) {
            fprintf(stderr, "Usage: presolve_example [--gpu] [INPUT.mps]\n");
            return 2;
        } else {
            input = argv[i];
            explicit_input = 1;
        }
    }
    Problem original = {0};
    Result result = {0};
    char message[256];
    int code = 1;
    printf("input=%s\n", input);
    Status status = read_mps(input, &original, message, sizeof(message));
    if (status) {
        fprintf(stderr, "%s: %s\n", status_string(status), message);
        goto done;
    }
    status = presolve(&original, &config, &result);
    if (status != OK && status != TIME_LIMIT) {
        fprintf(stderr, "%s: %s\n", status_string(status), result.message);
        goto done;
    }
    printf("backend=%s status=%s dimensions=(%d, %d) -> (%d, %d)\n",
           result.backend == CUDA ? "CUDA" : "CPU", status_string(status), original.A.rows,
           original.A.cols, result.reduced.A.rows, result.reduced.A.cols);
    printf("A nnz=%d Q nnz=%d presolve_seconds=%.9g\n", result.reduced.A.nnz, result.reduced.Q.nnz,
           result.presolve_seconds);
    if (result.reduced.A.rows != 1 || result.reduced.A.cols != 2)
        goto done;
    /* An external QP solver supplies these vectors in a real application.
     * This demonstration uses the analytically known reduced optimum. */
    const double xr[] = {3., 1.}, yr[] = {0.}, zr[] = {0., 0.};
    double x[3], y[1], z[3];
    status = postsolve(result.state, xr, 2, yr, 1, zr, 2, x, y, z);
    if (status)
        goto done;
    KKT kkt;
    status = compute_kkt_backend(&original, x, y, z, result.backend, 0, &kkt);
    if (status)
        goto done;
    printf("original x=[%.17g, %.17g, %.17g] objective=%.17g KKT=(%.9g, %.9g, %.9g)\n", x[0], x[1],
           x[2], kkt.primal_objective, kkt.primal_feasibility, kkt.dual_feasibility,
           kkt.relative_gap);
    if (fabs(x[0] - 1.) > 1e-10 || fabs(x[1] - 3.) > 1e-10 || fabs(x[2] - 1.) > 1e-10 ||
        kkt.primal_feasibility > 1e-10 || kkt.dual_feasibility > 1e-10 || kkt.relative_gap > 1e-10)
        goto done;
    code = 0;
done:
    if (code)
        fprintf(stderr, "Example failed (%s).\n", status_string(status));
    result_free(&result);
    problem_free(&original);
    return code;
}
