#include "gpupresolve_qp.h"
#include <stdio.h>
#include <string.h>

/* Compile this separately, using only an installed header/library/package. */
int main(int argc, char **argv) {
    if ((argc != 2 && argc != 3) || (argc == 3 && strcmp(argv[2], "--gpu")))
        return 2;
    Matrix standalone = {0};
    const int indices[] = {0};
    const double values[] = {1.};
    if (matrix_from_triplets(&standalone, 1, 1, 1, indices, indices, values) != OK)
        return 1;
    matrix_free(&standalone);
    if (standalone.row_ptr || standalone.col_ind || standalone.values || standalone.nnz)
        return 1;
    Problem p = {0};
    Result r = {0};
    Config c = default_config();
    char message[256];
    c.backend = CUDA;
    ExecutionOptions options = default_execution_options();
    options.schedule = SCHEDULE_TIERED;
    options.batch_limit = 2;
    c.rules = FIXED_COLS;
    Status status = read_mps(argv[1], &p, message, sizeof(message));
    if (status == OK)
        status = presolve_with_options(&p, &c, &options, &r);
    PostsolveWorkspace *workspace = NULL;
    PostsolveInfo info;
    ExecutionInfo execution;
    if (!status)
        status = execution_info(r.state, &execution);
    if (!status && c.backend == CUDA && !execution.gpu_reductions)
        status = INVALID_ARGUMENT;
    TransferInfo transfers;
    if (!status)
        status = transfer_info(r.state, &transfers);
    if (!status && (transfers.model_uploads != 1 || transfers.resident_commits != 1))
        status = INVALID_ARGUMENT;
    if (!status)
        status = postsolve_info(r.state, &info);
    if (!status && (info.original_rows != 1 || info.original_cols != 3 || info.reduced_rows != 1 ||
                    info.reduced_cols != 2 || info.records != 1))
        status = INVALID_ARGUMENT;
    if (!status)
        status = postsolve_workspace_create(r.state, &workspace);
    const double xr[] = {3, 1}, yr[] = {0}, zr[] = {0, 0};
    double x[3], y[1], z[3];
    if (!status)
        status = postsolve_with_workspace(r.state, workspace, xr, 2, yr, 1, zr, 2, x, y, z);
    KKT kkt;
    if (!status)
        status = compute_kkt(&p, x, y, z, &kkt);
    if (!status && (kkt.primal_feasibility > 1e-10 || kkt.dual_feasibility > 1e-10 ||
                    kkt.relative_gap > 1e-10))
        status = NUMERICAL_ERROR;
    postsolve_workspace_free(&workspace);
    printf("external C consumer: %s\n", status_string(status));
    result_free(&r);
    problem_free(&p);
    return status == OK ? 0 : 1;
}
