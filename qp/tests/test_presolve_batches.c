#include "internal.h"
#include "internal.h"
#include "support/qp_oracle.h"
static int checks, failures;
static Backend backend;
#define CHECK(e)                                                                                   \
    do {                                                                                           \
        ++checks;                                                                                  \
        if (!(e)) {                                                                                \
            ++failures;                                                                            \
            fprintf(stderr, "batch FAIL %d: %s\n", __LINE__, #e);                                  \
        }                                                                                          \
    } while (0)
static void matrix(Matrix *dst, int m, int n, COO *coo) {
    CHECK(coo_finish(coo, m, n, dst) == OK);
}
static void verify(Problem *p, Result *r, const double *target, const double *row_dual,
                   int bound_rows) {
    CHECK(state_export_host(r->state) == OK);
    int n = p->A.cols, m = p->A.rows, nr = r->reduced.A.cols, mr = r->reduced.A.rows;
    double *xr = checked_calloc(nr, sizeof(double)), *yr = checked_calloc(mr, sizeof(double)),
           *zr = checked_calloc(nr, sizeof(double));
    double *x = checked_calloc(n, sizeof(double)), *y = checked_calloc(m, sizeof(double)),
           *z = checked_calloc(n, sizeof(double));
    double *cpu = checked_calloc((size_t)2 * n + m, sizeof(double));
    CHECK(xr && yr && zr && x && y && z && cpu);
    for (int j = 0; j < nr; ++j)
        xr[j] = target[r->state->col_ids[j]];
    for (int i = 0; i < mr; ++i)
        yr[i] = bound_rows ? 0. : row_dual[r->state->row_ids[i]];
    PostsolveWorkspace *w = NULL, *reference = NULL;
    CHECK(postsolve_workspace_create(r->state, &w) == OK);
    CHECK(postsolve_workspace_create_backend(r->state, CPU, 0, &reference) == OK);
    CHECK(postsolve_with_workspace(r->state, reference, xr, nr, yr, mr, zr, nr, cpu,
                                       cpu + 2 * n, cpu + n) == OK);
    for (int repeat = 0; repeat < 3; ++repeat) {
        CHECK(postsolve_with_workspace(r->state, w, xr, nr, yr, mr, zr, nr, x, y, z) == OK);
        for (int j = 0; j < n; ++j) {
            CHECK(fabs(x[j] - target[j]) < 1e-9);
            CHECK(fabs(x[j] - cpu[j]) < 1e-10);
            CHECK(fabs(z[j] - cpu[n + j]) < 1e-9);
        }
        for (int i = 0; i < m; ++i)
            CHECK(fabs(y[i] - cpu[2 * n + i]) < 1e-9);
        KKT k;
        CHECK(compute_kkt(p, x, y, z, &k) == OK);
        CHECK(k.primal_feasibility < 1e-8 && k.dual_feasibility < 1e-8 && k.relative_gap < 1e-8);
    }
    postsolve_workspace_free(&w);
    postsolve_workspace_free(&reference);
    free(xr);
    free(yr);
    free(zr);
    free(x);
    free(y);
    free(z);
    free(cpu);
}
static void derived_bound_roundoff(void) {
    Problem p = {0};
    COO a = {0}, q = {0};
    const double target[] = {0., -.5}, dual[] = {0., 0.};
    CHECK(problem_alloc(&p, 2, 2) == OK);
    coo_add(&a, 0, 0, 3.);
    coo_add(&a, 0, 1, -1.);
    coo_add(&a, 1, 1, 1. / 3.);
    matrix(&p.A, 2, 2, &a);
    for (int j = 0; j < 2; ++j)
        coo_add(&q, j, j, 2.);
    matrix(&p.Q, 2, 2, &q);
    p.lower[0] = -2.;
    p.upper[0] = 2.;
    p.lower[1] = p.upper[1] = -.5;
    p.row_lower[0] = p.row_upper[0] = .5;
    p.row_lower[1] = p.row_upper[1] = nextafter(nextafter(-1. / 6., -INFINITY), -INFINITY);
    p.c[1] = 1.;
    Config cfg = default_config();
    cfg.backend = backend;
    cfg.rules = PROPAGATION | DOUBLETON_EQ;
    cfg.feasibility_tol = 1e-10;
    Result r = {0};
    CHECK(presolve(&p, &cfg, &r) == OK);
    verify(&p, &r, target, dual, 0);
    result_free(&r);
    cfg.feasibility_tol = 0.;
    CHECK(presolve(&p, &cfg, &r) == INFEASIBLE);
    result_free(&r);
    p.row_lower[1] = p.row_upper[1] = -.2;
    cfg.feasibility_tol = 1e-10;
    CHECK(presolve(&p, &cfg, &r) == INFEASIBLE);
    result_free(&r);
    problem_free(&p);
}
static void coupled_fixed(void) {
    int n = 256, m = 4;
    Problem p = {0};
    Result r = {0};
    COO q = {0}, a = {0};
    double target[256], dual[] = {.25, -.5, .75, 1.};
    CHECK(problem_alloc(&p, m, n) == OK);
    for (int j = 0; j < n; ++j) {
        target[j] = .25 * (j % 7 - 3);
        p.lower[j] = p.upper[j] = target[j];
        coo_add(&q, j, j, 4.);
        if (j + 1 < n) {
            coo_add(&q, j, j + 1, 1.);
            coo_add(&q, j + 1, j, 1.);
        }
        coo_add(&a, j % m, j, 1.);
    }
    matrix(&p.Q, n, n, &q);
    matrix(&p.A, m, n, &a);
    for (int i = 0; i < m; ++i)
        p.row_lower[i] = p.row_upper[i] = 0.;
    for (int j = 0; j < n; ++j) {
        p.c[j] = -4. * target[j] - (j ? target[j - 1] : 0.) - (j + 1 < n ? target[j + 1] : 0.) +
                 dual[j % m] + (j % 3 - 1);
        p.row_lower[j % m] += target[j];
        p.row_upper[j % m] += target[j];
    }
    Config cfg = default_config();
    cfg.backend = backend;
    cfg.rules = FIXED_COLS;
    CHECK(presolve(&p, &cfg, &r) == OK);
    CHECK(r.reduced.A.cols == 0 && r.reduced.A.rows == m);
    ExecutionInfo info;
    CHECK(execution_info(r.state, &info) == OK);
    TransferInfo traffic;
    CHECK(transfer_info(r.state, &traffic) == OK);
    if (backend == CUDA)
        CHECK(traffic.model_uploads == 1 && traffic.resident_commits == info.gpu_batches);
    if (backend == CUDA)
        CHECK(info.largest_batch == 256 && info.gpu_reductions == 256 && !info.cpu_fallbacks);
    else
        CHECK(!info.gpu_batches && !info.gpu_candidate_passes);
    verify(&p, &r, target, dual, 0);
    result_free(&r);
    problem_free(&p);
}
static void substitutions(int shared, int coupled, int limit, int tiered) {
    int m = coupled ? 12 : 64, n = shared ? m + 1 : 2 * m;
    Problem p = {0};
    Result r = {0};
    COO q = {0}, a = {0};
    double target[128], dual[64];
    CHECK(problem_alloc(&p, m, n) == OK);
    for (int j = 0; j < n; ++j) {
        target[j] = .25 * (j % 5 - 2);
        coo_add(&q, j, j, 4.);
    }
    for (int i = 0; i < m; ++i) {
        int pivot = shared ? i : 2 * i, support = shared ? m : 2 * i + 1;
        coo_add(&a, i, pivot, 1.);
        coo_add(&a, i, support, 1.);
        p.row_lower[i] = p.row_upper[i] = target[pivot] + target[support];
        dual[i] = .25 * (i % 3 - 1);
        if (!shared && i % 2 == 0) {
            p.lower[pivot] = -10.;
            p.upper[pivot] = 10.;
        }
    }
    if (coupled)
        for (int i = 0; i < m; ++i)
            for (int j = 0; j < m; ++j)
                coo_add(&q, 2 * i, 2 * j, 1.);
    matrix(&p.Q, n, n, &q);
    matrix(&p.A, m, n, &a);
    for (int j = 0; j < n; ++j) {
        for (int k = p.Q.row_ptr[j]; k < p.Q.row_ptr[j + 1]; ++k)
            p.c[j] -= p.Q.values[k] * target[p.Q.col_ind[k]];
        for (int i = 0; i < m; ++i)
            p.c[j] += matrix_get(&p.A, i, j) * dual[i];
    }
    Config cfg = default_config();
    cfg.backend = backend;
    cfg.rules = SINGLETON_EQ;
    cfg.max_q_fill = -1;
    cfg.max_q_fill_ratio = INFINITY;
    ExecutionOptions options = default_execution_options();
    options.batch_limit = limit;
    options.schedule = tiered ? SCHEDULE_TIERED : SCHEDULE_FIXED;
    CHECK(presolve_with_options(&p, &cfg, &options, &r) == OK);
    CHECK(r.reduced.A.cols == n - m && r.reduced.A.rows == (shared ? 0 : (m + 1) / 2));
    ExecutionInfo info;
    CHECK(execution_info(r.state, &info) == OK);
    TransferInfo traffic;
    CHECK(transfer_info(r.state, &traffic) == OK);
    if (backend == CUDA) {
        CHECK(traffic.model_uploads == 1 && traffic.resident_commits == info.gpu_batches);
        CHECK(info.gpu_reductions == (uint64_t)m && !info.cpu_fallbacks);
        CHECK(info.largest_batch == (size_t)(coupled ? 1 : limit < m ? limit : m));
    }
    verify(&p, &r, target, dual, 1);
    result_free(&r);
    problem_free(&p);
}
static void fill_limits(void) {
    Problem p = {0};
    Result r = {0};
    COO q = {0}, a = {0};
    double target[6] = {0}, dual[2] = {0};
    CHECK(problem_alloc(&p, 2, 6) == OK);
    for (int j = 0; j < 6; ++j)
        coo_add(&q, j, j, 1.);
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 3; ++j)
            coo_add(&a, i, 3 * i + j, 1.);
        p.row_lower[i] = p.row_upper[i] = 0.;
    }
    matrix(&p.Q, 6, 6, &q);
    matrix(&p.A, 2, 6, &a);
    Config cfg = default_config();
    cfg.backend = backend;
    cfg.rules = SINGLETON_EQ;
    cfg.max_q_fill = 1;
    cfg.max_q_fill_ratio = INFINITY;
    for (int mode = 0; mode < 3; ++mode) {
        cfg.max_q_fill = mode == 2 ? 0 : 1;
        cfg.max_q_fill_ratio = mode == 1 ? 1.2 : INFINITY;
        CHECK(presolve(&p, &cfg, &r) == OK);
        CHECK(r.reduced.A.rows == (mode == 2 ? 2 : 0));
        CHECK(r.reduced.A.cols == (mode == 2 ? 6 : 4));
        ExecutionInfo info;
        TransferInfo traffic;
        BatchInfo batch;
        CHECK(execution_info(r.state, &info) == OK);
        CHECK(transfer_info(r.state, &traffic) == OK);
        CHECK(batch_info(r.state, &batch) == OK);
        if (backend == CUDA) {
            CHECK(info.cpu_fallbacks == 0 && info.gpu_reductions == (mode == 2 ? 0u : 2u));
            CHECK(traffic.model_uploads == 1);
            CHECK(info.gpu_batches == (mode == 0 ? 1u : mode == 1 ? 2u : 0u));
            CHECK(batch.retries == (mode == 1 ? 1u : 0u));
            CHECK(batch.fill_rejections == batch.retries);
            CHECK(batch.cpu_fallback_phases == 0);
        } else
            CHECK(batch.retries == 0 && traffic.model_uploads == 0);
        verify(&p, &r, target, dual, 1);
        result_free(&r);
    }
    problem_free(&p);
}
static void individual_fill_guard(void) {
    Problem p = {0};
    Result r = {0};
    COO a = {0}, q = {0};
    double target[7] = {0}, dual[2] = {0};
    CHECK(problem_alloc(&p, 2, 7) == OK);
    for (int j = 0; j < 4; ++j) {
        coo_add(&a, 0, j, 1.);
        coo_add(&q, j, j, 1.);
    }
    for (int i = 4; i < 7; ++i) {
        coo_add(&a, 1, i, 1.);
        for (int j = 4; j < 7; ++j)
            coo_add(&q, i, j, 1.);
    }
    p.row_lower[0] = p.row_upper[0] = 0.;
    p.row_lower[1] = p.row_upper[1] = 0.;
    matrix(&p.A, 2, 7, &a);
    matrix(&p.Q, 7, 7, &q);
    Config cfg = default_config();
    cfg.backend = backend;
    cfg.rules = SINGLETON_EQ;
    cfg.max_q_fill = 1;
    cfg.max_q_fill_ratio = INFINITY;
    CHECK(presolve(&p, &cfg, &r) == OK);
    /* The dense rank-one block disappears, but its negative fill must not
     * hide the other pivot's excessive growth of five nonzeros. */
    CHECK(r.reduced.A.rows == 1 && r.reduced.A.cols == 6 && r.reduced.Q.nnz == 4);
    verify(&p, &r, target, dual, 1);
    result_free(&r);
    problem_free(&p);
}
static void feasible_bound_cycle(void) {
    Problem p = {0};
    COO a = {0};
    double A[8][6] = {
        {-2, -2, 0, 0, 0, -1}, {0, 0, 0, 1, 1, 0},   {0, -1, 0, 2, 0, -1}, {1, 0, -2, 1, 0, -1},
        {1, 0, -1, 0, 0, 0},   {1, -2, 0, -2, 0, 0}, {2, 0, -1, 0, 0, 2},  {0}};
    double lower[] = {4.5, 2, 1.5, -4.5, -4, -4, -8, 0},
           upper[] = {5.5, 4, 2.5, -4.5, -4, -4, -7, 0};
    double c[] = {-3.0923248480066494, -5.331040728856987,  2.871502960155669,
                  -7.041850402832292,  -1.5242990417403262, -0.9675375902191341};
    double target[] = {-2, 0, 2, 1, 2, -.5};
    CHECK(problem_alloc(&p, 8, 6) == OK);
    for (int i = 0; i < 8; ++i)
        for (int j = 0; j < 6; ++j)
            if (A[i][j])
                coo_add(&a, i, j, A[i][j]);
    matrix(&p.A, 8, 6, &a);
    for (int j = 0; j < 6; ++j) {
        p.lower[j] = -2.;
        p.upper[j] = 2.;
        p.c[j] = c[j];
    }
    memcpy(p.row_lower, lower, sizeof(lower));
    memcpy(p.row_upper, upper, sizeof(upper));
    Config cfg = default_config();
    cfg.backend = backend;
    cfg.bound_tol = cfg.feasibility_tol = 1e-10;
    cfg.max_iterations = 20;
    cfg.max_q_fill = -1;
    cfg.max_q_fill_ratio = INFINITY;
    cfg.monotone_empty_columns = 1;
    for (int schedule = 0; schedule < 2; ++schedule) {
        Result r = {0};
        ExecutionOptions o = default_execution_options();
        o.schedule = (Schedule)schedule;
        CHECK(presolve_with_options(&p, &cfg, &o, &r) == OK);
        if (r.state) {
            double xr[12] = {0}, yr[12] = {0}, zr[12] = {0}, x[6], y[8], z[6];
            CHECK(test_solve(&r.reduced, xr, yr, zr));
            CHECK(postsolve(r.state, xr, r.reduced.A.cols, yr, r.reduced.A.rows, zr,
                                r.reduced.A.cols, x, y, z) == OK);
            KKT k;
            CHECK(compute_kkt(&p, x, y, z, &k) == OK);
            CHECK(k.primal_feasibility < 1e-7 && k.dual_feasibility < 1e-7 &&
                  k.relative_gap < 1e-7);
            CHECK(fabs(objective(&p, x) - objective(&p, target)) < 1e-6);
        }
        result_free(&r);
    }
    problem_free(&p);
    CHECK(problem_alloc(&p, 0, 1) == OK);
    p.lower[0] = 1.;
    p.upper[0] = nextafter(1., INFINITY);
    cfg.rules = FIXED_COLS;
    Result r = {0};
    CHECK(presolve(&p, &cfg, &r) == OK);
    CHECK(r.reduced.A.cols == 1 && r.reduced.lower[0] != r.reduced.upper[0]);
    result_free(&r);
    problem_free(&p);
}
static void unchanged_bound_side(void) {
    for (int propagation = 0; propagation < 2; ++propagation)
        for (int sign = -1; sign <= 1; sign += 2) {
            Problem p = {0};
            Result r = {0};
            COO q = {0}, a = {0};
            CHECK(problem_alloc(&p, 1, 1) == OK);
            coo_add(&q, 0, 0, 1.);
            coo_add(&a, 0, 0, 1.);
            matrix(&p.Q, 1, 1, &q);
            matrix(&p.A, 1, 1, &a);
            p.lower[0] = -2.;
            p.upper[0] = 2.;
            p.c[0] = -3. * sign;
            if (sign > 0)
                p.row_lower[0] = 1.;
            else
                p.row_upper[0] = -1.;
            Config cfg = default_config();
            cfg.backend = backend;
            cfg.bound_tol = 1e-10;
            cfg.rules = propagation ? PROPAGATION : SINGLETON_ROWS;
            CHECK(presolve(&p, &cfg, &r) == OK);
            double xr[] = {sign * (2. - 1e-9)}, yr[] = {0.}, zr[] = {xr[0] + p.c[0]}, x[1], y[1],
                   z[1];
            CHECK(postsolve(r.state, xr, 1, yr, r.reduced.A.rows, zr, 1, x, y, z) == OK);
            CHECK(x[0] == xr[0] && y[0] == 0. && z[0] == zr[0]);
            KKT k;
            CHECK(compute_kkt(&p, x, y, z, &k) == OK);
            CHECK(k.primal_feasibility < 1e-8 && k.dual_feasibility < 1e-8 &&
                  k.relative_gap < 1e-8);
            result_free(&r);
            problem_free(&p);
        }
}
static void subnormal_parallel(void) {
    Problem p = {0};
    Result r = {0};
    COO q = {0}, a = {0};
    double unit = nextafter(0., 1.);
    CHECK(problem_alloc(&p, 1, 2) == OK);
    /* [[5,7],[7,10]] has positive determinant 1. Raw unscaled FMA
     * residuals round to zero at this scale and must not erase its curvature. */
    coo_add(&q, 0, 0, 5 * unit);
    coo_add(&q, 0, 1, 7 * unit);
    coo_add(&q, 1, 0, 7 * unit);
    coo_add(&q, 1, 1, 10 * unit);
    coo_add(&a, 0, 0, 5 * unit);
    coo_add(&a, 0, 1, 7 * unit);
    matrix(&p.Q, 2, 2, &q);
    matrix(&p.A, 1, 2, &a);
    p.lower[0] = p.lower[1] = -1e160;
    p.upper[0] = p.upper[1] = 1e160;
    p.row_lower[0] = p.row_upper[0] = 0.;
    Config c = default_config();
    c.backend = backend;
    c.rules = PARALLEL_COLS;
    CHECK(presolve(&p, &c, &r) == OK);
    CHECK(r.reduced.A.cols == 2 && r.reduced.Q.nnz == 4);
    CHECK(!coefficient_exact_scaled(7 * unit, 10 * unit, 1.4));
    result_free(&r);
    problem_free(&p);
}
static void invalid_options(void) {
    Problem p = {0};
    Result r = {0};
    Config c = default_config();
    c.backend = backend;
    CHECK(problem_alloc(&p, 0, 0) == OK);
    for (int k = 0; k < 4; ++k) {
        ExecutionOptions o = default_execution_options();
        if (k == 0)
            o.schedule = (Schedule)2;
        if (k == 1)
            o.batch_limit = 0;
        if (k == 2)
            o.cleanup_rounds = 0;
        if (k == 3)
            o.global_period = 0;
        CHECK(presolve_with_options(&p, &c, &o, &r) == INVALID_ARGUMENT);
        CHECK(!r.state);
        result_free(&r);
    }
    problem_free(&p);
}
static void resident_bounds(void) {
    enum { n = 64, m = n - 1 };
    Problem p = {0};
    Result r = {0};
    COO a = {0}, q = {0};
    double target[n] = {0}, dual[m] = {0};
    CHECK(problem_alloc(&p, m, n) == OK);
    for (int j = 0; j < n; ++j) {
        p.lower[j] = -1.;
        p.upper[j] = j == n - 1 ? 0. : 1.;
        coo_add(&q, j, j, 2.);
        if (j < m) {
            coo_add(&a, j, j, 1.);
            coo_add(&a, j, j + 1, -1.);
            p.row_upper[j] = 0.;
        }
    }
    matrix(&p.Q, n, n, &q);
    matrix(&p.A, m, n, &a);
    Config c = default_config();
    c.backend = backend;
    c.rules = PROPAGATION;
    c.max_iterations = 100;
    CHECK(presolve(&p, &c, &r) == OK);
    for (int j = 0; j < n; ++j)
        CHECK(r.reduced.upper[j] == 0.);
    TransferInfo info;
    CHECK(transfer_info(r.state, &info) == OK);
    CHECK(transfer_info(NULL, &info) == INVALID_ARGUMENT);
    CHECK(transfer_info(r.state, NULL) == INVALID_ARGUMENT);
    BatchInfo batches;
    CHECK(batch_info(r.state, &batches) == OK);
    CHECK(batch_info(NULL, &batches) == INVALID_ARGUMENT);
    CHECK(batch_info(r.state, NULL) == INVALID_ARGUMENT);
    if (backend == CUDA) {
        uint64_t initial = (uint64_t)(n + 1 + m + 1) * sizeof(int) +
                           (uint64_t)(p.Q.nnz + p.A.nnz) * (sizeof(int) + sizeof(double)) +
                           (uint64_t)(3 * n + 2 * m) * sizeof(double);
        CHECK(info.model_uploads == 1 && info.resident_commits == 0);
        CHECK(info.bound_updates == m && info.downloaded_candidates == 0);
        CHECK(info.host_to_device_bytes == initial + sizeof(double));
        CHECK(info.device_candidates >= n && info.device_candidates < 8 * (uint64_t)n);
    } else
        CHECK(!info.host_to_device_bytes && !info.model_uploads);
    verify(&p, &r, target, dual, 1);
    result_free(&r);
    problem_free(&p);
}
static void propagation_deferred(void) {
    enum { n = 128, m = n - 1 };
    Problem p = {0};
    Result r = {0};
    COO a = {0}, q = {0};
    double target[n] = {0}, dual[m] = {0};
    CHECK(problem_alloc(&p, m, n) == OK);
    for (int j = 0; j < n; ++j) {
        p.lower[j] = -1.;
        p.upper[j] = j == n - 1 ? 0. : 1.;
        coo_add(&q, j, j, 2.);
        if (j < m) {
            coo_add(&a, j, j, 1.);
            coo_add(&a, j, n - 1, -1.);
            p.row_upper[j] = 0.;
        }
    }
    matrix(&p.Q, n, n, &q);
    matrix(&p.A, m, n, &a);
    Config cfg = default_config();
    cfg.backend = backend;
    cfg.rules = PROPAGATION;
    ExecutionOptions options = default_execution_options();
    options.batch_limit = 1;
    CHECK(presolve_with_options(&p, &cfg, &options, &r) == OK);
    for (int j = 0; j < n; ++j)
        CHECK(r.reduced.upper[j] == 0.);
    verify(&p, &r, target, dual, 1);
    result_free(&r);
    problem_free(&p);
}
static void propagation_dual_invalidation(void) {
    if (backend != CUDA)
        return;
    Problem p = {0};
    COO a = {0};
    CHECK(problem_alloc(&p, 1, 2) == OK);
    coo_add(&a, 0, 0, 1.);
    coo_add(&a, 0, 1, 1.);
    matrix(&p.A, 1, 2, &a);
    p.lower[0] = -10.;
    p.upper[0] = 10.;
    p.lower[1] = 0.;
    p.upper[1] = 1.;
    p.row_lower[0] = 0.;
    p.row_upper[0] = 3.;
    p.c[0] = 1.;
    State *state = state_create(0, 0);
    CHECK(state != NULL);
    state->original_m = state->reduced_m = 1;
    state->original_n = state->reduced_n = 2;
    state->backend = CUDA;
    state->tolerance = 1e-6;
    void *plan = NULL;
    Problem shape = {0};
    CHECK(cuda_resident_create(&p, state, &plan, &shape) == OK);
    Config cfg = default_config();
    cfg.bound_tol = 100.;
    CHECK(cuda_resident_apply(state, &cfg, PROPAGATION, 65536, &plan, INFINITY) == 0);
    CHECK(cuda_resident_apply(state, &cfg, SINGLETON_DUAL, 65536, &plan, INFINITY) == 1);
    cfg.bound_tol = 1e-10;
    CHECK(cuda_resident_apply(state, &cfg, PROPAGATION, 65536, &plan, INFINITY) == 1);
    cuda_plan_free(&plan);
    state_free(&state);
    problem_free(&p);
}
static void doubleton_chain(int retained) {
    enum { n = 17, m = n - 1 };
    Problem p = {0};
    Result r = {0};
    COO a = {0}, q = {0};
    double target[n], dual[m] = {0};
    CHECK(problem_alloc(&p, m, n) == OK);
    target[n - 1] = .75;
    for (int j = m - 1; j >= 0; --j) {
        double beta = ldexp(j % 2 ? -1. : 1., j % 3 - 1);
        double alpha = .25 * (j % 3 - 1);
        target[j] = alpha + beta * target[j + 1];
        coo_add(&a, j, j, 1.);
        coo_add(&a, j, j + 1, -beta);
        p.row_lower[j] = p.row_upper[j] = alpha;
    }
    if (retained)
        for (int j = 0; j < n; ++j) {
            p.lower[j] = target[j] - 10.;
            p.upper[j] = target[j] + 10.;
        }
    coo_add(&q, n - 1, n - 1, 2.);
    p.c[n - 1] = -1.5;
    p.objective_offset = 1.25;
    matrix(&p.Q, n, n, &q);
    matrix(&p.A, m, n, &a);
    Config cfg = default_config();
    cfg.backend = backend;
    cfg.rules = DOUBLETON_EQ;
    ExecutionOptions options = default_execution_options();
    options.batch_limit = 1;
    CHECK(presolve_with_options(&p, &cfg, &options, &r) == OK);
    CHECK(r.reduced.A.cols == 1 && r.reduced.A.rows == (retained ? m : 0));
    verify(&p, &r, target, dual, 1);
    result_free(&r);
    problem_free(&p);
}
static void doubleton_reverse_pivot(int retained) {
    Problem p = {0};
    Result r = {0};
    COO a = {0}, q = {0};
    double target[] = {.75, 2., -1.}, dual[] = {0., 0.};
    CHECK(problem_alloc(&p, 2, 3) == OK);
    coo_add(&q, 0, 0, 2.);
    coo_add(&q, 0, 2, 1.);
    coo_add(&q, 2, 0, 1.);
    coo_add(&q, 2, 2, 2.);
    p.c[0] = -.5;
    p.c[2] = 1.25;
    coo_add(&a, 0, 0, 2.);
    coo_add(&a, 0, 1, -1.);
    coo_add(&a, 1, 1, 1.);
    coo_add(&a, 1, 2, 1.);
    p.row_lower[0] = p.row_upper[0] = -.5;
    p.row_upper[1] = 2.;
    if (retained) {
        p.lower[1] = 0.;
        p.upper[1] = 4.;
    }
    matrix(&p.Q, 3, 3, &q);
    matrix(&p.A, 2, 3, &a);
    Config cfg = default_config();
    cfg.backend = backend;
    cfg.rules = DOUBLETON_EQ;
    cfg.max_q_fill = 0;
    Result *result = &r;
    CHECK(presolve(&p, &cfg, result) == OK);
    CHECK(r.reduced.A.cols == 2);
    CHECK(state_export_host(r.state) == OK);
    if (backend == CUDA)
        CHECK(r.state->records[0].j == 1);
    verify(&p, &r, target, dual, 1);
    result_free(&r);
    problem_free(&p);
}
static void doubleton_delta(void) {
    if (backend != CUDA)
        return;
    enum { n = 4096, m = n };
    Problem p = {0};
    Result r = {0};
    COO a = {0}, q = {0};
    double *target = checked_calloc(n, sizeof(double)), *dual = checked_calloc(m, sizeof(double));
    CHECK(target && dual);
    CHECK(problem_alloc(&p, m, n) == OK);
    for (int j = 0; j < n; ++j) {
        target[j] = j ? .5 : 1.5;
        if (j) {
            coo_add(&q, j, j, 2.);
            p.c[j] = -1.;
        }
    }
    p.lower[0] = 0.;
    p.upper[0] = 3.;
    coo_add(&a, 0, 0, 1.);
    coo_add(&a, 0, 1, -1.);
    p.row_lower[0] = p.row_upper[0] = 1.;
    coo_add(&a, 1, 0, 1.);
    coo_add(&a, 1, 2, 2.);
    p.row_upper[1] = 4.;
    for (int i = 2; i < m; ++i) {
        coo_add(&a, i, i, 1.);
        p.row_upper[i] = 1.5;
    }
    matrix(&p.Q, n, n, &q);
    matrix(&p.A, m, n, &a);
    Config cfg = default_config();
    cfg.backend = backend;
    cfg.rules = DOUBLETON_EQ;
    CHECK(presolve(&p, &cfg, &r) == OK);
    CHECK(r.reduced.A.cols == n - 1 && r.reduced.A.rows == m);
    TransferInfo traffic;
    CHECK(transfer_info(r.state, &traffic) == OK);
    ResidencyInfo resident;
    CHECK(residency_info(r.state, &resident) == OK && resident.resident);
    CHECK(traffic.model_uploads == 1 && resident.iterations.device_to_host_bytes < 4096);
    CHECK(resident.iterations.downloaded_candidates == 0 && resident.tape_export_bytes == 0);
    verify(&p, &r, target, dual, 1);
    result_free(&r);
    problem_free(&p);
    free(target);
    free(dual);
}
static void doubleton_cancellation(int retained, int perturbed) {
    Problem p = {0};
    Result r = {0};
    COO a = {0}, q = {0};
    double target[] = {.5, .5, .75}, dual[] = {0., 0.};
    CHECK(problem_alloc(&p, 2, 3) == OK);
    coo_add(&q, 1, 1, 2.);
    coo_add(&q, 2, 2, 2.);
    p.c[1] = -1.;
    p.c[2] = -1.5;
    coo_add(&a, 0, 0, 1.);
    coo_add(&a, 0, 1, -1.);
    coo_add(&a, 1, 0, 1.);
    coo_add(&a, 1, 1, perturbed ? -1. - 1e-12 : -1.);
    p.row_lower[0] = p.row_upper[0] = 0.;
    p.row_upper[1] = 1.;
    if (retained) {
        p.lower[0] = -1.;
        p.upper[0] = 2.;
    }
    matrix(&p.Q, 3, 3, &q);
    matrix(&p.A, 2, 3, &a);
    Config cfg = default_config();
    cfg.backend = backend;
    cfg.rules = DOUBLETON_EQ;
    CHECK(presolve(&p, &cfg, &r) == OK);
    CHECK(r.reduced.A.nnz == retained + perturbed);
    if (perturbed)
        CHECK(matrix_get(&r.reduced.A, retained, 0) != 0.);
    verify(&p, &r, target, dual, 1);
    result_free(&r);
    problem_free(&p);
}
static void staged_propagation_budget(void) {
    if (backend != CUDA)
        return;
    enum { n = 8192, m = n - 1 };
    Problem p = {0};
    Result r = {0};
    COO a = {0};
    CHECK(problem_alloc(&p, m, n) == OK);
    for (int j = 0; j < n; ++j) {
        p.lower[j] = -1.;
        p.upper[j] = j == n - 1 ? 0. : 1.;
        if (j < m) {
            coo_add(&a, j, j, 1.);
            coo_add(&a, j, j + 1, -1.);
            p.row_upper[j] = 0.;
        }
    }
    matrix(&p.A, m, n, &a);
    Config cfg = default_config();
    cfg.backend = backend;
    cfg.rules = PROPAGATION;
    cfg.max_iterations = 2;
    ExecutionOptions options = default_execution_options();
    options.schedule = SCHEDULE_TIERED;
    CHECK(presolve_with_options(&p, &cfg, &options, &r) == OK);
    RuleInfo info;
    CHECK(rule_info(r.state, PROPAGATION, &info) == OK);
    CHECK(info.calls == 64 && info.reductions == 64);
    CHECK(r.reduced.upper[n - 65] == 0. && r.reduced.upper[n - 66] == 1.);
    double *target = checked_calloc(n, sizeof(double)), *dual = checked_calloc(m, sizeof(double));
    CHECK(target && dual);
    verify(&p, &r, target, dual, 1);
    free(target);
    free(dual);
    result_free(&r);
    problem_free(&p);
}
static void grouped_bounds(int columns) {
    const int size = 300, m = columns ? 1 : size, n = columns ? size : 2;
    Problem p = {0};
    Result reference = {0}, r = {0};
    COO a = {0}, q = {0};
    CHECK(problem_alloc(&p, m, n) == OK);
    for (int t = 0; t < size; ++t) {
        double scale = ldexp(t % 2 ? -1. : 1., t % 3 - 1);
        if (columns) {
            coo_add(&a, 0, t, scale);
            p.lower[t] = -1. - .125 * t;
            p.upper[t] = 2. + .25 * t;
            p.c[t] = .75 * scale;
        } else {
            coo_add(&a, t, 0, scale);
            coo_add(&a, t, 1, -scale);
            double lo = -1. / (t + 1), hi = 2. / (t + 1);
            p.row_lower[t] = scale > 0 ? scale * lo : scale * hi;
            p.row_upper[t] = scale > 0 ? scale * hi : scale * lo;
        }
    }
    if (!columns) {
        coo_add(&q, 0, 0, 2.);
        coo_add(&q, 1, 1, 2.);
        p.c[0] = 1.;
        p.c[1] = -1.;
    }
    matrix(&p.A, m, n, &a);
    matrix(&p.Q, n, n, &q);
    Config c = default_config();
    c.backend = CUDA;
    c.rules = columns ? PARALLEL_COLS : PARALLEL_ROWS;
    /* Check batched bounds against one-reduction-at-a-time CUDA execution. */
    ExecutionOptions serial = default_execution_options();
    serial.batch_limit = 1;
    CHECK(presolve_with_options(&p, &c, &serial, &reference) == OK);
    ExecutionOptions o = default_execution_options();
    o.batch_limit = 129;
    CHECK(presolve_with_options(&p, &c, &o, &r) == OK);
    CHECK(r.reduced.A.rows == reference.reduced.A.rows);
    CHECK(r.reduced.A.cols == reference.reduced.A.cols);
    for (int i = 0; i < r.reduced.A.rows; ++i) {
        CHECK(r.reduced.row_lower[i] == reference.reduced.row_lower[i]);
        CHECK(r.reduced.row_upper[i] == reference.reduced.row_upper[i]);
    }
    for (int j = 0; j < r.reduced.A.cols; ++j) {
        CHECK(r.reduced.lower[j] == reference.reduced.lower[j]);
        CHECK(r.reduced.upper[j] == reference.reduced.upper[j]);
    }
    double *xr = checked_calloc(r.reduced.A.cols, sizeof(double));
    double *yr = checked_calloc(r.reduced.A.rows, sizeof(double));
    double *zr = checked_calloc(r.reduced.A.cols, sizeof(double));
    double *x = checked_calloc(n, sizeof(double)), *y = checked_calloc(m, sizeof(double));
    double *z = checked_calloc(n, sizeof(double));
    if (columns) {
        xr[0] = r.reduced.lower[0];
        zr[0] = r.reduced.c[0];
    } else {
        xr[0] = -.5 / size;
        xr[1] = .5 / size;
        yr[0] = 2. * (1. - 1. / size);
    }
    CHECK(postsolve(r.state, xr, r.reduced.A.cols, yr, r.reduced.A.rows, zr, r.reduced.A.cols,
                        x, y, z) == OK);
    KKT k;
    CHECK(compute_kkt(&p, x, y, z, &k) == OK);
    CHECK(k.primal_feasibility < 1e-9 && k.dual_feasibility < 1e-9 && k.relative_gap < 1e-9);
    free(xr);
    free(yr);
    free(zr);
    free(x);
    free(y);
    free(z);
    result_free(&reference);
    result_free(&r);
    problem_free(&p);
}
static void overflowing_group(void) {
    Problem p = {0};
    Result r = {0};
    COO a = {0};
    CHECK(problem_alloc(&p, 1, 130) == OK);
    for (int j = 0; j < 130; ++j) {
        p.lower[j] = 1e308;
        coo_add(&a, 0, j, 1.);
    }
    matrix(&p.A, 1, 130, &a);
    Config c = default_config();
    c.backend = backend;
    c.rules = PARALLEL_COLS;
    /* Bound-addition overflow must report numerical failure, never invent an
     * infeasibility certificate from an overflowing +infinity intermediate. */
    CHECK(presolve(&p, &c, &r) == NUMERICAL_ERROR);
    result_free(&r);
    problem_free(&p);
}
static void singleton_group(void) {
    enum { m = 300 };
    Problem p = {0};
    Result r = {0};
    COO a = {0}, q = {0};
    CHECK(problem_alloc(&p, m, 1) == OK);
    coo_add(&q, 0, 0, 2.);
    matrix(&p.Q, 1, 1, &q);
    for (int i = 0; i < m; ++i) {
        double scale = ldexp(i % 2 ? -1. : 1., i % 3 - 1);
        coo_add(&a, i, 0, scale);
        double lo = (double)i / m, hi = 2.;
        p.row_lower[i] = scale > 0 ? scale * lo : scale * hi;
        p.row_upper[i] = scale > 0 ? scale * hi : scale * lo;
    }
    matrix(&p.A, m, 1, &a);
    Config c = default_config();
    c.backend = backend;
    c.rules = SINGLETON_ROWS;
    ExecutionOptions o = default_execution_options();
    o.batch_limit = 129;
    CHECK(presolve_with_options(&p, &c, &o, &r) == OK);
    CHECK(r.reduced.A.rows == 0 && r.reduced.A.cols == 1);
    CHECK(r.reduced.lower[0] == nextafter((double)(m - 1) / m, -INFINITY) &&
          r.reduced.upper[0] == nextafter(2., INFINITY));
    double xr = r.reduced.lower[0], zr = 2 * xr, x, y[m], z;
    CHECK(postsolve(r.state, &xr, 1, NULL, 0, &zr, 1, &x, y, &z) == OK);
    KKT k;
    CHECK(compute_kkt(&p, &x, y, &z, &k) == OK);
    CHECK(k.primal_feasibility < 1e-9 && k.dual_feasibility < 1e-9 && k.relative_gap < 1e-9);
    result_free(&r);
    problem_free(&p);
}
static void long_row_bound_recovery(void) {
    enum { n = 257 };
    Problem p = {0}; Result r = {0}; COO a = {0}, q = {0};
    CHECK(problem_alloc(&p, 1, n) == OK);
    for (int j = 0; j < n; ++j) {
        coo_add(&a, 0, j, j % 2 ? -1. : 1.);
        p.lower[j] = 0.; p.upper[j] = j ? 0. : 10.;
    }
    matrix(&p.A, 1, n, &a); coo_add(&q, 0, 0, 2.); matrix(&p.Q, n, n, &q);
    p.row_lower[0] = 2.;
    Config cfg = default_config(); cfg.backend = backend; cfg.rules = PROPAGATION;
    CHECK(presolve(&p, &cfg, &r) == OK);
    CHECK(r.reduced.lower[0] > 1.99 && r.reduced.lower[0] <= 2.);
    double xr[n] = {2.}, zr[n] = {4.}, x[n], y[1], z[n], yr = 0.;
    CHECK(postsolve(r.state, xr, n, &yr, 1, zr, n, x, y, z) == OK);
    CHECK(fabs(y[0] - 4.) < 1e-9 && fabs(z[0]) < 1e-9);
    KKT k;
    CHECK(compute_kkt(&p, x, y, z, &k) == OK);
    CHECK(k.primal_feasibility < 1e-9 && k.dual_feasibility < 1e-9 && k.relative_gap < 1e-9);
    result_free(&r); problem_free(&p);
}
static void long_row_cache_refresh(void) {
    enum { n = 260 };
    Problem p = {0}; Result r = {0}; COO a = {0}, q = {0};
    CHECK(problem_alloc(&p, 2, n) == OK);
    for (int j = 0; j < n; ++j) {
        p.lower[j] = j < 3 ? -1. : 0.;
        p.upper[j] = j < 2 ? 1. : 0.;
        coo_add(&q, j, j, 2.);
        if (j >= 3) { coo_add(&a, 0, j, 1.); coo_add(&a, 1, j, -1.); }
    }
    coo_add(&a, 0, 0, 1.); coo_add(&a, 0, 1, -1.);
    coo_add(&a, 1, 1, 1.); coo_add(&a, 1, 2, -1.);
    matrix(&p.A, 2, n, &a); matrix(&p.Q, n, n, &q);
    p.row_upper[0] = p.row_upper[1] = 0.;
    Config cfg = default_config(); cfg.backend = backend; cfg.rules = PROPAGATION;
    ExecutionOptions options = default_execution_options(); options.batch_limit = 1;
    CHECK(presolve_with_options(&p, &cfg, &options, &r) == OK);
    /* x1 <= x2 must be published before a new snapshot can derive x0 <= x1. */
    CHECK(r.reduced.upper[0] == 0. && r.reduced.upper[1] == 0.);
    double target[n] = {0}, dual[2] = {0};
    verify(&p, &r, target, dual, 1);
    result_free(&r); problem_free(&p);
}
static void warp_propagation_order(void) {
    Problem p = {0}; Result r = {0}; COO a = {0}, q = {0};
    CHECK(problem_alloc(&p, 97, 2) == OK);
    for (int i = 0; i < 97; ++i) { coo_add(&a, i, 0, 1.); coo_add(&a, i, 1, 1.); }
    matrix(&p.A, 97, 2, &a); coo_add(&q, 0, 0, 2.); matrix(&p.Q, 2, 2, &q);
    p.lower[0] = -100.; p.upper[0] = 100.; p.lower[1] = p.upper[1] = 0.; p.c[0] = -8.;
    const int source[] = {31,32,64,96};
    for (int k = 0; k < 4; ++k) p.row_lower[source[k]] = k + 1.;
    Config cfg = default_config(); cfg.backend = backend; cfg.rules = PROPAGATION;
    CHECK(presolve(&p, &cfg, &r) == OK);
    CHECK(r.reduced.lower[0] > 3.99 && r.reduced.lower[0] <= 4.);
    CHECK(state_export_host(r.state) == OK && r.state->count == 4);
    for (size_t k = 0; k < r.state->count && k < 4; ++k)
        CHECK(r.state->records[k].row == source[k] && r.state->records[k].j == 0);
    double target[] = {4.,0.}, dual[97] = {0};
    verify(&p, &r, target, dual, 1);
    result_free(&r); problem_free(&p);
}
static void row_objective_preserved(void) {
    Problem p = {0};
    COO a = {0}, q = {0};
    CHECK(problem_alloc(&p, 6, 2) == OK);
    coo_add(&a, 1, 0, 1.);
    for (int i = 2; i < 6; ++i) {
        coo_add(&a, i, 0, i == 4 ? -2. : 1.);
        coo_add(&a, i, 1, i == 4 ? -2. : i == 5 ? -1. : 1.);
    }
    matrix(&p.A, 6, 2, &a);
    coo_add(&q, 0, 0, 2.); coo_add(&q, 0, 1, .1);
    coo_add(&q, 1, 0, .1); coo_add(&q, 1, 1, 3.);
    matrix(&p.Q, 2, 2, &q);
    p.c[0] = 1e200; p.c[1] = -0.; p.objective_offset = .125;
    for (int j = 0; j < 2; ++j) { p.lower[j] = -10.; p.upper[j] = 10.; }
    p.row_lower[1] = 0.; p.row_upper[1] = 2.;
    p.row_lower[2] = -100.; p.row_upper[2] = 100.;
    p.row_upper[3] = 1.; p.row_lower[4] = -2.; p.row_lower[5] = -3.;
    const uint32_t rules[] = {EMPTY_ROWS, SINGLETON_ROWS, ACTIVITY, PARALLEL_ROWS,
                             EMPTY_ROWS | SINGLETON_ROWS | ACTIVITY | PARALLEL_ROWS};
    for (size_t k = 0; k < sizeof(rules) / sizeof(*rules); ++k)
        for (int single = 0; single < 2; ++single) {
            Config cfg = default_config();
            cfg.backend = backend; cfg.rules = rules[k];
            ExecutionOptions options = default_execution_options();
            options.batch_limit = single ? 1 : 256;
            Result r = {0};
            CHECK(presolve_with_options(&p, &cfg, &options, &r) == OK);
            CHECK(r.reductions > 0 && r.reduced.A.cols == 2);
            CHECK(validate(&r.reduced, NULL, 0) == OK);
            /* Row reductions must not reconstruct/round the quadratic objective. */
            CHECK(r.reduced.Q.nnz == p.Q.nnz);
            CHECK(!memcmp(r.reduced.Q.row_ptr, p.Q.row_ptr, 3 * sizeof(int)));
            CHECK(!memcmp(r.reduced.Q.col_ind, p.Q.col_ind, 4 * sizeof(int)));
            CHECK(!memcmp(r.reduced.Q.values, p.Q.values, 4 * sizeof(double)));
            CHECK(!memcmp(r.reduced.c, p.c, 2 * sizeof(double)));
            CHECK(!memcmp(&r.reduced.objective_offset, &p.objective_offset, sizeof(double)));
            result_free(&r);
        }
    problem_free(&p);
}
int main(int argc, char **argv) {
    if (argc != 2 || strcmp(argv[1], "--gpu"))
        return 2;
    backend = CUDA;
    resident_bounds();
    propagation_deferred();
    propagation_dual_invalidation();
    doubleton_chain(0);
    doubleton_chain(1);
    doubleton_reverse_pivot(0);
    doubleton_reverse_pivot(1);
    doubleton_delta();
    doubleton_cancellation(0, 0);
    doubleton_cancellation(1, 0);
    doubleton_cancellation(0, 1);
    doubleton_cancellation(1, 1);
    staged_propagation_budget();
    grouped_bounds(0);
    grouped_bounds(1);
    overflowing_group();
    singleton_group();
    row_objective_preserved();
    long_row_bound_recovery();
    long_row_cache_refresh();
    warp_propagation_order();
    feasible_bound_cycle();
    derived_bound_roundoff();
    unchanged_bound_side();
    subnormal_parallel();
    invalid_options();
    coupled_fixed();
    substitutions(0, 0, 65536, 0);
    substitutions(1, 0, 65536, 0);
    substitutions(1, 0, 7, 1);
    substitutions(0, 1, 65536, 0);
    fill_limits();
    individual_fill_guard();
    printf("batch backend=%d checks=%d failures=%d\n", backend, checks, failures);
    return failures != 0;
}
