#include "internal.h"
extern "C" {
#include "support/qp_oracle.h"
}
#include <algorithm>
#include <cuda_runtime.h>
#include <vector>

static int checks, failures;
#define CHECK(e)                                                                                   \
    do {                                                                                           \
        ++checks;                                                                                  \
        if (!(e)) {                                                                                \
            ++failures;                                                                            \
            fprintf(stderr, "folding FAIL %d: %s\n", __LINE__, #e);                                \
        }                                                                                          \
    } while (0)
#define CU(e)                                                                                      \
    do {                                                                                           \
        cudaError_t code = (e);                                                                    \
        if (code != cudaSuccess) {                                                                 \
            fprintf(stderr, "CUDA %d: %s\n", __LINE__, cudaGetErrorString(code));                  \
            exit(2);                                                                               \
        }                                                                                          \
    } while (0)
static bool near(double a, double b, double tol = 1e-9) {
    return a == b || fabs(a - b) <= tol * (1. + fmax(fabs(a), fabs(b)));
}
static void dense(Matrix *a, int m, int n, const std::vector<double> &v) {
    COO coo = {};
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j)
            coo_add(&coo, i, j, v[i * n + j]);
    CHECK(coo_finish(&coo, m, n, a) == OK);
}
static Config config(uint32_t rules = FOLDING) {
    Config c = default_config();
    c.backend = CUDA;
    c.rules = rules;
    return c;
}
static void resident(const Problem &p, Result &r, bool device_input = false) {
    ResidencyInfo info = {};
    CHECK(residency_info(r.state, &info) == OK);
    CHECK(info.resident && !info.model_export_bytes && !info.tape_export_bytes);
    CHECK(!r.state->records && !r.state->row_ids && !r.state->col_ids);
    CHECK(!info.iterations.downloaded_candidates);
    uint64_t initial = ((uint64_t)p.A.rows + p.A.cols + 2) * sizeof(int) +
                       ((uint64_t)p.A.nnz + p.Q.nnz) * (sizeof(int) + sizeof(double)) +
                       ((uint64_t)3 * p.A.cols + 2 * p.A.rows + 1) * sizeof(double);
    CHECK(info.iterations.host_to_device_bytes == (device_input ? 0 : initial));
    CHECK(info.iterations.model_uploads == (device_input ? 0 : 1));
    ExecutionInfo exec = {};
    CHECK(execution_info(r.state, &exec) == OK && !exec.cpu_fallbacks);
}
static void kkt(const Problem &p, const std::vector<double> &x, const std::vector<double> &y,
                const std::vector<double> &z) {
    KKT k = {};
    CHECK(compute_kkt(&p, x.data(), y.data(), z.data(), &k) == OK);
    CHECK(k.primal_feasibility < 1e-8 && k.dual_feasibility < 1e-8 && k.relative_gap < 1e-8);
    /* Independent stationarity and signed complementarity, without the KKT helper. */
    for (int j = 0; j < p.A.cols; ++j) {
        long double g = p.c[j] - z[j];
        for (int t = p.Q.row_ptr[j]; t < p.Q.row_ptr[j + 1]; ++t)
            g += (long double)p.Q.values[t] * x[p.Q.col_ind[t]];
        for (int i = 0; i < p.A.rows; ++i)
            g -= (long double)matrix_get(&p.A, i, j) * y[i];
        CHECK(fabsl(g) < 1e-7L);
        CHECK(x[j] >= p.lower[j] - 1e-8 && x[j] <= p.upper[j] + 1e-8);
        if (z[j] > 1e-8)
            CHECK(near(x[j], p.lower[j]));
        if (z[j] < -1e-8)
            CHECK(near(x[j], p.upper[j]));
    }
    for (int i = 0; i < p.A.rows; ++i) {
        double a = 0.;
        for (int t = p.A.row_ptr[i]; t < p.A.row_ptr[i + 1]; ++t)
            a += p.A.values[t] * x[p.A.col_ind[t]];
        if (y[i] > 1e-8)
            CHECK(near(a, p.row_lower[i]));
        if (y[i] < -1e-8)
            CHECK(near(a, p.row_upper[i]));
    }
}
static void recover(const Problem &p, Result &r, bool oracle_original = true) {
    CHECK(result_download(&r) == OK);
    CHECK(validate(&r.reduced, nullptr, 0) == OK);
    int n = p.A.cols, m = p.A.rows, nr = r.reduced.A.cols, mr = r.reduced.A.rows;
    std::vector<double> xr(nr), yr(mr), zr(nr), x(n), y(m), z(n);
    CHECK(test_solve(&r.reduced, xr.data(), yr.data(), zr.data()));
    kkt(r.reduced, xr, yr, zr);
    PostsolveWorkspace *gpu = nullptr, *cpu = nullptr;
    CHECK(postsolve_workspace_create(r.state, &gpu) == OK);
    CHECK(postsolve_workspace_create_backend(r.state, CPU, 0, &cpu) == OK);
    std::vector<double> cx(n), cy(m), cz(n);
    for (int repeat = 0; repeat < 2; ++repeat) {
        CHECK(postsolve_with_workspace(r.state, gpu, xr.data(), nr, yr.data(), mr, zr.data(), nr,
                                       x.data(), y.data(), z.data()) == OK);
        CHECK(postsolve_with_workspace(r.state, cpu, xr.data(), nr, yr.data(), mr, zr.data(), nr,
                                       cx.data(), cy.data(), cz.data()) == OK);
        CHECK(x == cx && y == cy && z == cz);
        kkt(p, x, y, z);
        CHECK(near(objective(&p, x.data()), objective(&r.reduced, xr.data())));
    }
    if (oracle_original) {
        CHECK(test_solve(&p, cx.data(), cy.data(), cz.data()));
        CHECK(near(objective(&p, cx.data()), objective(&p, x.data())));
    }
    /* Device-input and device-output recovery, repeated and in-place compatible. */
    double *din = nullptr, *dout = nullptr;
    CU(cudaMalloc((void **)&din, ((size_t)2 * nr + mr + 1) * sizeof(double)));
    CU(cudaMalloc((void **)&dout, ((size_t)2 * n + m + 1) * sizeof(double)));
    if (nr) {
        CU(cudaMemcpy(din, xr.data(), nr * sizeof(double), cudaMemcpyHostToDevice));
        CU(cudaMemcpy(din + nr, zr.data(), nr * sizeof(double), cudaMemcpyHostToDevice));
    }
    if (mr)
        CU(cudaMemcpy(din + 2 * nr, yr.data(), mr * sizeof(double), cudaMemcpyHostToDevice));
    CHECK(postsolve_device(r.state, gpu, din, nr, din + 2 * nr, mr, din + nr, nr, dout,
                           dout + 2 * n, dout + n) == OK);
    if (n) {
        CU(cudaMemcpy(cx.data(), dout, n * sizeof(double), cudaMemcpyDeviceToHost));
        CU(cudaMemcpy(cz.data(), dout + n, n * sizeof(double), cudaMemcpyDeviceToHost));
    }
    if (m)
        CU(cudaMemcpy(cy.data(), dout + 2 * n, m * sizeof(double), cudaMemcpyDeviceToHost));
    CHECK(x == cx && y == cy && z == cz);
    if (nr) {
        double bad = NAN;
        CU(cudaMemcpy(din, &bad, sizeof(double), cudaMemcpyHostToDevice));
        CHECK(postsolve_device(r.state, gpu, din, nr, din + 2 * nr, mr, din + nr, nr, dout,
                               dout + 2 * n, dout + n) == INVALID_ARGUMENT);
        CU(cudaMemcpy(din, xr.data(), sizeof(double), cudaMemcpyHostToDevice));
        CHECK(postsolve_device(r.state, gpu, din, nr, din + 2 * nr, mr, din + nr, nr, dout,
                               dout + 2 * n, dout + n) == OK);
    }
    CU(cudaFree(din));
    CU(cudaFree(dout));
    CHECK(!r.state->records); /* CPU diagnostic replay owns its own snapshot. */
    postsolve_workspace_free(&cpu);
    postsolve_workspace_free(&gpu);
}
static Problem cycle(int n, int kind) {
    Problem p = {};
    CHECK(problem_alloc(&p, n, n) == OK);
    std::vector<double> a(n * n), q(n * n);
    for (int i = 0; i < n; ++i) {
        a[i * n + i] = 1.;
        a[i * n + (i + 1) % n] = 1.;
        if (kind != 1) {
            q[i * n + i] = kind == 2 ? 2. : 4.;
            q[i * n + (i + 1) % n] = q[i * n + (i + n - 1) % n] = -1.;
        }
        p.c[i] = 1.;
        p.lower[i] = 0.;
        p.row_lower[i] = 2.;
    }
    p.objective_offset = 7.25;
    dense(&p.A, n, n, a);
    dense(&p.Q, n, n, q);
    return p;
}
static void symmetric(void) {
    for (int kind = 0; kind < 3; ++kind) {
        Problem p = cycle(4, kind);
        Config c = config();
        Result r = {};
        CHECK(presolve_resident(&p, &c, nullptr, &r) == OK);
        resident(p, r);
        CHECK(r.reduced.A.rows == 1 && r.reduced.A.cols == 1);
        CHECK(result_download(&r) == OK);
        CHECK(matrix_get(&r.reduced.A, 0, 0) == 2.);
        CHECK(matrix_get(&r.reduced.Q, 0, 0) == (kind ? 0. : 8.));
        CHECK(r.reduced.c[0] == 4. && r.reduced.objective_offset == 7.25);
        recover(p, r);
        result_free(&r);
        /* Real folding -> main composition; main consumes the
         * folded singleton row and empty/fixed column. */
        c = config(ALL_RULES);
        CHECK(presolve_resident(&p, &c, nullptr, &r) == OK);
        resident(p, r);
        RuleInfo info = {};
        CHECK(rule_info(r.state, FOLDING, &info) == OK && info.reductions == 6);
        recover(p, r);
        result_free(&r);
        problem_free(&p);
    }
}
static void composition(void) {
    Problem p = {};
    CHECK(problem_alloc(&p, 5, 5) == OK);
    std::vector<double> a(25), q(25);
    for (int j = 0; j < 4; ++j) {
        a[j * 5 + j] = a[j * 5 + (j + 1) % 4] = 1.;
        a[j * 5 + 4] = 1.;
        p.row_lower[j] = 4.;
        p.lower[j] = 0.;
        q[j * 5 + j] = 4.;
        q[j * 5 + (j + 1) % 4] = q[j * 5 + (j + 3) % 4] = -1.;
        q[j * 5 + 4] = q[4 * 5 + j] = .5;
    }
    a[24] = 1.;
    q[24] = 3.;
    p.c[4] = -1.;
    p.lower[4] = p.upper[4] = 2.;
    p.row_lower[4] = p.row_upper[4] = 2.;
    dense(&p.A, 5, 5, a);
    dense(&p.Q, 5, 5, q);
    for (int schedule = 0; schedule < 2; ++schedule) {
        Config c = config(ALL_RULES);
        Result r = {};
        ExecutionOptions o = default_execution_options();
        o.schedule = (Schedule)schedule;
        CHECK(presolve_resident(&p, &c, &o, &r) == OK);
        resident(p, r);
        RuleInfo info = {};
        CHECK(rule_info(r.state, FOLDING, &info) == OK && info.reductions == 6);
        State *copy = nullptr;
        CHECK(cuda_state_host_clone(r.state, &copy) == OK);
        size_t first = copy->count, last = 0;
        for (size_t k = 0; k < copy->count; ++k)
            if (copy->records[k].type == REC_FOLD_COL || copy->records[k].type == REC_FOLD_ROW) {
                first = std::min(first, k);
                last = k;
            }
        /* Folding sees the original fixed variable and singleton row. Their
         * quadratic-gradient recovery records must follow the folding map. */
        CHECK(first == 0 && last + 1 < copy->count);
        CHECK(copy->records[0].m == p.A.rows && copy->records[0].n == p.A.cols);
        state_free(&copy);
        recover(p, r);
        result_free(&r);
    }
    problem_free(&p);
}
static void verify_aggregation(const Problem &p, Result &r) {
    CHECK(result_download(&r) == OK);
    State *copy = nullptr;
    CHECK(cuda_state_host_clone(r.state, &copy) == OK);
    int nr = r.reduced.A.cols, mr = r.reduced.A.rows;
    std::vector<int> cols(p.A.cols, -1), rows(p.A.rows, -1), cs(nr), rs(mr);
    for (size_t k = 0; k < copy->count; ++k) {
        const Record &rec = copy->records[k];
        bool col = rec.type == REC_FOLD_COL;
        CHECK(col || rec.type == REC_FOLD_ROW);
        int id = col ? rec.j : rec.row, group = -1;
        for (int g = 0; g < (col ? nr : mr); ++g)
            if ((col ? copy->col_ids[g] : copy->row_ids[g]) == id)
                group = g;
        CHECK(group >= 0);
        for (int j = 0; j < rec.coefficients.size; ++j)
            (col ? cols : rows)[rec.coefficients.index[j]] = group;
        (col ? cs : rs)[group] = rec.coefficients.size;
    }
    for (int g = 0; g < nr; ++g) {
        long double cost = 0.;
        for (int j = 0; j < p.A.cols; ++j)
            if (cols[j] == g)
                cost += p.c[j];
        CHECK(near(r.reduced.c[g], (double)cost));
        for (int h = 0; h < nr; ++h) {
            long double q = 0.;
            for (int j = 0; j < p.A.cols; ++j)
                if (cols[j] == g)
                    for (int k = 0; k < p.A.cols; ++k)
                        if (cols[k] == h)
                            q += matrix_get(&p.Q, j, k);
            CHECK(near(matrix_get(&r.reduced.Q, g, h), (double)q));
        }
        for (int h = 0; h < mr; ++h) {
            long double a = 0.;
            for (int i = 0; i < p.A.rows; ++i)
                if (rows[i] == h)
                    for (int j = 0; j < p.A.cols; ++j)
                        if (cols[j] == g)
                            a += matrix_get(&p.A, i, j);
            CHECK(near(matrix_get(&r.reduced.A, h, g), (double)(a / rs[h])));
        }
    }
    state_free(&copy);
}
static void path(void) {
    Problem p = {};
    CHECK(problem_alloc(&p, 1, 6) == OK);
    std::vector<double> a(6, 1.), q(36);
    for (int i = 0; i < 6; ++i) {
        q[i * 6 + i] = 3.;
        if (i < 5)
            q[i * 6 + i + 1] = q[(i + 1) * 6 + i] = 1.;
    }
    p.row_lower[0] = p.row_upper[0] = 6.;
    dense(&p.A, 1, 6, a);
    dense(&p.Q, 6, 6, q);
    Result r = {};
    Config c = config();
    CHECK(presolve_resident(&p, &c, nullptr, &r) == OK);
    resident(p, r);
    CHECK(r.reduced.A.rows == 1 && r.reduced.A.cols == 3);
    ExecutionInfo info = {};
    CHECK(execution_info(r.state, &info) == OK && info.gpu_candidate_passes == 3);
    recover(p, r);
    result_free(&r);
    problem_free(&p);
}
static void unequal_and_bounds(void) {
    for (int seed = 0; seed < 32; ++seed) {
        Problem p = {};
        CHECK(problem_alloc(&p, 3, 5) == OK);
        std::vector<double> a = {2., -1., 1., 1., 1., -1., 2., 1., 1., 1., 2., 2., -1., -1., -1.};
        std::vector<double> q(25);
        double d = (seed % 3) * .5;
        for (int j = 0; j < 5; ++j) {
            int group = j < 2 ? 0 : 1;
            for (int k = 0; k < 5; ++k) {
                int other = k < 2 ? 0 : 1;
                q[j * 5 + k] = (group ? 1. : .5) * (other ? 1. : .5);
                if (group == other)
                    q[j * 5 + k] += d * ((j == k ? (group ? 3. : 2.) : 0.) - 1.);
            }
            p.c[j] = group ? 2. : 1.;
            p.lower[j] = -INFINITY;
            p.upper[j] = INFINITY;
            switch ((seed / 4 + group) % 6) {
            case 0:
                p.lower[j] = 1.;
                break;
            case 1:
                p.upper[j] = 1.;
                p.c[j] *= -4.;
                break;
            case 2:
                p.lower[j] = p.upper[j] = 1.;
                p.c[j] *= -4.;
                break;
            case 3:
                p.lower[j] = p.upper[j] = 1.;
                break;
            case 4:
                p.lower[j] = -2.;
                p.upper[j] = 3.;
                break;
            default:
                break;
            }
        }
        for (int i = 0; i < 2; ++i) {
            p.row_lower[i] = seed % 4 == 0 ? 4. : -INFINITY;
            p.row_upper[i] = seed % 4 == 1 ? 4. : INFINITY;
            if (seed % 4 == 2)
                p.row_lower[i] = p.row_upper[i] = 4.;
        }
        p.row_lower[2] = p.row_upper[2] = 1.;
        p.objective_offset = -3.5;
        dense(&p.A, 3, 5, a);
        dense(&p.Q, 5, 5, q);
        Config c = config();
        Result r = {};
        CHECK(presolve_resident(&p, &c, nullptr, &r) == OK);
        resident(p, r);
        CHECK(r.reduced.A.rows == 2 && r.reduced.A.cols == 2);
        verify_aggregation(p, r);
        recover(p, r);
        result_free(&r);
        problem_free(&p);
    }
}
static void no_merge(void) {
    /* Q distinguishes otherwise identical LP columns, including tiny terms. */
    for (int variant = 0; variant < 4; ++variant) {
        Problem p = {};
        CHECK(problem_alloc(&p, 2, 2) == OK);
        std::vector<double> a = {1., 0., 0., 1.};
        std::vector<double> q = {1., 0., 0., variant == 0 ? 2. : 1.};
        p.row_lower[0] = p.row_lower[1] = 1.;
        if (variant == 1)
            p.c[1] = 1e-100;
        if (variant == 2)
            p.lower[1] = 1e-100;
        if (variant == 3)
            p.row_lower[1] = nextafter(1., 2.);
        dense(&p.A, 2, 2, a);
        dense(&p.Q, 2, 2, q);
        Config c = config();
        Result r = {};
        CHECK(presolve_resident(&p, &c, nullptr, &r) == OK);
        resident(p, r);
        CHECK(r.reduced.A.rows == 2 && r.reduced.A.cols == 2 && !r.reductions);
        recover(p, r);
        result_free(&r);
        problem_free(&p);
    }
    /* Equal rounded Hessian row sums hide a genuine one-unit asymmetry. */
    Problem p = {};
    CHECK(problem_alloc(&p, 1, 3) == OK);
    dense(&p.A, 1, 3, {1., 1., 1.});
    dense(&p.Q, 3, 3, {1e16, 1., 0., 1., 1e16, 2., 0., 2., 1e16});
    p.row_lower[0] = 1.;
    Config c = config();
    Result r = {};
    CHECK(presolve_resident(&p, &c, nullptr, &r) == OK);
    CHECK(r.reduced.A.cols == 3 && !r.reductions);
    result_free(&r);
    problem_free(&p);
}
static void conservative_rejection(void) {
    for (int variant = 0; variant < 3; ++variant) {
        Problem p = {};
        CHECK(problem_alloc(&p, 3, 3) == OK);
        std::vector<double> a(9), q(9);
        for (int i = 0; i < 3; ++i) {
            a[i * 3 + i] = 1.;
            if (!variant) {
                a[i * 3 + (i + 1) % 3] = ldexp(1., -54);
                a[i * 3 + (i + 2) % 3] = ldexp(1., -108);
            }
            q[i * 3 + i] = variant == 1 ? DBL_MAX : 1.;
            if (variant == 2)
                p.c[i] = DBL_MAX;
            p.row_lower[i] = 0.;
        }
        dense(&p.A, 3, 3, a);
        dense(&p.Q, 3, 3, q);
        Config c = config();
        Result r = {};
        CHECK(presolve_resident(&p, &c, nullptr, &r) == OK);
        resident(p, r);
        BatchInfo info = {};
        CHECK(batch_info(r.state, &info) == OK && info.numerical_rejections == 1);
        CHECK(!r.reductions && r.reduced.A.rows == 3 && r.reduced.A.cols == 3);
        CHECK(result_download(&r) == OK);
        CHECK(r.reduced.Q.nnz == p.Q.nnz && r.reduced.A.nnz == p.A.nnz);
        for (int k = 0; k < p.Q.nnz; ++k)
            CHECK(p.Q.values[k] == r.reduced.Q.values[k]);
        for (int k = 0; k < p.A.nnz; ++k)
            CHECK(p.A.values[k] == r.reduced.A.values[k]);
        CHECK(r.state->count == 0);
        result_free(&r);
        problem_free(&p);
    }
}
static void row_only(void) {
    Problem p = {};
    CHECK(problem_alloc(&p, 3, 2) == OK);
    dense(&p.A, 3, 2, {1., 2., 1., 2., 1., 2.});
    dense(&p.Q, 2, 2, {2., -1., -1., 2.});
    p.c[0] = 1.;
    p.c[1] = 2.;
    for (int i = 0; i < 3; ++i)
        p.row_lower[i] = 3.;
    Config c = config();
    Result r = {};
    CHECK(presolve_resident(&p, &c, nullptr, &r) == OK);
    resident(p, r);
    CHECK(r.reduced.A.rows == 1 && r.reduced.A.cols == 2);
    verify_aggregation(p, r);
    recover(p, r);
    result_free(&r);
    problem_free(&p);
}
static void signature_collision(void) {
    // Every coefficient is zero in the projection field. An exact certificate
    // must reject these non-equitable Q columns even with two hash channels.
    Problem p = {};
    CHECK(problem_alloc(&p, 1, 2) == OK);
    dense(&p.A, 1, 2, {1., 1.});
    dense(&p.Q, 2, 2, {2147483647., 0., 0., 4294967294.});
    p.row_lower[0] = 1e-3;
    Config c = config();
    Result r = {};
    CHECK(presolve_resident(&p, &c, nullptr, &r) == OK);
    resident(p, r);
    CHECK(!r.reductions && r.state->count == 0 && r.reduced.A.cols == 2);
    BatchInfo info = {};
    CHECK(batch_info(r.state, &info) == OK && info.numerical_rejections == 1);
    recover(p, r);
    result_free(&r);
    problem_free(&p);
}
static void long_refinement(void) {
    // Mirrored path classes propagate inward by one edge each round. This
    // projection path must finish without losing these valid mirrored classes.
    enum { n = 96 };
    Problem p = {};
    COO a = {}, q = {};
    CHECK(problem_alloc(&p, 1, n) == OK);
    for (int j = 0; j < n; ++j) {
        coo_add(&a, 0, j, 1.);
        coo_add(&q, j, j, 3.);
        if (j + 1 < n) {
            coo_add(&q, j, j + 1, 1.);
            coo_add(&q, j + 1, j, 1.);
        }
    }
    CHECK(coo_finish(&a, 1, n, &p.A) == OK);
    CHECK(coo_finish(&q, n, n, &p.Q) == OK);
    Config c = config();
    Result r = {};
    CHECK(presolve_resident(&p, &c, nullptr, &r) == OK);
    resident(p, r);
    CHECK(r.reductions == n / 2 && r.reduced.A.cols == n / 2 && r.state->count > 0);
    ExecutionInfo info = {};
    CHECK(execution_info(r.state, &info) == OK && info.gpu_candidate_passes == n / 2);
    CHECK(result_download(&r) == OK);
    verify_aggregation(p, r);
    result_free(&r);
    problem_free(&p);
}
static void small_reduction(void) {
    // Retain a small but exact fold: the LP 80% heuristic is too aggressive here.
    Problem p = {};
    CHECK(problem_alloc(&p, 10, 10) == OK);
    std::vector<double> a(100), q(100);
    for (int j = 0; j < 10; ++j) {
        a[j * 10 + j] = 1.;
        q[j * 10 + j] = 2.;
        p.c[j] = j < 2 ? 0. : j;
        p.row_lower[j] = j < 2 ? 0. : j;
    }
    dense(&p.A, 10, 10, a);
    dense(&p.Q, 10, 10, q);
    Config c = config();
    Result r = {};
    CHECK(presolve_resident(&p, &c, nullptr, &r) == OK);
    resident(p, r);
    CHECK(r.reductions == 2 && r.state->count > 0 && r.reduced.A.cols == 9 &&
          r.reduced.A.rows == 9);
    ExecutionInfo info = {};
    CHECK(execution_info(r.state, &info) == OK && info.gpu_candidate_passes > 0);
    recover(p, r);
    result_free(&r);
    problem_free(&p);
}
static void device_input_and_large(void) {
    Problem p = cycle(2048, 0);
    Config identity = config(0), folding = config();
    Result source = {}, r = {};
    CHECK(presolve_resident(&p, &identity, nullptr, &source) == OK);
    DeviceProblemView view = {};
    CHECK(device_problem_view(source.state, &view) == OK);
    CHECK(presolve_device(&view, &folding, nullptr, &r) == OK);
    resident(p, r, true);
    CHECK(r.reduced.A.rows == 1 && r.reduced.A.cols == 1);
    result_free(&source); /* Folding owns both its reduced model and maps. */
    recover(p, r, false);
    result_free(&r);
    problem_free(&p);
}
static void boundaries(void) {
    for (int side = 0; side < 3; ++side) {
        Problem p = {};
        CHECK(problem_alloc(&p, side == 1 ? 2 : 0, side == 2 ? 2 : 0) == OK);
        Config c = config();
        Result r = {};
        CHECK(presolve_resident(&p, &c, nullptr, &r) == OK);
        resident(p, r);
        CHECK(r.reduced.A.rows == p.A.rows && r.reduced.A.cols == p.A.cols);
        recover(p, r);
        result_free(&r);
        problem_free(&p);
    }
    Problem p = cycle(4, 0);
    Result r = {};
    Config c = config();
    c.max_iterations = 0;
    CHECK(presolve_resident(&p, &c, nullptr, &r) == OK && !r.reductions);
    result_free(&r);
    c.max_iterations = 10;
    c.max_seconds = 0.;
    CHECK(presolve_resident(&p, &c, nullptr, &r) == TIME_LIMIT && !r.reductions);
    recover(p, r);
    result_free(&r);
    c = config();
    c.backend = CPU;
    CHECK(presolve(&p, &c, &r) == INVALID_ARGUMENT && !r.state);
    result_free(&r);
    problem_free(&p);
}
int main() {
    if (!cuda_available(0))
        return 77;
    symmetric();
    composition();
    path();
    unequal_and_bounds();
    no_merge();
    conservative_rejection();
    row_only();
    signature_collision();
    long_refinement();
    small_reduction();
    device_input_and_large();
    boundaries();
    printf("folding checks=%d failures=%d\n", checks, failures);
    return failures ? 1 : 0;
}
