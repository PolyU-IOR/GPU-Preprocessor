#ifdef __APPLE__
#define _DARWIN_C_SOURCE
#endif
#define _POSIX_C_SOURCE 200809L
#include "gpupresolve_qp.h"
#include "support/qp_oracle.h"
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <zlib.h>

static int assertions = 0, failures = 0;
static const Backend backend = CUDA;
#define CHECK(condition)                                                                           \
    do {                                                                                           \
        assertions++;                                                                              \
        if (!(condition)) {                                                                        \
            fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #condition);                   \
            failures++;                                                                            \
        }                                                                                          \
    } while (0)
static int near(double a, double b, double tol) {
    return a == b ||
           (isfinite(a) && isfinite(b) && fabs(a - b) <= tol * (1. + fmax(fabs(a), fabs(b))));
}
static Config config(uint32_t rules) {
    Config c = default_config();
    c.backend = backend;
    c.rules = rules;
    c.bound_tol = 1e-10;
    c.feasibility_tol = 1e-10;
    return c;
}
static double get(const Matrix *a, int i, int j) {
    for (int k = a->row_ptr[i]; k < a->row_ptr[i + 1]; k++)
        if (a->col_ind[k] == j)
            return a->values[k];
    return 0.;
}
static void dense_matrix(Matrix *a, int m, int n, const double *data) {
    size_t count = (size_t)m * (size_t)n;
    int *row = calloc(count ? count : 1, sizeof(int)),
        *col = calloc(count ? count : 1, sizeof(int));
    double *v = calloc(count ? count : 1, sizeof(double));
    int nz = 0;
    CHECK(row && col && v);
    if (!row || !col || !v)
        exit(2);
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
            if (data[(size_t)i * n + j] != 0.) {
                row[nz] = i;
                col[nz] = j;
                v[nz++] = data[(size_t)i * n + j];
            }
    CHECK(matrix_from_triplets(a, m, n, nz, row, col, v) == OK);
    free(row);
    free(col);
    free(v);
}
static Problem problem(int m, int n, const double *Q, const double *A, const double *c) {
    Problem p = {0};
    CHECK(problem_alloc(&p, m, n) == OK);
    dense_matrix(&p.Q, n, n, Q);
    dense_matrix(&p.A, m, n, A);
    for (int j = 0; j < n; j++)
        p.c[j] = c[j];
    return p;
}
static void independent_kkt(const Problem *p, const double *x, const double *y,
                            const double *z) {
    for (int j = 0; j < p->A.cols; j++) {
        double g = p->c[j] - z[j];
        for (int k = 0; k < p->A.cols; k++)
            g += get(&p->Q, j, k) * x[k];
        for (int i = 0; i < p->A.rows; i++)
            g -= get(&p->A, i, j) * y[i];
        CHECK(fabs(g) < 1e-7 * (1. + fabs(p->c[j])));
        CHECK(x[j] >= p->lower[j] - 1e-7 && x[j] <= p->upper[j] + 1e-7);
        if (z[j] > 1e-7)
            CHECK(near(x[j], p->lower[j], 1e-7));
        if (z[j] < -1e-7)
            CHECK(near(x[j], p->upper[j], 1e-7));
    }
    for (int i = 0; i < p->A.rows; i++) {
        double a = 0.;
        for (int j = 0; j < p->A.cols; j++)
            a += get(&p->A, i, j) * x[j];
        CHECK(a >= p->row_lower[i] - 1e-7 && a <= p->row_upper[i] + 1e-7);
        if (y[i] > 1e-7)
            CHECK(near(a, p->row_lower[i], 1e-7));
        if (y[i] < -1e-7)
            CHECK(near(a, p->row_upper[i], 1e-7));
    }
}
static void compare_models(const Problem *a, const Problem *b, double bound_tol) {
    CHECK(a->A.rows == b->A.rows && a->A.cols == b->A.cols);
    CHECK(near(a->objective_offset, b->objective_offset, 1e-12));
    if (a->A.rows != b->A.rows || a->A.cols != b->A.cols)
        return;
    for (int i = 0; i < a->A.rows; i++) {
        CHECK(near(a->row_lower[i], b->row_lower[i], 1e-12));
        CHECK(near(a->row_upper[i], b->row_upper[i], 1e-12));
        for (int j = 0; j < a->A.cols; j++)
            CHECK(near(get(&a->A, i, j), get(&b->A, i, j), 1e-12));
    }
    for (int j = 0; j < a->A.cols; j++) {
        CHECK(near(a->c[j], b->c[j], 1e-12));
        /* Snapshot-based parallel propagation can stop at a different point
         * within bound_tol. Structural/objective coefficients remain strict;
         * primal, dual and complementarity checks above use the original QP. */
        CHECK(near(a->lower[j], b->lower[j], bound_tol));
        CHECK(near(a->upper[j], b->upper[j], bound_tol));
        for (int k = 0; k < a->A.cols; k++)
            CHECK(near(get(&a->Q, j, k), get(&b->Q, j, k), 1e-12));
    }
}
static void check_case(const Problem *p, Config c, const double *expected, int wanted_m,
                       int wanted_n) {
    Result r = {0};
    Status status = presolve(p, &c, &r);
    CHECK(status == OK);
    if (status) {
        fprintf(stderr, "presolve: %s\n", r.message);
        result_free(&r);
        return;
    }
    if (wanted_m >= 0)
        CHECK(r.reduced.A.rows == wanted_m);
    if (wanted_n >= 0)
        CHECK(r.reduced.A.cols == wanted_n);
    double xr[12] = {0}, yr[12] = {0}, zr[12] = {0}, x[12], y[12], z[12];
    int solved = test_solve(&r.reduced, xr, yr, zr);
    CHECK(solved);
    if (!solved) {
        result_free(&r);
        return;
    }
    for (int repeat = 0; repeat < 2; repeat++) {
        CHECK(postsolve(r.state, xr, r.reduced.A.cols, yr, r.reduced.A.rows, zr,
                            r.reduced.A.cols, x, y, z) == OK);
        independent_kkt(p, x, y, z);
        if (expected)
            for (int j = 0; j < p->A.cols; j++)
                CHECK(near(x[j], expected[j], 1e-8));
        CHECK(near(objective(p, x), objective(&r.reduced, xr), 1e-9));
        KKT k, cpu;
        CHECK(compute_kkt_backend(p, x, y, z, backend, 0, &k) == OK);
        CHECK(compute_kkt(p, x, y, z, &cpu) == OK);
        CHECK(k.primal_feasibility < 1e-8);
        CHECK(k.dual_feasibility < 1e-8);
        CHECK(k.relative_gap < 1e-8);
        CHECK(near(k.dual_objective, cpu.dual_objective, 1e-10));
    }
    CHECK(postsolve(r.state, xr, r.reduced.A.cols + 1, yr, r.reduced.A.rows, zr,
                        r.reduced.A.cols, x, y, z) == INVALID_ARGUMENT);
    state_free(&r.state);
    CHECK(postsolve(r.state, xr, 0, yr, 0, zr, 0, x, y, z) == INVALID_ARGUMENT);
    result_free(&r);
    result_free(&r);
}
static void test_examples(void) {
    Problem p = {0};
    char msg[256];
    CHECK(read_mps(SOURCE_DIR "/qp/examples/qp_fixed.mps", &p, msg, sizeof(msg)) == OK);
    double expected[] = {1, 3, 1};
    check_case(&p, config(FIXED_COLS), expected, 1, 2);
    check_case(&p, config(ALL_RULES), expected, -1, -1);
    double x[] = {1, 3, 1}, y[] = {0}, z[] = {-2, 0, 0};
    KKT k;
    CHECK(compute_kkt(&p, x, y, z, &k) == OK);
    CHECK(k.primal_objective == -13. && k.dual_objective == -13. && k.relative_gap == 0.);
    x[0] = NAN;
    CHECK(compute_kkt(&p, x, y, z, &k) == OK);
    CHECK(isinf(k.primal_feasibility));
    Config c = config(ALL_RULES);
    c.max_seconds = 0.;
    Result r = {0};
    CHECK(presolve(&p, &c, &r) == TIME_LIMIT);
    compare_models(&p, &r.reduced, 1e-12);
    result_free(&r);
    c.max_iterations = 0;
    c.max_seconds = 100.;
    CHECK(presolve(&p, &c, &r) == OK);
    compare_models(&p, &r.reduced, 1e-12);
    result_free(&r);
    c.zero_tol = NAN;
    CHECK(presolve(&p, &c, &r) == INVALID_ARGUMENT);
    result_free(&r);
    c = config(ALL_RULES);
    c.rules = 1u << 30;
    CHECK(presolve(&p, &c, &r) == INVALID_ARGUMENT);
    result_free(&r);
    problem_free(&p);
}
static void test_rules(void) {
    const double q2[] = {2}, q0[] = {0}, c0[] = {0}, a1[] = {1};
    double one[] = {1};
    Problem p = problem(1, 1, q2, a1, c0);
    p.lower[0] = 0.;
    p.row_lower[0] = 1.;
    check_case(&p, config(SINGLETON_ROWS | EMPTY_COLS), one, 0, 0);
    problem_free(&p);
    double q[] = {2, 0, 0, 2}, a[] = {1, 1}, c[] = {0, -4}, target[] = {1, 2};
    p = problem(1, 2, q, a, c);
    p.lower[0] = p.lower[1] = 0.;
    p.upper[0] = p.upper[1] = 2.;
    p.row_lower[0] = 3.;
    check_case(&p, config(PROPAGATION), target, 1, 2);
    problem_free(&p);
    double ap[] = {1, 2}, two[] = {2};
    p = problem(2, 1, q2, ap, c0);
    p.row_lower[1] = 4.;
    check_case(&p, config(PARALLEL_ROWS), two, 1, 1);
    problem_free(&p);
    double qc[] = {4, 1, 1, 3}, cc[] = {-3, -7}, tc[] = {1, 2};
    p = problem(0, 2, qc, NULL, cc);
    p.lower[0] = p.upper[0] = 1.;
    check_case(&p, config(FIXED_COLS), tc, 0, 1);
    problem_free(&p);
    double cm[] = {1, 2};
    p = problem(0, 2, qc, NULL, cm);
    p.lower[0] = .1;
    p.lower[1] = .2;
    p.upper[0] = p.upper[1] = 1.;
    Config conf = config(EMPTY_COLS);
    conf.monotone_empty_columns = 1;
    double tm[] = {.1, .2};
    check_case(&p, conf, tm, 0, 0);
    problem_free(&p);
    double qd[] = {0, 0, 0, 2}, cd[] = {1, -1}, td[] = {1, 1};
    p = problem(1, 2, qd, a, cd);
    p.row_lower[0] = 2.;
    p.row_upper[0] = 4.;
    p.lower[1] = 0.;
    p.upper[1] = 1.;
    check_case(&p, config(SINGLETON_DUAL | SINGLETON_EQ), td, 0, 1);
    problem_free(&p);
    double qpar[] = {2, 4, 4, 8}, cpar[] = {-6, -12}, apar[] = {1, 2};
    p = problem(1, 2, qpar, apar, cpar);
    p.lower[0] = p.lower[1] = 0.;
    p.upper[0] = p.upper[1] = 4.;
    p.row_lower[0] = p.row_upper[0] = 3.;
    check_case(&p, config(PARALLEL_COLS), NULL, 1, 1);
    problem_free(&p);
    double qneg[] = {2, -4, -4, 8}, cneg[] = {-6, 12}, aneg[] = {1, -2};
    p = problem(1, 2, qneg, aneg, cneg);
    p.lower[0] = p.lower[1] = -4.;
    p.upper[0] = p.upper[1] = 4.;
    p.row_lower[0] = p.row_upper[0] = 3.;
    check_case(&p, config(PARALLEL_COLS), NULL, 1, 1);
    problem_free(&p);
    p = problem(1, 1, q2, a1, c0);
    p.lower[0] = 1.;
    p.upper[0] = 3.;
    p.row_upper[0] = 4.;
    check_case(&p, config(ACTIVITY), one, 0, 1);
    problem_free(&p);
    double cf[] = {-5}, tf[] = {1};
    p = problem(1, 1, q2, a1, cf);
    p.lower[0] = 0.;
    p.upper[0] = 1.;
    check_case(&p, config(DUAL_FIX), tf, 1, 0);
    problem_free(&p);
    Result r = {0};
    p = problem(1, 1, q2, q0, c0);
    p.row_lower[0] = 1.;
    CHECK(presolve(&p, &conf, &r) == OK);
    result_free(&r);
    conf = config(EMPTY_ROWS);
    CHECK(presolve(&p, &conf, &r) == INFEASIBLE);
    CHECK(!r.state);
    result_free(&r);
    problem_free(&p);
    double cu[] = {-1};
    p = problem(0, 1, q0, NULL, cu);
    conf = config(EMPTY_COLS);
    CHECK(presolve(&p, &conf, &r) == UNBOUNDED);
    CHECK(!r.state);
    result_free(&r);
    problem_free(&p);
    p = problem(1, 1, q2, a1, c0);
    p.lower[0] = 2.;
    p.upper[0] = 1.;
    CHECK(presolve(&p, &conf, &r) == INFEASIBLE);
    result_free(&r);
    problem_free(&p);
}
/* Regression for random audit seed 20261030, case 1044. Dependent rows and
 * fixed bounds used to produce an uncertified 1e19 dual in the test oracle. */
static void test_degenerate_oracle(void) {
    const double q[] = {
        6.252837934986111,   .6122303958734484,   2.6012419934706448,  1.9497308800070225,
        .4585972348505956,   -.7896464249131112,  .6122303958734484,   3.8609299467362055,
        .08567446281816671,  -.7118579846478285,  -.22658064692590585, -.6031923677756871,
        2.6012419934706448,  .08567446281816671,  7.3055713350429405,  -2.638746541360824,
        -4.232779015471203,  -1.6949000404144396, 1.9497308800070225,  -.7118579846478285,
        -2.638746541360824,  13.50930523138148,   .6469716248602719,   4.157873661150906,
        .4585972348505956,   -.22658064692590585, -4.232779015471203,  .6469716248602719,
        8.747507503232537,   2.73000831627692,    -.7896464249131112,  -.6031923677756871,
        -1.6949000404144396, 4.157873661150906,   2.73000831627692,    5.527535343638462};
    const double a[] = {1, 0, 0, 1, 0, 0, -2, 0, 0, 0,  0, 0, -2, 0, 0,
                        0, 2, 1, 1, 1, 0, 0,  0, 0, -2, 0, 0, 0,  0, -2};
    const double c[] = {-6.389996545498191,  -9.050373973974322, -8.236519795238125,
                        -17.678624766252327, 6.92666772553896,   -5.715139555972519};
    const double lower[] = {-1, 1, -2, 2, -.25, -.5}, upper[] = {-1, 2, 2, 2, .25, -.5};
    const double rl[] = {1, 1, 1, 0, 3}, ru[] = {2, 2, 2, 1, 3};
    const double expected[] = {-1, 2, 2, 2, .25, -.5};
    Problem p = problem(5, 6, q, a, c);
    memcpy(p.lower, lower, sizeof(lower));
    memcpy(p.upper, upper, sizeof(upper));
    memcpy(p.row_lower, rl, sizeof(rl));
    memcpy(p.row_upper, ru, sizeof(ru));
    double x[6], y[5], z[6];
    int solved = test_solve(&p, x, y, z);
    CHECK(solved);
    if (solved) {
        independent_kkt(&p, x, y, z);
        for (int j = 0; j < 6; ++j)
            CHECK(near(x[j], expected[j], 1e-8));
    }
    problem_free(&p);
}
static void test_numerics(void) {
    for (int sign = -1; sign <= 1; sign += 2) {
        double q[] = {1e-12}, c[] = {(double)sign}, expected[] = {-sign * 1e12};
        Problem p = problem(0, 1, q, NULL, c);
        check_case(&p, config(ALL_RULES), expected, 0, 0);
        problem_free(&p);
    }
    double q[] = {1, 0, 0, 0, 1, 0, 0, 0, 0}, a[] = {1, -1, -5e-6}, c[] = {0, 0, 0};
    Problem p = problem(1, 3, q, a, c);
    p.row_lower[0] = p.row_upper[0] = 0.;
    Config conf = config(SINGLETON_EQ);
    conf.max_q_fill = -1;
    conf.max_q_fill_ratio = INFINITY;
    conf.max_iterations = 1;
    Result r = {0};
    CHECK(presolve(&p, &conf, &r) == OK);
    CHECK(r.reduced.A.cols == 2);
    CHECK(near(get(&r.reduced.Q, 1, 1), 2.5e-11, 1e-25));
    CHECK(get(&r.reduced.Q, 1, 1) > 0.);
    CHECK(get(&r.reduced.Q, 0, 0) * get(&r.reduced.Q, 1, 1) -
              get(&r.reduced.Q, 0, 1) * get(&r.reduced.Q, 1, 0) >
          0.);
    double xr[] = {-5e3, 1e9}, z[] = {0, 0}, x[3], yo[1], zo[3];
    CHECK(postsolve(r.state, xr, 2, NULL, 0, z, 2, x, yo, zo) == OK);
    CHECK(near(x[0] - x[1] - 5e-6 * x[2], 0., 1e-9));
    CHECK(near(objective(&p, x), objective(&r.reduced, xr), 1e-12));
    result_free(&r);
    conf.max_substitution_support = 0;
    CHECK(presolve(&p, &conf, &r) == OK);
    CHECK(r.reduced.A.cols == 3);
    result_free(&r);
    problem_free(&p);
    for (int difference = 0; difference < 3; difference++) {
        double Q[] = {1, 1, 1, 1}, A[] = {1, 1, 1, 1}, C[] = {0, 0};
        if (difference == 0)
            Q[3] += 5e-7;
        if (difference == 1)
            C[1] = 5e-7;
        if (difference == 2)
            A[3] += 5e-7;
        p = problem(2, 2, Q, A, C);
        p.row_lower[0] = p.row_upper[0] = 0.;
        p.lower[1] = 1e4;
        conf = config(PARALLEL_COLS);
        CHECK(presolve(&p, &conf, &r) == OK);
        CHECK(r.reduced.A.cols == 2);
        compare_models(&p, &r.reduced, 1e-12);
        result_free(&r);
        problem_free(&p);
    }
    /* Bound replacement during a non-singleton doubleton substitution. */
    double Q[] = {2, 0, 0, 0, 3, 0, 0, 0, 4}, A[] = {1, 2, 0, 1, 0, 1}, C[] = {-2, -6, -12},
           target[] = {1, 2, 3};
    p = problem(2, 3, Q, A, C);
    p.row_lower[0] = p.row_upper[0] = 5.;
    p.row_lower[1] = p.row_upper[1] = 4.;
    p.lower[0] = 0.;
    p.upper[0] = 2.;
    check_case(&p, config(DOUBLETON_EQ), target, -1, -1);
    problem_free(&p);
}
static void test_contract_edges(void) {
    Matrix standalone = {0};
    int index[] = {0};
    double value[] = {2.};
    CHECK(matrix_from_triplets(&standalone, 1, 1, 1, index, index, value) == OK);
    matrix_free(&standalone);
    CHECK(!standalone.rows && !standalone.cols && !standalone.nnz);
    CHECK(!standalone.row_ptr && !standalone.col_ind && !standalone.values);
    matrix_free(&standalone);
    matrix_free(NULL);
    Result r = {0};
    Config c = config(EMPTY_COLS);
    double tiny[] = {1e-300}, large[] = {1e300};
    Problem p = problem(0, 1, tiny, NULL, large);
    CHECK(presolve(&p, &c, &r) == NUMERICAL_ERROR);
    result_free(&r);
    problem_free(&p);
    double zero[] = {0.}, linear[] = {1e-300};
    p = problem(0, 1, zero, NULL, linear);
    CHECK(presolve(&p, &c, &r) == UNBOUNDED);
    result_free(&r);
    problem_free(&p);
    double Q[] = {1, 0, 0, 0, 1, 0, 0, 0, 1}, A[] = {1, 1, 1}, C[] = {0, 0, 0};
    p = problem(1, 3, Q, A, C);
    p.row_lower[0] = p.row_upper[0] = 0.;
    c = config(SINGLETON_EQ);
    c.max_q_fill = 0;
    c.max_q_fill_ratio = INFINITY;
    CHECK(presolve(&p, &c, &r) == OK);
    compare_models(&p, &r.reduced, 1e-12);
    result_free(&r);
    c.max_q_fill = -1;
    c.max_q_fill_ratio = 1.;
    CHECK(presolve(&p, &c, &r) == OK);
    compare_models(&p, &r.reduced, 1e-12);
    result_free(&r);
    problem_free(&p);
    char msg[256];
    CHECK(read_mps(SOURCE_DIR "/qp/examples/qp_fixed.mps", &p, msg, sizeof(msg)) == OK);
    c = config(FIXED_COLS);
    CHECK(presolve(&p, &c, &r) == OK);
    problem_free(&p);
    r.reduced.c[0] = 123.; /* Recovery metadata is independently owned. */
    double x[] = {3, 1, 0}, y[] = {0}, z[] = {0, 0, 0};
    CHECK(postsolve(r.state, x, 2, y, 1, z, 2, x, y, z) == OK);
    CHECK(x[0] == 1. && x[1] == 3. && x[2] == 1. && z[0] == -2.);
    result_free(&r);
    CHECK(problem_alloc(&p, 0, 0) == OK);
    p.objective_offset = 2.;
    check_case(&p, config(ALL_RULES), NULL, 0, 0);
    problem_free(&p);
    Matrix matrix = {0};
    int rows[] = {0, 0}, cols[] = {0, 0};
    double values[] = {1e-12, 1e-12};
    CHECK(matrix_from_triplets(&matrix, 1, 1, 2, rows, cols, values) == OK);
    CHECK(matrix.nnz == 1 && matrix.values[0] == 2e-12);
    /* Transfer the matrix into an owning problem to free it through the API. */
    CHECK(problem_alloc(&p, 0, 1) == OK);
    free(p.Q.row_ptr);
    p.Q = matrix;
    problem_free(&p);
}

static unsigned seed = 71;
static double random_value(void) {
    seed = 1664525u * seed + 1013904223u;
    return ((double)(seed >> 8) / 16777216.) * 2. - 1.;
}
static void test_random(void) {
    for (int trial = 0; trial < 100; trial++) {
        int n = 4 + trial % 4, m = 1 + trial % 3;
        double B[64], Q[64] = {0}, A[64] = {0}, c[8] = {0}, target[8];
        for (int i = 0; i < n; i++) {
            target[i] = random_value();
            for (int j = 0; j < n; j++)
                B[i * n + j] = random_value();
        }
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++) {
                double sum = i == j ? 1. : 0.;
                for (int k = 0; k < n; k++)
                    sum += B[k * n + i] * B[k * n + j];
                Q[i * n + j] = sum;
            }
        for (int i = 0; i < m; i++) {
            A[i * n + i] = 1.;
            for (int j = m; j < n; j++)
                A[i * n + j] = random_value();
        }
        double dual[8];
        for (int i = 0; i < m; i++)
            dual[i] = random_value();
        for (int j = 0; j < n; j++) {
            for (int k = 0; k < n; k++)
                c[j] -= Q[j * n + k] * target[k];
            for (int i = 0; i < m; i++)
                c[j] += A[i * n + j] * dual[i];
        }
        Problem p = problem(m, n, Q, A, c);
        p.objective_offset = 2.;
        for (int i = 0; i < m; i++) {
            double b = 0.;
            for (int j = 0; j < n; j++)
                b += A[i * n + j] * target[j];
            p.row_lower[i] = p.row_upper[i] = b;
        }
        Config conf = config(trial % 2 ? ALL_RULES : SINGLETON_EQ);
        conf.max_substitution_support = -1;
        conf.max_q_fill = -1;
        conf.max_q_fill_ratio = INFINITY;
        check_case(&p, conf, target, 0, n - m);
        Result r = {0};
        CHECK(presolve(&p, &conf, &r) == OK);
        for (int sample = 0; sample < 4; sample++) {
            double xr[8], zr[8] = {0}, x[8], y[8], z[8];
            for (int j = 0; j < r.reduced.A.cols; j++)
                xr[j] = random_value();
            CHECK(postsolve(r.state, xr, r.reduced.A.cols, NULL, 0, zr, r.reduced.A.cols, x, y,
                                z) == OK);
            for (int i = 0; i < m; i++) {
                double b = 0.;
                for (int j = 0; j < n; j++)
                    b += A[i * n + j] * x[j];
                CHECK(near(b, p.row_lower[i], 1e-10));
            }
            double expected_objective = 2.;
            for (int i = 0; i < n; i++) {
                expected_objective += c[i] * x[i];
                for (int j = 0; j < n; j++)
                    expected_objective += .5 * x[i] * Q[i * n + j] * x[j];
            }
            CHECK(near(expected_objective, objective(&r.reduced, xr), 1e-10));
        }
        result_free(&r);
        problem_free(&p);
    }
}
static void test_random_inequalities(void) {
    for (int trial = 0; trial < 80; trial++) {
        int n = 3 + trial % 2, m = 1 + trial % 2;
        double B[16], Q[16] = {0}, A[8], c[4] = {0}, expected[4], dual[2], bound_dual[4] = {0};
        for (int j = 0; j < n; j++) {
            expected[j] = .5 * random_value();
            if ((trial + j) % 3 == 0) {
                expected[j] = 1.;
                bound_dual[j] = -.3;
            }
            if ((trial + j) % 5 == 0) {
                expected[j] = -1.;
                bound_dual[j] = .4;
            }
            for (int k = 0; k < n; k++)
                B[j * n + k] = random_value();
        }
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++) {
                double sum = i == j ? 1. : 0.;
                for (int k = 0; k < n; k++)
                    sum += B[k * n + i] * B[k * n + j];
                Q[i * n + j] = sum;
            }
        for (int i = 0; i < m; i++) {
            dual[i] = (trial + i) % 2 ? -.7 : .6;
            for (int j = 0; j < n; j++)
                A[i * n + j] = random_value();
        }
        for (int j = 0; j < n; j++) {
            c[j] = bound_dual[j];
            for (int k = 0; k < n; k++)
                c[j] -= Q[j * n + k] * expected[k];
            for (int i = 0; i < m; i++)
                c[j] += A[i * n + j] * dual[i];
        }
        Problem p = problem(m, n, Q, A, c);
        for (int j = 0; j < n; j++) {
            p.lower[j] = -1.;
            p.upper[j] = 1.;
        }
        for (int i = 0; i < m; i++) {
            double rhs = 0.;
            for (int j = 0; j < n; j++)
                rhs += A[i * n + j] * expected[j];
            if (dual[i] > 0.)
                p.row_lower[i] = rhs;
            else
                p.row_upper[i] = rhs;
        }
        check_case(&p, config(ALL_RULES), expected, -1, -1);
        problem_free(&p);
    }
}

static void write_text(const char *path, const char *text) {
    FILE *f = fopen(path, "wx");
    CHECK(f != NULL);
    if (f) {
        CHECK(fputs(text, f) >= 0);
        CHECK(fclose(f) == 0);
    }
}
static void test_mps(void) {
    char directory[] = "/tmp/presolve-native-XXXXXX";
    CHECK(mkdtemp(directory) != NULL);
    char path[512], gzpath[512], msg[256];
    Problem p = {0}, q = {0};
    snprintf(path, sizeof(path), "%s/model.mps", directory);
    snprintf(gzpath, sizeof(gzpath), "%s/model.mps.gz", directory);
    CHECK(read_mps(SOURCE_DIR "/qp/examples/qp_fixed.mps", &p, msg, sizeof(msg)) == OK);
    CHECK(write_mps(path, &p, msg, sizeof(msg)) == OK);
    CHECK(write_mps(path, &p, msg, sizeof(msg)) == IO_ERROR);
    CHECK(read_mps(path, &q, msg, sizeof(msg)) == OK);
    compare_models(&p, &q, 1e-12);
    problem_free(&q);
    FILE *f = fopen(path, "rb");
    gzFile gz = gzopen(gzpath, "wb");
    CHECK(f && gz);
    if (f && gz) {
        char buffer[1024];
        size_t got;
        while ((got = fread(buffer, 1, sizeof(buffer), f)))
            CHECK(gzwrite(gz, buffer, (unsigned)got) == (int)got);
        fclose(f);
        CHECK(gzclose(gz) == Z_OK);
    }
    CHECK(read_mps(gzpath, &q, msg, sizeof(msg)) == OK);
    compare_models(&p, &q, 1e-12);
    problem_free(&p);
    problem_free(&q);
    unlink(path);
    unlink(gzpath);
    const char *invalid[] = {"NAME T\nOBJSENSE\n MAX\nROWS\n N O\nCOLUMNS\n X O 1\nENDATA\n",
                             "NAME T\nROWS\n N O\nCOLUMNS\n M 'MARKER' 'INTORG'\nENDATA\n",
                             "NAME T\nROWS\n N O\nCOLUMNS\n X O 1\nBOUNDS\n BV B X\nENDATA\n",
                             "NAME T\nROWS\n N O\nCOLUMNS\n X BAD 1\nENDATA\n",
                             "NAME T\nROWS\n N O\nCOLUMNS\n X O NaN\nENDATA\n",
                             "NAME T\nROWS\n N O\nCOLUMNS\n X O 1\nSOS\nENDATA\n"};
    for (size_t i = 0; i < sizeof(invalid) / sizeof(*invalid); i++) {
        write_text(path, invalid[i]);
        CHECK(read_mps(path, &p, msg, sizeof(msg)) == INVALID_ARGUMENT);
        problem_free(&p);
        unlink(path);
    }
    write_text(
        path,
        "NAME T\nROWS\n N O\n L R\nCOLUMNS\n X O 1 R 2\n Y O 0 R -1\nRHS\n RHS R 5 O -2\nRANGES\n "
        "RNG R 3\nBOUNDS\n FR B X\n UP B Y -1\nQUADOBJ\n X X 2\n X Y .5\n Y Y 3\nENDATA\n");
    CHECK(read_mps(path, &p, msg, sizeof(msg)) == OK);
    CHECK(p.row_lower[0] == 2. && p.row_upper[0] == 5. && p.objective_offset == 2.);
    CHECK(get(&p.Q, 1, 0) == .5 && get(&p.Q, 0, 1) == .5);
    CHECK(p.lower[1] == -INFINITY);
    unlink(path);
    CHECK(write_mps(path, &p, msg, sizeof(msg)) == OK);
    CHECK(read_mps(path, &q, msg, sizeof(msg)) == OK);
    compare_models(&p, &q, 1e-12);
    problem_free(&p);
    problem_free(&q);
    unlink(path);
    write_text(
        path,
        "NAME KEYWORDS\nROWS\n N OBJ\nCOLUMNS\nRHS OBJ 1\nHAS_MARKER OBJ 0\nSMALL OBJ 0\nBOUNDS\n "
        "FR B RHS\n FR B HAS_MARKER\n FR B SMALL\nQMATRIX\n RHS RHS 2\n RHS HAS_MARKER .5\n "
        "HAS_MARKER RHS .5\n HAS_MARKER HAS_MARKER 3\n SMALL SMALL 1e-320\nENDATA\n");
    CHECK(read_mps(path, &p, msg, sizeof(msg)) == OK);
    CHECK(p.A.cols == 3 && p.c[0] == 1. && get(&p.Q, 0, 1) == .5 && get(&p.Q, 1, 0) == .5);
    CHECK(get(&p.Q, 2, 2) > 0. && get(&p.Q, 2, 2) < DBL_MIN);
    problem_free(&p);
    unlink(path);
    Problem empty = {0};
    CHECK(problem_alloc(&empty, 0, 0) == OK);
    empty.objective_offset = 2.;
    CHECK(write_mps(path, &empty, msg, sizeof(msg)) == OK);
    CHECK(read_mps(path, &q, msg, sizeof(msg)) == OK);
    compare_models(&empty, &q, 1e-12);
    problem_free(&q);
    problem_free(&empty);
    unlink(path);
    rmdir(directory);
}
int main(int argc, char **argv) {
    if (argc != 2 || strcmp(argv[1], "--gpu"))
        return 2;
    if (backend == CUDA && !cuda_available(0)) {
        fprintf(stderr, "GPU tests requested but CUDA is unavailable.\n");
        return 1;
    }
    test_examples();
    test_rules();
    test_degenerate_oracle();
    test_numerics();
    test_contract_edges();
    test_random();
    test_random_inequalities();
    test_mps();
    printf("backend=%s assertions=%d failures=%d\n", backend == CUDA ? "CUDA" : "CPU",
           assertions, failures);
    return failures ? 1 : 0;
}
