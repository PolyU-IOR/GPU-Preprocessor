#ifdef __APPLE__
#define _DARWIN_C_SOURCE
#endif
#define _POSIX_C_SOURCE 200809L
#include "internal.h"
#include <unistd.h>
#include <zlib.h>

static int checks, failures;
static Backend backend;
#define CHECK(x)                                                                                   \
    do {                                                                                           \
        checks++;                                                                                  \
        if (!(x)) {                                                                                \
            failures++;                                                                            \
            fprintf(stderr, "audit FAIL line %d: %s\n", __LINE__, #x);                             \
        }                                                                                          \
    } while (0)
static void matrix(Matrix *m, int rows, int cols, int nnz, const int *r, const int *c,
                   const double *v) {
    CHECK(matrix_from_triplets(m, rows, cols, nnz, r, c, v) == OK);
}
static Config config(uint32_t rules) {
    Config c = default_config();
    c.backend = backend;
    c.rules = rules;
    return c;
}
static void rule_masks_and_diagnostics(void) {
    Problem p = {0};
    Result out = {0};
    CHECK(problem_alloc(&p, 0, 2) == OK);
    p.lower[0] = p.upper[0] = 1.;
    p.lower[1] = p.upper[1] = -2.;
    Config c = config(FIXED_COLS);
    CHECK(default_config().rules == ALL_RULES);
    CHECK(presolve(&p, &c, &out) == OK);
    RuleInfo info;
    for (uint32_t bit = 1; bit <= FOLDING; bit <<= 1) {
        CHECK(rule_info(out.state, bit, &info) == OK);
        if (bit == FIXED_COLS)
            CHECK(info.reductions == 2 && info.calls > 0 && info.seconds >= 0.);
        else
            CHECK(info.reductions == 0 && info.calls == 0 && info.seconds == 0.);
    }
    CHECK(rule_info(out.state, 0, &info) == INVALID_ARGUMENT);
    CHECK(rule_info(out.state, EMPTY_ROWS | FIXED_COLS, &info) == INVALID_ARGUMENT);
    const uint32_t unsupported[] = {1u << 13, 1u << 14, 1u << 15, UINT32_MAX};
    for (size_t i = 0; i < sizeof(unsupported) / sizeof(*unsupported); ++i)
        CHECK(rule_info(out.state, unsupported[i], &info) == INVALID_ARGUMENT);
    result_free(&out);
    for (size_t i = 0; i < sizeof(unsupported) / sizeof(*unsupported); ++i) {
        c.rules = FIXED_COLS | unsupported[i];
        CHECK(presolve(&p, &c, &out) == INVALID_ARGUMENT);
        CHECK(out.state == NULL);
        result_free(&out);
    }
    problem_free(&p);
}
static void numerical_cases(void) {
    Problem p = {0};
    Result out = {0};
    Config c = config(ALL_RULES);
    int r[] = {0}, col[] = {0}, ar[] = {0, 0}, ac[] = {0, 1};
    double q[] = {1e-300}, av[] = {1, 1e-100};
    CHECK(problem_alloc(&p, 1, 2) == OK);
    matrix(&p.Q, 2, 2, 1, r, col, q);
    matrix(&p.A, 1, 2, 2, ar, ac, av);
    p.row_lower[0] = p.row_upper[0] = 0.;
    p.c[1] = 1.;
    CHECK(presolve(&p, &c, &out) == NUMERICAL_ERROR);
    CHECK(!out.state);
    result_free(&out);
    problem_free(&p);
    CHECK(problem_alloc(&p, 1, 2) == OK);
    double av2[] = {1e100, 1e-300};
    matrix(&p.A, 1, 2, 2, ar, ac, av2);
    p.row_lower[0] = p.row_upper[0] = 0.;
    c = config(SINGLETON_EQ);
    CHECK(presolve(&p, &c, &out) == NUMERICAL_ERROR);
    result_free(&out);
    problem_free(&p);
    CHECK(problem_alloc(&p, 0, 0) == OK);
    c = config(0);
    c.max_seconds = 0.;
    CHECK(presolve(&p, &c, &out) == TIME_LIMIT);
    CHECK(out.state != NULL);
    result_free(&out);
    problem_free(&p);
    double product, factors[] = {1e-300, 1e-100, 1e100};
    CHECK(checked_product(factors, 3, &product));
    CHECK(fabs(product / 1e-300 - 1.) < 1e-14);
    CHECK(problem_alloc(&p, 0, 1) == OK);
    double large[] = {1e308}, x[] = {1e308}, z[] = {0};
    matrix(&p.Q, 1, 1, 1, r, col, large);
    KKT k;
    CHECK(compute_kkt_backend(&p, x, NULL, z, backend, 0, &k) == NUMERICAL_ERROR);
    CHECK(isinf(k.dual_feasibility) && isinf(k.relative_gap));
    CHECK(compute_kkt_backend(&p, x, NULL, z, CPU, -1, &k) == INVALID_ARGUMENT);
    problem_free(&p);
    /* Independent cancellation fixture: the true row activity is negative,
       although ordinary double accumulation overflows its positive prefix. */
    CHECK(problem_alloc(&p, 1, 5) == OK);
    int ir[] = {0, 0, 0, 0, 0}, ic[] = {0, 1, 2, 3, 4};
    double iv[] = {1e308, 1e308, 1e308, -1e308, -1e308};
    matrix(&p.A, 1, 5, 5, ir, ic, iv);
    for (int j = 0; j < 5; j++)
        p.lower[j] = p.upper[j] = j < 3 ? 1. : 1.7;
    /* Exercise the production CUDA activity path. This box has a
     * feasible activity near -4e307 despite the overflowing positive prefix. */
    p.row_lower[0] = -5e307;
    p.row_upper[0] = -3e307;
    c = config(ACTIVITY);
    CHECK(presolve(&p, &c, &out) == OK);
    result_free(&out);
    problem_free(&p);
    CHECK(problem_alloc(&p, 1, 2) == OK);
    double av3[] = {1e308, 1.};
    matrix(&p.A, 1, 2, 2, ar, ac, av3);
    p.lower[0] = 0.;
    p.upper[0] = 10.;
    p.lower[1] = p.upper[1] = -1e308;
    p.row_lower[0] = 1e308;
    c = config(PROPAGATION);
    CHECK(presolve(&p, &c, &out) == OK);
    CHECK(out.reduced.lower[0] <= 2. && out.reduced.upper[0] >= 2.);
    result_free(&out);
    problem_free(&p);
}
static void cancellation_cases(void) {
    Problem p = {0};
    Result r = {0};
    Config c = config(ALL_RULES);
    CHECK(problem_alloc(&p, 0, 1) == OK);
    int rows[] = {0, 0, 0}, cols[] = {0, 0, 0};
    double values[] = {-1e20, 1., 1e20};
    matrix(&p.Q, 1, 1, 3, rows, cols, values);
    CHECK(p.Q.nnz == 1 && p.Q.values[0] == 1.);
    p.c[0] = -1.;
    CHECK(presolve(&p, &c, &r) == OK);
    CHECK(r.reduced.objective_offset == -.5);
    result_free(&r);
    problem_free(&p);
    CHECK(problem_alloc(&p, 1, 1) == OK);
    double one[] = {1.};
    matrix(&p.Q, 1, 1, 1, rows, cols, one);
    matrix(&p.A, 1, 1, 1, rows, cols, one);
    p.c[0] = 1.;
    p.row_lower[0] = 0.;
    p.row_upper[0] = 1.;
    c = config(SINGLETON_ROWS | EMPTY_COLS);
    CHECK(presolve(&p, &c, &r) == OK);
    CHECK(r.reduced.objective_offset == 0.);
    result_free(&r);
    problem_free(&p);
}
static void boundary_duals(void) {
    Problem p = {0};
    Result out = {0};
    Config c = config(SINGLETON_DUAL | SINGLETON_EQ);
    CHECK(problem_alloc(&p, 1, 2) == OK);
    int qr[] = {1}, qc[] = {1}, ar[] = {0, 0}, ac[] = {0, 1};
    double q[] = {2}, a[] = {1, 1};
    matrix(&p.Q, 2, 2, 1, qr, qc, q);
    matrix(&p.A, 1, 2, 2, ar, ac, a);
    p.c[0] = 1.;
    p.lower[0] = 0.;
    p.upper[0] = 1.;
    p.lower[1] = p.upper[1] = 0.;
    p.row_lower[0] = 0.;
    p.row_upper[0] = 1.;
    CHECK(presolve(&p, &c, &out) == OK);
    CHECK(out.reduced.A.cols == 2 && out.reduced.row_upper[0] == 1.);
    if (out.reduced.A.cols == 2) {
        double xr[] = {0, 0}, yr[] = {0}, zr[] = {1, 0}, x[2], y[1], z[2];
        KKT k;
        CHECK(postsolve(out.state, xr, 2, yr, 1, zr, 2, x, y, z) == OK);
        CHECK(compute_kkt(&p, x, y, z, &k) == OK);
        CHECK(k.relative_gap == 0. && k.dual_feasibility == 0.);
    }
    result_free(&out);
    problem_free(&p);
}
static void bound_pivot_residue(void) {
    for (int rule = 0; rule < 2; ++rule)
        for (int sign_a = -1; sign_a <= 1; sign_a += 2)
            for (int sign_c = -1; sign_c <= 1; sign_c += 2) {
                Problem p = {0};
                Result r = {0};
                CHECK(problem_alloc(&p, 1, 1) == OK);
                int row[] = {0}, col[] = {0};
                double a[] = {.3 * sign_a}, cost = .7 * sign_c;
                matrix(&p.A, 1, 1, 1, row, col, a);
                p.c[0] = cost;
                if (sign_a == sign_c)
                    p.row_lower[0] = a[0];
                else
                    p.row_upper[0] = a[0];
                Config cfg = config(rule ? SINGLETON_ROWS : PROPAGATION);
                cfg.max_iterations = 1;
                CHECK(presolve(&p, &cfg, &r) == OK);
                double xr[] = {1.}, yr[] = {0.}, zr[] = {cost}, x[1], y[1], z[1];
                CHECK(postsolve(r.state, xr, 1, yr, r.reduced.A.rows, zr, 1, x, y, z) ==
                      OK);
                CHECK(z[0] == 0. && fabs(y[0] - cost / a[0]) < 1e-14);
                KKT k;
                CHECK(compute_kkt(&p, x, y, z, &k) == OK);
                CHECK(k.primal_feasibility < 1e-14 && k.dual_feasibility < 1e-14 &&
                      k.relative_gap < 1e-14);
                result_free(&r);
                problem_free(&p);
            }
}
static void text(const char *path, const char *s) {
    FILE *f = fopen(path, "wx");
    CHECK(f != NULL);
    if (!f)
        return;
    CHECK(fputs(s, f) >= 0);
    CHECK(!fclose(f));
}
static void parser_cases(void) {
    char tmp[] = "/tmp/presolve-audit-XXXXXX";
    CHECK(mkdtemp(tmp) != NULL);
    char path[512], msg[256];
    snprintf(path, sizeof(path), "%s/probe.mps", tmp);
    Problem p = {0};
    text(path,
         "NAME V\nROWS\n N OBJ\n E R1\n E R2\nCOLUMNS\n X OBJ 0 R1 1\n X R2 1\nRHS\n FIRST R1 1 R2 "
         "2\n SECOND R1 9\n R2 8\nRANGES\n FIRST R1 2 R2 3\n SECOND R1 9\n R2 8\nENDATA\n");
    CHECK(read_mps(path, &p, msg, sizeof(msg)) == OK);
    CHECK(p.row_lower[0] == 1. && p.row_lower[1] == 2. && p.row_upper[0] == 3. &&
          p.row_upper[1] == 5.);
    problem_free(&p);
    unlink(path);
    text(path, "NAME C\nROWS\n N OBJ\nCOLUMNS\n X OBJ -1e20\n X OBJ 1\n X OBJ 1e20\n Y OBJ "
               "0\nQSECTION OBJ\n X X 2 Y .5\n Y Y 3\nENDATA\n");
    CHECK(read_mps(path, &p, msg, sizeof(msg)) == OK);
    CHECK(p.c[0] == 1. && matrix_get(&p.Q, 0, 1) == .5);
    problem_free(&p);
    unlink(path);
    const char *invalid[] = {
        "NAME T\nROWS\n N OBJ\n L R\nCOLUMNS\n X OBJ 0 R 1\nQSECTION R\n X X 2\nENDATA\n",
        "NAME T\nOBJNAME MISSING\nROWS\n N OBJ\nCOLUMNS\n X MISSING 1\nENDATA\n",
        "NAME T\nOBJSENSE\nROWS\n N OBJ\nCOLUMNS\n X OBJ 1\nENDATA\n",
        "NAME T\nROWS\n N OBJ\n N OBJ\nCOLUMNS\n X OBJ 1\nENDATA\n",
        "NAME T\nROWS\n N OBJ\nCOLUMNS\n X OBJ 1\nQUADOBJ\n X X 2\nQMATRIX\n X X 3\nENDATA\n"};
    for (size_t i = 0; i < sizeof(invalid) / sizeof(*invalid); i++) {
        text(path, invalid[i]);
        CHECK(read_mps(path, &p, msg, sizeof(msg)) == INVALID_ARGUMENT);
        problem_free(&p);
        unlink(path);
    }
    text(path, "NAME T\nROWS\n N OBJ\n L R\nCOLUMNS\n X OBJ 0 R 1\nRHS\n V R -1e308\nRANGES\n V R "
               "1e308\nENDATA\n");
    CHECK(read_mps(path, &p, msg, sizeof(msg)) == NUMERICAL_ERROR);
    unlink(path);
    CHECK(problem_alloc(&p, 1, 0) == OK);
    p.row_lower[0] = -1e308;
    p.row_upper[0] = 1e308;
    CHECK(write_mps(path, &p, msg, sizeof(msg)) == NUMERICAL_ERROR);
    CHECK(access(path, F_OK) != 0);
    p.row_lower[0] = 2.;
    p.row_upper[0] = 1.;
    CHECK(write_mps(path, &p, msg, sizeof(msg)) == INVALID_ARGUMENT);
    CHECK(access(path, F_OK) != 0);
    problem_free(&p);
    const char *valid = "NAME T\nROWS\n N OBJ\nCOLUMNS\n X OBJ 0\nENDATA\n";
    for (int mode = 0; mode < 2; mode++) {
        gzFile gz = gzopen(path, "wb");
        CHECK(gz != NULL);
        if (!gz)
            continue;
        CHECK(gzputs(gz, valid) > 0);
        CHECK(gzclose(gz) == Z_OK);
        FILE *f = fopen(path, "r+b");
        CHECK(f != NULL);
        if (!f)
            continue;
        CHECK(fseek(f, -8, SEEK_END) == 0);
        long trailer = ftell(f);
        if (mode == 0) {
            int b = fgetc(f);
            CHECK(b != EOF);
            CHECK(fseek(f, trailer, SEEK_SET) == 0);
            CHECK(fputc(b ^ 0xff, f) != EOF);
        }
        CHECK(fclose(f) == 0);
        if (mode == 1)
            CHECK(truncate(path, trailer + 3) == 0);
        CHECK(read_mps(path, &p, msg, sizeof(msg)) == IO_ERROR);
        problem_free(&p);
        unlink(path);
    }
    CHECK(rmdir(tmp) == 0);
}
int main(int argc, char **argv) {
    if (argc != 2 || strcmp(argv[1], "--gpu"))
        return 2;
    backend = CUDA;
    if (backend == CUDA && !cuda_available(0))
        return 1;
    rule_masks_and_diagnostics();
    numerical_cases();
    cancellation_cases();
    boundary_duals();
    bound_pivot_residue();
    parser_cases();
    printf("audit backend=%s assertions=%d failures=%d\n", backend == CUDA ? "CUDA" : "CPU",
           checks, failures);
    return failures ? 1 : 0;
}
