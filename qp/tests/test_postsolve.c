#include "gpupresolve_qp.h"
#include <float.h>
#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <string.h>

static int checks, failures;
#define CHECK(expr)                                                                                \
    do {                                                                                           \
        ++checks;                                                                                  \
        if (!(expr)) {                                                                             \
            ++failures;                                                                            \
            fprintf(stderr, "postsolve FAIL line %d: %s\n", __LINE__, #expr);                      \
        }                                                                                          \
    } while (0)
static int near(double a, double b) {
    return fabs(a - b) < 1e-11 * (1. + fabs(b));
}
static Status fixture(Backend backend, Problem *p, Result *r) {
    /* Deleting a middle column and the first row makes both survivor maps
     * nontrivial. Cross terms test the fixed variable's saved QP gradient. */
    int qr[] = {0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3};
    int qc[] = {0, 1, 3, 0, 1, 2, 1, 2, 3, 0, 2, 3};
    double qv[] = {4, 1, .5, 1, 5, 1, 1, 6, .2, .5, .2, 7};
    int ar[] = {1, 1}, ac[] = {0, 2};
    double av[] = {1, 1}, c[] = {-7.25, -9, -20.05, -29.1};
    Status status = problem_alloc(p, 2, 4);
    if (!status)
        status = matrix_from_triplets(&p->Q, 4, 4, 12, qr, qc, qv);
    if (!status)
        status = matrix_from_triplets(&p->A, 2, 4, 2, ar, ac, av);
    if (status)
        return status;
    memcpy(p->c, c, sizeof(c));
    p->lower[1] = p->upper[1] = 2.;
    p->row_lower[1] = p->row_upper[1] = 4.;
    Config cfg = default_config();
    cfg.backend = backend;
    cfg.rules = EMPTY_ROWS | FIXED_COLS;
    return presolve(p, &cfg, r);
}
static int replay(const State *s, PostsolveWorkspace *w, double delta) {
    double xr[] = {1. + delta, 3. + 2. * delta, 4. - delta};
    double yr[] = {.75 + delta}, zr[] = {delta, -delta, 2. * delta};
    double x[4], y[2], z[4];
    if (postsolve_with_workspace(s, w, xr, 3, yr, 1, zr, 3, x, y, z))
        return 0;
    return near(x[0], xr[0]) && x[1] == 2. && near(x[2], xr[1]) && near(x[3], xr[2]) &&
           y[0] == 0. && near(y[1], yr[0]) && near(z[0], zr[0]) && near(z[1], 5. + 3. * delta) &&
           near(z[2], zr[1]) && near(z[3], zr[2]);
}
typedef struct {
    const State *state;
    int index, ok;
} Worker;
static void *worker(void *arg) {
    Worker *t = arg;
    PostsolveWorkspace *w = NULL;
    t->ok = postsolve_workspace_create(t->state, &w) == OK;
    for (int k = 0; k < 250 && t->ok; ++k)
        t->ok = replay(t->state, w, .001 * (k - 125) + t->index);
    postsolve_workspace_free(&w);
    return NULL;
}
static void workspace_cases(Backend backend) {
    Problem p = {0};
    Result r = {0};
    CHECK(fixture(backend, &p, &r) == OK);
    if (!r.state) {
        problem_free(&p);
        return;
    }
    PostsolveInfo info, after;
    CHECK(postsolve_info(r.state, &info) == OK);
    CHECK(info.original_rows == 2 && info.original_cols == 4 && info.reduced_rows == 1 &&
          info.reduced_cols == 3 && info.records == 2 && info.payload_entries > 0 &&
          info.owned_bytes > info.payload_entries * (sizeof(int) + sizeof(double)));
    PostsolveWorkspace *w = NULL;
    CHECK(postsolve_workspace_create(r.state, &w) == OK);
    CHECK(postsolve_workspace_create(r.state, &w) == INVALID_ARGUMENT);
    CHECK(replay(r.state, w, 0.));
    double x[] = {1, 3, 4, 99}, y[] = {.75, 99}, z[] = {0, 0, 0, 99};
    CHECK(postsolve_with_workspace(r.state, w, x, 3, y, 1, z, 3, x, y, z) == OK);
    KKT kkt;
    CHECK(compute_kkt(&p, x, y, z, &kkt) == OK);
    CHECK(kkt.primal_feasibility < 1e-12 && kkt.dual_feasibility < 1e-12 &&
          kkt.relative_gap < 1e-12);
    double xr[] = {1, 3, 4}, yr[] = {.75}, zr[] = {0, 0, 0};
    double bad[] = {NAN, 3, 4}, huge[] = {DBL_MAX, DBL_MAX, 4};
    double sentinel_x[] = {9, 8, 7, 6}, sentinel_y[] = {5, 4}, sentinel_z[] = {3, 2, 1, 0};
    memcpy(x, sentinel_x, sizeof(x));
    memcpy(y, sentinel_y, sizeof(y));
    memcpy(z, sentinel_z, sizeof(z));
    CHECK(postsolve_with_workspace(r.state, w, xr, 2, yr, 1, zr, 3, x, y, z) ==
          INVALID_ARGUMENT);
    CHECK(postsolve_with_workspace(r.state, w, bad, 3, yr, 1, zr, 3, x, y, z) ==
          INVALID_ARGUMENT);
    CHECK(postsolve_with_workspace(r.state, w, huge, 3, yr, 1, zr, 3, x, y, z) ==
          NUMERICAL_ERROR);
    CHECK(!memcmp(x, sentinel_x, sizeof(x)) && !memcmp(y, sentinel_y, sizeof(y)) &&
          !memcmp(z, sentinel_z, sizeof(z)));
    CHECK(replay(r.state, w, -.4));
    /* The tape owns its stage snapshots, independent of both models. */
    problem_free(&p);
    problem_free(&r.reduced);
    for (int i = 0; i < 100; ++i)
        CHECK(replay(r.state, w, .01 * i));
    CHECK(postsolve(r.state, xr, 3, yr, 1, zr, 3, x, y, z) == OK);
    pthread_t threads[4];
    Worker workers[4];
    int started = 0;
    for (int i = 0; i < 4; ++i) {
        workers[i] = (Worker){r.state, i, 0};
        int rc = pthread_create(&threads[i], NULL, worker, &workers[i]);
        CHECK(rc == 0);
        if (rc)
            break;
        ++started;
    }
    for (int i = 0; i < started; ++i) {
        CHECK(pthread_join(threads[i], NULL) == 0);
        CHECK(workers[i].ok);
    }
    CHECK(postsolve_info(r.state, &after) == OK);
    CHECK(info.records == after.records && info.payload_entries == after.payload_entries &&
          info.owned_bytes == after.owned_bytes);
    result_free(&r);
    /* Workspace can outlive its creator state and be reused for smaller states. */
    Config cfg = default_config();
    cfg.backend = backend;
    cfg.rules = 0;
    CHECK(problem_alloc(&p, 0, 1) == OK);
    CHECK(presolve(&p, &cfg, &r) == OK);
    CHECK(postsolve_with_workspace(r.state, w, xr, 1, NULL, 0, zr, 1, x, NULL, z) == OK);
    CHECK(x[0] == xr[0] && z[0] == zr[0]);
    PostsolveWorkspace *small = NULL;
    CHECK(postsolve_workspace_create(r.state, &small) == OK);
    result_free(&r);
    problem_free(&p);
    CHECK(fixture(backend, &p, &r) == OK);
    CHECK(postsolve_with_workspace(r.state, small, xr, 3, yr, 1, zr, 3, x, y, z) ==
          INVALID_ARGUMENT);
    postsolve_workspace_free(&small);
    postsolve_workspace_free(&small);
    result_free(&r);
    problem_free(&p);
    postsolve_workspace_free(&w);
    CHECK(w == NULL && small == NULL);
}
static void empty_cases(Backend backend) {
    Problem p = {0};
    Result r = {0};
    PostsolveWorkspace *w = NULL;
    Config cfg = default_config();
    cfg.backend = backend;
    CHECK(problem_alloc(&p, 0, 0) == OK);
    CHECK(presolve(&p, &cfg, &r) == OK);
    CHECK(postsolve_workspace_create(r.state, &w) == OK);
    CHECK(postsolve_with_workspace(r.state, w, NULL, 0, NULL, 0, NULL, 0, NULL, NULL, NULL) ==
          OK);
    CHECK(postsolve(NULL, NULL, 0, NULL, 0, NULL, 0, NULL, NULL, NULL) == INVALID_ARGUMENT);
    CHECK(postsolve_info(NULL, NULL) == INVALID_ARGUMENT);
    CHECK(postsolve_workspace_create(NULL, &w) == INVALID_ARGUMENT);
    postsolve_workspace_free(&w);
    postsolve_workspace_free(NULL);
    result_free(&r);
    problem_free(&p);
}
int main(int argc, char **argv) {
    if (argc != 2 || strcmp(argv[1], "--gpu"))
        return 2;
    Backend backend = CUDA;
    workspace_cases(backend);
    empty_cases(backend);
    printf("postsolve: %d checks, %d failures\n", checks, failures);
    return failures != 0;
}
