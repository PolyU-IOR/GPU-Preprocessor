#include "internal.h"
#include <cuda_profiler_api.h>
#include <cuda_runtime.h>
#include <vector>
#include "backend/cuda_common.cuh"
#include "presolve/cuda_row_activity.cuh"
#include "backend/cuda_download.cuh"
static int checks, failures;
static bool profile_postsolve = false;
static bool profile_device_input = false;
#define CHECK(e)                                                                                   \
    do {                                                                                           \
        ++checks;                                                                                  \
        if (!(e)) {                                                                                \
            ++failures;                                                                            \
            fprintf(stderr, "device resident FAIL %d: %s\n", __LINE__, #e);                        \
        }                                                                                          \
    } while (0)
#define CUDA(e)                                                                                    \
    do {                                                                                           \
        cudaError_t code = (e);                                                                    \
        if (code != cudaSuccess) {                                                                 \
            fprintf(stderr, "CUDA FAIL %d: %s\n", __LINE__, cudaGetErrorString(code));             \
            exit(2);                                                                               \
        }                                                                                          \
    } while (0)
static void matrix(Matrix *out, int m, int n, COO *coo) {
    CHECK(coo_finish(coo, m, n, out) == OK);
}
static void device_pointer(const void *p) {
    cudaPointerAttributes a = {};
    CUDA(cudaPointerGetAttributes(&a, p));
    CHECK(a.type == cudaMemoryTypeDevice);
}
__global__ static void check_chain_view(DeviceProblemView view, int *error) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < view.A.cols && (view.upper[j] != 0. || view.lower[j] != -1. || view.c[j] != 0.))
        atomicExch(error, 1);
}
static void contract(Problem *p, Result *r) {
    CHECK(!r->reduced.A.row_ptr && !r->reduced.Q.row_ptr && !r->reduced.c);
    CHECK(!r->state->records && !r->state->row_ids && !r->state->col_ids);
    ResidencyInfo info = {};
    CHECK(residency_info(r->state, &info) == OK && info.resident);
    CHECK(!info.model_export_bytes && !info.tape_export_bytes);
    CHECK(info.iterations.model_uploads == 1 && info.iterations.downloaded_candidates == 0);
    uint64_t initial = ((uint64_t)p->A.rows + p->A.cols + 2) * sizeof(int) +
                       ((uint64_t)p->A.nnz + p->Q.nnz) * (sizeof(int) + sizeof(double)) +
                       ((uint64_t)3 * p->A.cols + 2 * p->A.rows + 1) * sizeof(double);
    CHECK(info.iterations.host_to_device_bytes == initial);
    ExecutionInfo execution;
    CHECK(execution_info(r->state, &execution) == OK && !execution.cpu_fallbacks);
    DeviceProblemView view = {};
    CHECK(device_problem_view(r->state, &view) == OK);
    CHECK(view.A.rows == r->reduced.A.rows && view.A.cols == r->reduced.A.cols);
    device_pointer(view.A.row_ptr);
    device_pointer(view.Q.row_ptr);
    device_pointer(view.AT.row_ptr);
    device_pointer(view.objective_offset);
    printf("resident records=%zu payload=%zu presolve_h2d=%llu control_d2h=%llu\n",
           info.tape_records, info.tape_entries,
           (unsigned long long)info.iterations.host_to_device_bytes,
           (unsigned long long)info.iterations.device_to_host_bytes);
}
static void recover(Problem *p, Result *r, const std::vector<double> &reduced_x,
                    const std::vector<double> &expected,
                    const std::vector<double> &reduced_z = {}) {
    int n = p->A.cols, m = p->A.rows, nr = r->reduced.A.cols, mr = r->reduced.A.rows;
    double *input = nullptr, *output = nullptr;
    CUDA(cudaMalloc((void **)&input, ((size_t)2 * nr + mr + 1) * sizeof(double)));
    CUDA(cudaMalloc((void **)&output, ((size_t)2 * n + m + 1) * sizeof(double)));
    CUDA(cudaMemset(output, 0, ((size_t)2 * n + m + 1) * sizeof(double)));
    CUDA(cudaMemset(input, 0, ((size_t)2 * nr + mr + 1) * sizeof(double)));
    if (nr)
        CUDA(cudaMemcpy(input, reduced_x.data(), (size_t)nr * sizeof(double),
                        cudaMemcpyHostToDevice));
    if (!reduced_z.empty())
        CUDA(cudaMemcpy(input + nr, reduced_z.data(), (size_t)nr * sizeof(double),
                        cudaMemcpyHostToDevice));
    PostsolveWorkspace *w = nullptr;
    if (profile_postsolve)
        cudaProfilerStart();
    CHECK(postsolve_workspace_create(r->state, &w) == OK);
    CHECK(w && !w->data);
    CHECK(!r->state->records && !r->state->row_ids);
    if (nr) {
        double host[1] = {0.};
        CHECK(postsolve_device(r->state, w, host, nr, input + 2 * nr, mr, input + nr, nr,
                                   output, output + 2 * n, output + n) == INVALID_ARGUMENT);
        CHECK(postsolve_device(r->state, w, input, nr + 1, input + 2 * nr, mr, input + nr, nr,
                                   output, output + 2 * n, output + n) == INVALID_ARGUMENT);
    }
    std::vector<double> values((size_t)2 * n + m + 1);
    for (int repeat = 0; repeat < 3; ++repeat) {
        CHECK(postsolve_device(r->state, w, input, nr, input + 2 * nr, mr, input + nr, nr,
                                   output, output + 2 * n, output + n) == OK);
        if (profile_postsolve && !repeat)
            cudaProfilerStop();
        CHECK(!w->data);
        CUDA(cudaMemcpy(values.data(), output, values.size() * sizeof(double),
                        cudaMemcpyDeviceToHost));
        for (int j = 0; j < n; ++j)
            CHECK(fabs(values[j] - expected[j]) < 1e-12);
        KKT kkt;
        CHECK(compute_kkt(p, values.data(), values.data() + 2 * n, values.data() + n, &kkt) ==
              OK);
        CHECK(kkt.primal_feasibility < 1e-12 && kkt.dual_feasibility < 1e-12 &&
              kkt.relative_gap < 1e-12);
    }
    /* An independent scalar replay checks every x/y/z coordinate, including
     * nonzero bound transfers and component/segment boundaries. */
    PostsolveWorkspace *reference = nullptr;
    CHECK(postsolve_workspace_create_backend(r->state, CPU, 0, &reference) == OK);
    std::vector<double> cpu(values.size()), yr((size_t)mr, 0.), zr((size_t)nr, 0.);
    if (!reduced_z.empty())
        zr = reduced_z;
    CHECK(postsolve_with_workspace(r->state, reference, reduced_x.data(), nr, yr.data(), mr,
                                       zr.data(), nr, cpu.data(), cpu.data() + 2 * n,
                                       cpu.data() + n) == OK);
    for (size_t i = 0; i < (size_t)2 * n + m; ++i)
        CHECK(cpu[i] == values[i]);
    CHECK(!r->state->records);
    postsolve_workspace_free(&reference);
    ResidencyInfo before = {}, after = {};
    CHECK(residency_info(r->state, &before) == OK);
    CHECK(!before.model_export_bytes && !before.tape_export_bytes && !r->state->records);
    CHECK(result_download(r) == OK);
    CHECK(validate(&r->reduced, nullptr, 0) == OK);
    CHECK(residency_info(r->state, &after) == OK);
    CHECK(after.model_export_bytes > 0 && !after.tape_export_bytes);
    CHECK(result_download(r) == OK);
    CHECK(residency_info(r->state, &before) == OK &&
          before.model_export_bytes == after.model_export_bytes);
    CHECK(state_export_maps(r->state) == OK && !r->state->records);
    CHECK(state_export_host(r->state) == OK);
    CHECK(r->state->records != nullptr);
    CUDA(cudaFree(input));
    CUDA(cudaFree(output));
    result_free(r); /* A workspace owns its shared tape independently. */
    postsolve_workspace_free(&w);
}
static void chain(void) {
    enum { n = 128, m = n - 1 };
    Problem p = {};
    COO a = {}, q = {};
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
    matrix(&p.A, m, n, &a);
    matrix(&p.Q, n, n, &q);
    Config cfg = default_config();
    cfg.backend = CUDA;
    cfg.rules = PROPAGATION;
    Result r = {};
    if (!profile_postsolve && !profile_device_input)
        cudaProfilerStart();
    Status status = presolve_resident(&p, &cfg, nullptr, &r);
    if (!profile_postsolve && !profile_device_input)
        cudaProfilerStop();
    CHECK(status == OK);
    if (status) {
        result_free(&r);
        problem_free(&p);
        return;
    }
    contract(&p, &r);
    CHECK(r.reductions == m && p.upper[0] == 1.);
    DeviceProblemView view;
    CHECK(device_problem_view(r.state, &view) == OK);
    int *error = nullptr, host = 0;
    CUDA(cudaMalloc((void **)&error, sizeof(int)));
    CUDA(cudaMemset(error, 0, sizeof(int)));
    check_chain_view<<<1, 256>>>(view, error);
    CUDA(cudaMemcpy(&host, error, sizeof(int), cudaMemcpyDeviceToHost));
    CHECK(!host);
    CUDA(cudaFree(error));
    double *original_upper = nullptr;
    CUDA(cudaMalloc((void **)&original_upper, n * sizeof(double)));
    CUDA(cudaMemcpy(original_upper, p.upper, n * sizeof(double), cudaMemcpyHostToDevice));
    DeviceProblemView input_view = view;
    input_view.upper = original_upper;
    Result copy = {};
    if (profile_device_input)
        cudaProfilerStart();
    CHECK(presolve_device(&input_view, &cfg, nullptr, &copy) == OK);
    if (profile_device_input)
        cudaProfilerStop();
    ResidencyInfo copy_info = {};
    CHECK(residency_info(copy.state, &copy_info) == OK && copy_info.resident);
    CHECK(!copy_info.iterations.host_to_device_bytes && !copy_info.iterations.model_uploads);
    CHECK(!copy_info.model_export_bytes && !copy_info.tape_export_bytes && copy.reductions == m);
    CHECK(!copy.reduced.A.row_ptr && !copy.state->records);
    result_free(&copy);
    double unchanged;
    CUDA(cudaMemcpy(&unchanged, original_upper, sizeof(double), cudaMemcpyDeviceToHost));
    CHECK(unchanged == 1.);
    CUDA(cudaFree(original_upper));
    DeviceProblemView invalid = view;
    invalid.A.row_ptr = p.A.row_ptr;
    CHECK(presolve_device(&invalid, &cfg, nullptr, &copy) == INVALID_ARGUMENT);
    CHECK(!copy.state);
    result_free(&copy);
    recover(&p, &r, std::vector<double>(n, 0.), std::vector<double>(n, 0.));
    problem_free(&p);
}
static void doubleton(int retained) {
    Problem p = {};
    COO a = {}, q = {};
    CHECK(problem_alloc(&p, 1, 3) == OK);
    coo_add(&a, 0, 0, 1.);
    coo_add(&a, 0, 1, 1.);
    coo_add(&q, 1, 1, 2.);
    coo_add(&q, 2, 2, 2.);
    p.c[1] = -4.;
    p.row_lower[0] = p.row_upper[0] = 3.;
    if (retained) {
        p.lower[0] = 0.;
        p.upper[0] = 5.;
    }
    matrix(&p.A, 1, 3, &a);
    matrix(&p.Q, 3, 3, &q);
    Config cfg = default_config();
    cfg.backend = CUDA;
    cfg.rules = DOUBLETON_EQ;
    Result r = {};
    CHECK(presolve_resident(&p, &cfg, nullptr, &r) == OK);
    contract(&p, &r);
    CHECK(r.reduced.A.rows == retained && r.reduced.A.cols == 2);
    recover(&p, &r, {2., 0.}, {1., 2., 0.});
    problem_free(&p);
}
static void empty(void) {
    Problem p = {};
    Result r = {};
    CHECK(problem_alloc(&p, 0, 0) == OK);
    Config cfg = default_config();
    cfg.backend = CUDA;
    CHECK(presolve_resident(&p, &cfg, nullptr, &r) == OK);
    contract(&p, &r);
    PostsolveWorkspace *w = nullptr;
    CHECK(postsolve_workspace_create(r.state, &w) == OK);
    CHECK(postsolve_device(r.state, w, nullptr, 0, nullptr, 0, nullptr, 0, nullptr, nullptr,
                               nullptr) == OK);
    CHECK(state_export_host(r.state) == OK);
    CHECK(result_download(&r) == OK);
    result_free(&r);
    postsolve_workspace_free(&w);
    problem_free(&p);
}
static void segmented_bounds(void) {
    enum { n = 96, m = n - 1 };
    Problem p = {};
    COO a = {}, q = {};
    CHECK(problem_alloc(&p, m, n) == OK);
    for (int j = 0; j < n; ++j) {
        p.lower[j] = j == 0 ? 0. : -1.;
        p.upper[j] = j == n - 1 ? 0. : 1.;
        p.c[j] = -1.;
        coo_add(&q, j, j, 2.);
        if (j + 1 < n) {
            coo_add(&a, j, j, 1.);
            coo_add(&a, j, j + 1, -1.);
            p.row_upper[j] = 0.;
        }
    }
    matrix(&p.A, m, n, &a);
    matrix(&p.Q, n, n, &q);
    Config cfg = default_config();
    cfg.backend = CUDA;
    cfg.rules = PROPAGATION;
    ExecutionOptions options = default_execution_options();
    options.batch_limit = 1;
    Result r = {};
    CHECK(presolve_resident(&p, &cfg, &options, &r) == OK);
    contract(&p, &r);
    CHECK(r.state->count > n); /* More than one owning segment, with shared dual writes. */
    recover(&p, &r, std::vector<double>(n, 0.), std::vector<double>(n, 0.),
            std::vector<double>(n, -1.));
    problem_free(&p);
}
static void independent_chains(void) {
    enum { groups = 64, length = 32, n = groups * length, m = groups * (length - 1) };
    Problem p = {};
    COO a = {}, q = {};
    CHECK(problem_alloc(&p, m, n) == OK);
    for (int g = 0; g < groups; ++g)
        for (int k = 0; k < length; ++k) {
            int j = g * length + k;
            p.lower[j] = k ? -1. : 0.;
            p.upper[j] = k + 1 == length ? 0. : 1.;
            p.c[j] = -1.;
            coo_add(&q, j, j, 2.);
            if (k + 1 < length) {
                int row = g * (length - 1) + k;
                coo_add(&a, row, j, 1.);
                coo_add(&a, row, j + 1, -1.);
                p.row_upper[row] = 0.;
            }
        }
    matrix(&p.A, m, n, &a);
    matrix(&p.Q, n, n, &q);
    Config cfg = default_config();
    cfg.backend = CUDA;
    cfg.rules = PROPAGATION;
    Result r = {};
    CHECK(presolve_resident(&p, &cfg, nullptr, &r) == OK);
    contract(&p, &r);
    CHECK(r.state->count > n && r.reductions == 2 * m);
    recover(&p, &r, std::vector<double>(n, 0.), std::vector<double>(n, 0.),
            std::vector<double>(n, -1.));
    problem_free(&p);
}

__global__ static void activity_exclusions(const RowActivity *cached, const double *a,
                                          const double *l, const double *u, int n,
                                          double *lo, double *hi) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < n)
        excluded_row_activity(*cached, a[j], l[j], u[j], lo[j], hi[j]);
}
static void activity_enclosures(int n, int mode) {
    std::vector<int> ptr{0,n}, col(n);
    std::vector<double> a(n), l(n), u(n), lo(n), hi(n);
    for (int j = 0; j < n; ++j) {
        col[j] = j;
        a[j] = ldexp(j % 2 ? -1. : 1., j % 21 - 10);
        l[j] = -ldexp(1., j % 11 - 5); u[j] = -l[j];
        if (mode == 4) { a[j] = j % 2 ? -1e308 : 1e308; l[j] = u[j] = 1.7; }
        if (mode == 5) { a[j] = j % 2 ? -1e-300 : 1e-300; l[j] = 1e-100; u[j] = 2e-100; }
    }
    if (mode == 1 || mode == 2) { l[0] = -INFINITY; u[0] = INFINITY; }
    if (mode == 2) { l[1] = -INFINITY; u[1] = INFINITY; }
    if (mode == 3) { l[1] = -INFINITY; u[2] = INFINITY; }
    Buffer<int> dp, dc; Buffer<double> da, dl, du, drl, dru, dlo, dhi;
    Buffer<RowActivity> cached;
    const double rl = -1., ru = 1.;
    CUDA(dp.upload(ptr.data(), 2)); CUDA(dc.upload(col.data(), n));
    CUDA(da.upload(a.data(), n)); CUDA(dl.upload(l.data(), n)); CUDA(du.upload(u.data(), n));
    CUDA(drl.upload(&rl, 1)); CUDA(dru.upload(&ru, 1));
    CUDA(cached.reserve(1)); CUDA(dlo.reserve(n)); CUDA(dhi.reserve(n));
    View v{1,n,dp.data,dc.data,da.data};
    cache_row_activity<<<1,256>>>(v, dl.data, du.data, drl.data, dru.data, cached.data);
    activity_exclusions<<<(n + 255) / 256,256>>>(cached.data, da.data, dl.data, du.data,
                                               n, dlo.data, dhi.data);
    CUDA(cudaGetLastError()); CUDA(dlo.download(lo.data(), n)); CUDA(dhi.download(hi.data(), n));
    for (int j = 0; j < n; ++j) {
        if (n > 1024 && j != 0 && j != 1 && j != n / 2 && j != n - 1) continue;
        long double exact_lo = 0., exact_hi = 0.;
        for (int k = 0; k < n; ++k) if (k != j) {
            exact_lo += (long double)a[k] * (a[k] > 0 ? l[k] : u[k]);
            exact_hi += (long double)a[k] * (a[k] > 0 ? u[k] : l[k]);
        }
        CHECK(!isnan(lo[j]) && !isnan(hi[j]));
        CHECK((long double)lo[j] <= exact_lo && (long double)hi[j] >= exact_hi);
    }
}
static void large_host_download(void) {
    const size_t bytes = (size_t)65 * 1024 * 1024 + 17;
    unsigned char *host = (unsigned char *)calloc(1, bytes), *device = nullptr;
    CHECK(host != nullptr);
    CUDA(cudaMalloc((void **)&device, bytes));
    CUDA(cudaMemset(device, 0xa5, bytes));
    CUDA(download_host_output(host, device, bytes));
    size_t bad = 0;
    for (size_t i = 0; i < bytes; ++i) bad += host[i] != 0xa5;
    CHECK(bad == 0);
    CUDA(cudaFree(device)); free(host);
}
int main(int argc, char **argv) {
    if (argc == 2 && !strcmp(argv[1], "--profile-postsolve"))
        profile_postsolve = true;
    else if (argc == 2 && !strcmp(argv[1], "--profile-device-input"))
        profile_device_input = true;
    else if (argc != 1)
        return 2;
    if (!cuda_available(0))
        return 77;
    if (!profile_postsolve && !profile_device_input) {
        large_host_download();
        for (int mode = 0; mode < 6; ++mode) {
            activity_enclosures(32, mode); activity_enclosures(33, mode);
            activity_enclosures(64, mode); activity_enclosures(257, mode);
        }
        activity_enclosures(100003, 0);
    }
    chain();
    doubleton(0);
    doubleton(1);
    segmented_bounds();
    independent_chains();
    empty();
    printf("device resident checks=%d failures=%d\n", checks, failures);
    return failures ? 1 : 0;
}
