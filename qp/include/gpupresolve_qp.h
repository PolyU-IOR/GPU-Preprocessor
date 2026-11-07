#ifndef GPUPRESOLVE_QP_H
#define GPUPRESOLVE_QP_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* All sparse indices are zero-based. Matrices own their buffers and use sorted,
 * duplicate-free CSR. Q is the full symmetric Hessian, not one triangle. */
typedef struct {
    int rows, cols, nnz;
    int *row_ptr, *col_ind;
    double *values;
} Matrix;

typedef struct {
    Matrix Q, A;
    double *c, *lower, *upper, *row_lower, *row_upper;
    double objective_offset;
} Problem;

typedef enum {
    OK = 0,
    INFEASIBLE,
    UNBOUNDED,
    TIME_LIMIT,
    INVALID_ARGUMENT,
    OUT_OF_MEMORY,
    IO_ERROR,
    NUMERICAL_ERROR,
    CUDA_ERROR
} Status;

/* Presolve accepts CUDA or AUTO (both require CUDA). CPU remains available
 * only for host KKT computation and explicit recovery-tape diagnostics. */
typedef enum { CPU = 0, CUDA = 1, AUTO = 2 } Backend;

enum {
    EMPTY_ROWS = 1u << 0,
    SINGLETON_ROWS = 1u << 1,
    ACTIVITY = 1u << 2,
    PROPAGATION = 1u << 3,
    PARALLEL_ROWS = 1u << 4,
    FIXED_COLS = 1u << 5,
    EMPTY_COLS = 1u << 6,
    SINGLETON_DUAL = 1u << 7,
    SINGLETON_EQ = 1u << 8,
    DOUBLETON_EQ = 1u << 9,
    DUAL_FIX = 1u << 10,
    PARALLEL_COLS = 1u << 11,
    /* Equitable partition stage on CUDA, before main presolve. */
    FOLDING = 1u << 12,
    ALL_RULES = (1u << 13) - 1
};

typedef struct {
    Backend backend; /* CUDA by default; CPU presolve is invalid. */
    int device, max_iterations;
    uint32_t rules;
    double max_seconds, feasibility_tol, bound_tol, zero_tol;
    int max_substitution_support, max_q_fill;
    double max_q_fill_ratio;
    int monotone_empty_columns;
} Config;

typedef enum { SCHEDULE_FIXED = 0, SCHEDULE_TIERED = 1 } Schedule;
typedef struct {
    Schedule schedule;
    int batch_limit, cleanup_rounds, global_period;
} ExecutionOptions;
typedef struct {
    uint64_t gpu_candidate_passes, gpu_batches, gpu_reductions, cpu_fallbacks;
    size_t largest_batch;
} ExecutionInfo;
typedef struct {
    uint64_t host_to_device_bytes, device_to_host_bytes;
    uint64_t host_to_device_calls, device_to_host_calls;
    uint64_t model_uploads, resident_commits, bound_updates;
    uint64_t device_candidates, downloaded_candidates;
} TransferInfo;
typedef struct {
    uint64_t retries, fill_rejections, numerical_rejections, cpu_fallback_phases;
} BatchInfo;
typedef struct {
    uint64_t calls, reductions;
    double seconds;
} RuleInfo;
typedef struct State State;
typedef struct PostsolveWorkspace PostsolveWorkspace;
/* Device pointers remain owned by the state; never dereference them on CPU.
 * Indices are int32 and values are float64, with the same CSR/Q convention. */
typedef struct {
    int rows, cols, nnz;
    const int *row_ptr, *col_ind;
    const double *values;
} DeviceMatrixView;
typedef struct {
    DeviceMatrixView Q, A, AT;
    const double *c, *lower, *upper, *row_lower, *row_upper, *objective_offset;
    int device;
} DeviceProblemView;
typedef struct {
    int resident;
    TransferInfo iterations;
    uint64_t model_export_bytes, tape_export_bytes;
    size_t tape_records, tape_entries, tape_device_bytes;
} ResidencyInfo;
typedef struct {
    int original_rows, original_cols, reduced_rows, reduced_cols;
    size_t records, payload_entries, owned_bytes;
} PostsolveInfo;
typedef struct {
    Status status;
    Backend backend;
    Problem reduced;
    State *state;
    double presolve_seconds;
    int iterations, reductions;
    char message[256];
} Result;

typedef struct {
    double primal_objective, dual_objective;
    double primal_feasibility, dual_feasibility, relative_gap;
} KKT;

const char *status_string(Status status);
Config default_config(void);
ExecutionOptions default_execution_options(void);
int cuda_available(int device);

/* Zero-initialize objects before first use. Free functions accept NULL and
 * zeroed objects; freeing an object repeatedly is safe. Never shallow-copy
 * owning structs. Input problems remain caller-owned and are never modified. */
Status problem_alloc(Problem *problem, int rows, int cols);
void problem_free(Problem *problem);
/* Release a standalone owning matrix, including one built from triplets. */
void matrix_free(Matrix *matrix);
Status problem_clone(const Problem *source, Problem *destination);
Status matrix_from_triplets(Matrix *matrix, int rows, int cols, int nnz, const int *row,
                            const int *col, const double *value);
Status validate(const Problem *problem, char *message, size_t capacity);
Status read_mps(const char *path, Problem *problem, char *message, size_t capacity);
/* Refuses to replace an existing file. Supports .mps output, .mps[.gz] input. */
Status write_mps(const char *path, const Problem *problem, char *message,
                 size_t capacity);

Status presolve(const Problem *problem, const Config *config, Result *result);
Status presolve_with_options(const Problem *problem, const Config *config,
                             const ExecutionOptions *options, Result *result);
/* CUDA-only entry point: no final host model or recovery-tape export. reduced
 * contains dimensions/nnz only until result_download is requested.
 * The ordinary APIs export the final host model once for C compatibility. */
Status presolve_resident(const Problem *problem, const Config *config,
                         const ExecutionOptions *options, Result *result);
/* Device-input counterpart. Inputs are borrowed and unchanged; the presolver
 * makes device copies and constructs AT itself. config.device must match. */
Status presolve_device(const DeviceProblemView *problem, const Config *config,
                       const ExecutionOptions *options, Result *result);
Status result_download(Result *result);
Status device_problem_view(const State *state, DeviceProblemView *view);
Status residency_info(const State *state, ResidencyInfo *info);
Status execution_info(const State *state, ExecutionInfo *info);
/* Explicit presolve payload copies; excludes CUDA/Thrust internal allocations,
 * postsolve and solver transfers. */
Status transfer_info(const State *state, TransferInfo *info);
/* Batch retries and fill/numeric rejections. CPU fallback counters stay zero.
 * Queries copy host counters without CUDA work. */
Status batch_info(const State *state, BatchInfo *info);
/* One enabled or disabled single-bit rule; host-side inclusive wall time. */
Status rule_info(const State *state, uint32_t rule, RuleInfo *info);
void state_free(State **state);
void result_free(Result *result);
/* x/y/z use Q*x+c-A^T*y-z=0; lower multipliers positive, upper negative.
 * Supply explicit reduced dimensions. The replay state and input vectors are
 * reusable. Output arrays have the original dimensions. */
Status postsolve(const State *state, const double *reduced_x, int nx,
                 const double *reduced_y, int ny, const double *reduced_z, int nz, double *x,
                 double *y, double *z);
/* State is immutable and may be shared by threads. Use a distinct workspace
 * per concurrent call. Repeat calls with the same state allocate nothing;
 * switching states may rebuild a cached device tape.
 * Workspace storage is independent of state lifetime; free it separately. */
Status postsolve_workspace_create(const State *state, PostsolveWorkspace **workspace);
/* Explicit backend selection supports CPU/GPU comparisons of one tape. */
Status postsolve_workspace_create_backend(const State *state, Backend backend,
                                          int device, PostsolveWorkspace **workspace);
void postsolve_workspace_free(PostsolveWorkspace **workspace);
Status postsolve_with_workspace(const State *state, PostsolveWorkspace *workspace,
                                const double *reduced_x, int nx, const double *reduced_y,
                                int ny, const double *reduced_z, int nz, double *x,
                                double *y, double *z);
/* All six vector pointers refer to allocations on the state's device. Only
 * status scalars cross to the host. Use a CUDA workspace on the same device. */
Status postsolve_device(const State *state, PostsolveWorkspace *workspace,
                        const double *reduced_x, int nx, const double *reduced_y, int ny,
                        const double *reduced_z, int nz, double *x, double *y, double *z);
Status postsolve_info(const State *state, PostsolveInfo *info);
Status compute_kkt(const Problem *problem, const double *x, const double *y,
                   const double *z, KKT *metrics);
Status compute_kkt_backend(const Problem *problem, const double *x, const double *y,
                           const double *z, Backend backend, int device, KKT *metrics);
double objective(const Problem *problem, const double *x);

#ifdef __cplusplus
}
#endif
#endif
