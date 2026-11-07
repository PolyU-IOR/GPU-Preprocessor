#ifndef INTERNAL_H
#define INTERNAL_H
#include "gpupresolve_qp.h"
#include "interval.h"
#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    int i, j;
    double v;
} Entry;
typedef struct {
    Entry *data;
    size_t size, capacity;
    int failed;
} COO;
typedef struct {
    int size;
    int *index;
    double *value;
} Vector;
typedef enum {
    REC_FIXED,
    REC_SUB,
    REC_ROW,
    REC_SINGLE_ROW,
    REC_BOUND,
    REC_PAR_ROW,
    REC_PAR_COL,
    REC_FOLD_ROW,
    REC_FOLD_COL
} RecordType;
typedef struct {
    RecordType type;
    int m, n, j, k, row, retained_row;
    double alpha, pivot, ratio, c, lower, upper, lower2, upper2;
    Vector qrow, acol, coefficients;
} Record;
enum { RULE_COUNT = 13 };
struct State {
    int original_m, original_n;
    /* Compact -> stable original IDs; updated only during presolve. Payload
     * indices are globalized once on commit, never during reverse replay. */
    int *row_ids, *col_ids;
    Record *records;
    size_t count, capacity;
    int reduced_m, reduced_n;
    double tolerance;
    Backend backend;
    int device;
    uint64_t generation;
    ExecutionInfo execution;
    TransferInfo transfers;
    BatchInfo batches;
    RuleInfo rules[RULE_COUNT];
    void *device_plan, *device_tape;
    ResidencyInfo residency;
};
struct PostsolveWorkspace {
    int rows, cols;
    Backend backend;
    int device;
    void *gpu;
    State *host_state;
    uint64_t host_generation;
    double *data;
};

#ifdef __cplusplus
extern "C" {
#endif
void *checked_calloc(size_t count, size_t size);
double monotonic_seconds(void);
void set_message(char *message, size_t capacity, const char *text);
double matrix_get(const Matrix *matrix, int row, int col);
void matrix_matvec(const Matrix *matrix, const double *x, double *y);
void matrix_transpose_matvec(const Matrix *matrix, const double *x, double *y);
Status matrix_transpose(const Matrix *matrix, Matrix *transpose);
void coo_add(COO *coo, int row, int col, double value);
Status coo_finish(COO *coo, int rows, int cols, Matrix *matrix);
void vector_free(Vector *vector);
Status save_row(const Matrix *matrix, int row, Vector *vector);
Status save_col(const Matrix *matrix, int col, Vector *vector);
void record_free(Record *record);
Status record_push(State *state, Record *record);
State *state_create(int rows, int cols);
/* Checked products use exponent scaling so a temporary underflow/overflow
 * cannot erase curvature or invent an unboundedness certificate. */
int checked_product(const double *factors, int count, double *value);
int checked_value(long double value, double *result);
Status cuda_products(const Problem *problem, int device, const double *x, const double *y,
                     double *Ax, double *Qx, double *ATy);
/* Test diagnostics expose private maps/records. Call before sharing a state
 * across threads; normal postsolve never mutates these host snapshots. */
Status state_export_host(State *state);
Status state_export_maps(State *state);
void cuda_plan_free(void **plan);
Status cuda_postsolve_create(const State *state, int device, void **context);
void cuda_postsolve_free(void **context);
/* packed holds 2*original_n+original_m doubles: x, z, then y. */
Status cuda_postsolve(const State *state, void *context, const double *xr,
                      const double *yr, const double *zr, double *packed);
Status cuda_resident_create(const Problem *, State *, void **, Problem *);
Status cuda_resident_create_device(const DeviceProblemView *, State *, void **,
                                   Problem *);
int cuda_resident_apply(State *, const Config *, uint32_t, int, void **, double);
int cuda_resident_fold(State *, void **, double);
Status cuda_resident_finish(void **, State *, Problem *);
void cuda_resident_tape_free(void **);
Status cuda_state_host_clone(const State *, State **);
Status cuda_resident_download(void *, Problem *, uint64_t *);
/* Validates device CSR before any kernel may use indices or binary search. */
Status cuda_validate_device_input(const DeviceProblemView *);
Status cuda_resident_view(void *, DeviceProblemView *);
Status cuda_resident_view_no_transpose(void *, DeviceProblemView *);
Status cuda_state_copy_ids(const State *, int *, int *);
Status cuda_postsolve_device(const State *, void *, const double *, const double *,
                             const double *, double *, double *, double *);
#ifdef __cplusplus
}
#endif
#endif
