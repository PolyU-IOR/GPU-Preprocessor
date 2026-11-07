#include "internal.h"
#include <stdatomic.h>
static _Atomic uint64_t next_generation = 1;

void vector_free(Vector *v) {
    free(v->index);
    free(v->value);
    memset(v, 0, sizeof(*v));
}
Status save_row(const Matrix *a, int row, Vector *v) {
    v->size = a->row_ptr[row + 1] - a->row_ptr[row];
    if (!v->size)
        return OK;
    v->index = checked_calloc((size_t)v->size, sizeof(int));
    v->value = checked_calloc((size_t)v->size, sizeof(double));
    if (!v->index || !v->value) {
        vector_free(v);
        return OUT_OF_MEMORY;
    }
    for (int k = 0; k < v->size; k++) {
        v->index[k] = a->col_ind[a->row_ptr[row] + k];
        v->value[k] = a->values[a->row_ptr[row] + k];
    }
    return OK;
}
Status save_col(const Matrix *a, int col, Vector *v) {
    for (int i = 0; i < a->rows; i++)
        if (matrix_get(a, i, col) != 0.)
            v->size++;
    if (!v->size)
        return OK;
    v->index = checked_calloc((size_t)v->size, sizeof(int));
    v->value = checked_calloc((size_t)v->size, sizeof(double));
    if (!v->index || !v->value) {
        vector_free(v);
        return OUT_OF_MEMORY;
    }
    int k = 0;
    for (int i = 0; i < a->rows; i++) {
        double aij = matrix_get(a, i, col);
        if (aij != 0.) {
            v->index[k] = i;
            v->value[k++] = aij;
        }
    }
    return OK;
}
void record_free(Record *r) {
    vector_free(&r->qrow);
    vector_free(&r->acol);
    vector_free(&r->coefficients);
}

State *state_create(int rows, int cols) {
    State *s = checked_calloc(1, sizeof(*s));
    if (!s)
        return NULL;
    s->generation = atomic_fetch_add(&next_generation, 1);
    s->original_m = s->reduced_m = rows;
    s->original_n = s->reduced_n = cols;
    s->row_ids = rows ? checked_calloc((size_t)rows, sizeof(int)) : NULL;
    s->col_ids = cols ? checked_calloc((size_t)cols, sizeof(int)) : NULL;
    if ((rows && !s->row_ids) || (cols && !s->col_ids)) {
        state_free(&s);
        return NULL;
    }
    for (int i = 0; i < rows; ++i)
        s->row_ids[i] = i;
    for (int j = 0; j < cols; ++j)
        s->col_ids[j] = j;
    return s;
}
static int valid_vector(const Vector *v, int limit) {
    if (v->size < 0 || (v->size && (!v->index || !v->value)))
        return 0;
    for (int k = 0; k < v->size; ++k)
        if (v->index[k] < 0 || v->index[k] >= limit)
            return 0;
    return 1;
}
static void map_vector(Vector *v, const int *ids) {
    for (int k = 0; k < v->size; ++k)
        v->index[k] = ids[v->index[k]];
}
static void remove_id(int *ids, int *count, int index) {
    int tail = *count - index - 1;
    if (tail)
        memmove(ids + index, ids + index + 1, (size_t)tail * sizeof(int));
    --*count;
}
Status record_push(State *s, Record *r) {
    if (!s || !r || r->m != s->reduced_m || r->n != s->reduced_n || r->type < REC_FIXED ||
        r->type > REC_PAR_COL || !valid_vector(&r->qrow, r->n) || !valid_vector(&r->acol, r->m) ||
        !valid_vector(&r->coefficients, r->n))
        return INVALID_ARGUMENT;
    int has_col = r->type == REC_FIXED || r->type == REC_SUB || r->type == REC_BOUND ||
                  r->type == REC_SINGLE_ROW || r->type == REC_PAR_COL;
    int has_row = r->type == REC_SUB || r->type == REC_ROW || r->type == REC_SINGLE_ROW ||
                  r->type == REC_BOUND || r->type == REC_PAR_ROW;
    if ((has_col && (r->j < 0 || r->j >= r->n)) || (has_row && (r->row < 0 || r->row >= r->m)) ||
        (r->type == REC_PAR_COL && (r->k < 0 || r->k >= r->n || r->j == r->k)) ||
        (r->type == REC_PAR_ROW &&
         (r->k < 0 || r->k >= r->m || r->row == r->k)))
        return INVALID_ARGUMENT;
    if (r->type == REC_BOUND || r->type == REC_SINGLE_ROW) {
        r->pivot = 0.;
        for (int k = 0; k < r->coefficients.size; ++k)
            if (r->coefficients.index[k] == r->j)
                r->pivot = r->coefficients.value[k];
        if (r->pivot == 0. || !isfinite(r->pivot))
            return INVALID_ARGUMENT;
    }
    if (s->count == s->capacity) {
        size_t cap = s->capacity ? 2 * s->capacity : 32;
        if (cap < s->capacity || cap > SIZE_MAX / sizeof(*s->records))
            return OUT_OF_MEMORY;
        Record *p = realloc(s->records, cap * sizeof(*p));
        if (!p)
            return OUT_OF_MEMORY;
        s->records = p;
        s->capacity = cap;
    }
    /* All checks/allocations precede mutation. Preserve the reduction sequence;
     * only its coordinate IDs change. QP gradient snapshots stay stage-local. */
    int col = r->j, row = r->row;
    map_vector(&r->qrow, s->col_ids);
    map_vector(&r->acol, s->row_ids);
    map_vector(&r->coefficients, s->col_ids);
    if (has_col)
        r->j = s->col_ids[col];
    if (has_row)
        r->row = s->row_ids[row];
    if (r->type == REC_PAR_COL)
        r->k = s->col_ids[r->k];
    if (r->type == REC_PAR_ROW)
        r->k = s->row_ids[r->k];
    if (r->type == REC_FIXED || r->type == REC_SUB || r->type == REC_PAR_COL)
        remove_id(s->col_ids, &s->reduced_n, col);
    if (r->type == REC_ROW || r->type == REC_SINGLE_ROW || r->type == REC_PAR_ROW ||
        (r->type == REC_SUB && !r->retained_row))
        remove_id(s->row_ids, &s->reduced_m, row);
    s->records[s->count++] = *r;
    memset(r, 0, sizeof(*r));
    return OK;
}
Status execution_info(const State *s, ExecutionInfo *info) {
    if (!s || !info)
        return INVALID_ARGUMENT;
    *info = s->execution;
    return OK;
}
Status transfer_info(const State *s, TransferInfo *info) {
    if (!s || !info)
        return INVALID_ARGUMENT;
    *info = s->transfers;
    return OK;
}
Status batch_info(const State *s, BatchInfo *info) {
    if (!s || !info)
        return INVALID_ARGUMENT;
    *info = s->batches;
    return OK;
}
Status rule_info(const State *s, uint32_t rule, RuleInfo *info) {
    if (!s || !info || !rule || (rule & (rule - 1)) || (rule & ~ALL_RULES))
        return INVALID_ARGUMENT;
    int slot = 0;
    while ((1u << slot) != rule)
        ++slot;
    *info = s->rules[slot];
    return OK;
}
void state_free(State **s) {
    if (!s || !*s)
        return;
    for (size_t k = 0; (*s)->records && k < (*s)->count; ++k)
        record_free(&(*s)->records[k]);
    cuda_plan_free(&(*s)->device_plan);
    cuda_resident_tape_free(&(*s)->device_tape);
    free((*s)->records);
    free((*s)->row_ids);
    free((*s)->col_ids);
    free(*s);
    *s = NULL;
}
Status postsolve_info(const State *s, PostsolveInfo *info) {
    if (!s || !info)
        return INVALID_ARGUMENT;
    if (s->device_tape) {
        *info = (PostsolveInfo){s->original_m,
                                   s->original_n,
                                   s->reduced_m,
                                   s->reduced_n,
                                   s->residency.tape_records,
                                   s->residency.tape_entries,
                                   s->residency.tape_device_bytes};
        return OK;
    }
    size_t entries = 0;
    for (size_t k = 0; k < s->count; ++k) {
        const Record *r = &s->records[k];
        entries += (size_t)r->qrow.size + (size_t)r->acol.size + (size_t)r->coefficients.size;
    }
    size_t bytes = sizeof(*s) + s->capacity * sizeof(*s->records) +
                   ((size_t)s->original_m + (size_t)s->original_n) * sizeof(int) +
                   entries * (sizeof(int) + sizeof(double));
    *info = (PostsolveInfo){s->original_m, s->original_n, s->reduced_m, s->reduced_n,
                               s->count,      entries,       bytes};
    return OK;
}
Status residency_info(const State *s, ResidencyInfo *info) {
    if (!s || !info)
        return INVALID_ARGUMENT;
    *info = s->residency;
    return OK;
}
Status device_problem_view(const State *s, DeviceProblemView *view) {
    if (!s || !view || !s->device_plan)
        return INVALID_ARGUMENT;
    return cuda_resident_view(s->device_plan, view);
}
Status result_download(Result *r) {
    if (!r || !r->state)
        return INVALID_ARGUMENT;
    if (r->reduced.A.row_ptr)
        return OK;
    if (!r->state->device_plan)
        return INVALID_ARGUMENT;
    return cuda_resident_download(r->state->device_plan, &r->reduced,
                                      &r->state->residency.model_export_bytes);
}
Status state_export_host(State *s) {
    if (!s)
        return INVALID_ARGUMENT;
    if (!s->device_tape || s->records)
        return OK;
    State *copy = NULL;
    Status status = cuda_state_host_clone(s, &copy);
    if (status)
        return status;
    free(s->row_ids);
    free(s->col_ids);
    s->row_ids = copy->row_ids;
    s->col_ids = copy->col_ids;
    s->records = copy->records;
    s->capacity = copy->capacity;
    s->residency.tape_export_bytes += copy->residency.tape_export_bytes;
    copy->row_ids = copy->col_ids = NULL;
    copy->records = NULL;
    copy->count = 0;
    state_free(&copy);
    return OK;
}
Status state_export_maps(State *s) {
    if (!s)
        return INVALID_ARGUMENT;
    if (!s->device_tape || (s->row_ids && s->col_ids))
        return OK;
    int *rows = checked_calloc((size_t)s->reduced_m, sizeof(int));
    int *cols = checked_calloc((size_t)s->reduced_n, sizeof(int));
    if (!rows || !cols) {
        free(rows);
        free(cols);
        return OUT_OF_MEMORY;
    }
    Status status = cuda_state_copy_ids(s, rows, cols);
    if (status) {
        free(rows);
        free(cols);
        return status;
    }
    free(s->row_ids);
    free(s->col_ids);
    s->row_ids = rows;
    s->col_ids = cols;
    s->residency.tape_export_bytes += ((uint64_t)s->reduced_m + s->reduced_n) * sizeof(int);
    return OK;
}
