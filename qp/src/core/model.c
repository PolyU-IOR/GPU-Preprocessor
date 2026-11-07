#define _POSIX_C_SOURCE 200809L
#include "internal.h"
#include <time.h>

void *checked_calloc(size_t n, size_t s) {
    if (s && n > SIZE_MAX / s)
        return NULL;
    return calloc(n ? n : 1, s ? s : 1);
}
double monotonic_seconds(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + 1e-9 * (double)t.tv_nsec;
}
void set_message(char *m, size_t c, const char *s) {
    if (m && c)
        snprintf(m, c, "%s", s);
}
const char *status_string(Status s) {
    static const char *names[] = {"OK",         "INFEASIBLE",       "UNBOUNDED",
                                  "TIME_LIMIT", "INVALID_ARGUMENT", "OUT_OF_MEMORY",
                                  "IO_ERROR",   "NUMERICAL_ERROR",  "CUDA_ERROR"};
    return (unsigned)s < sizeof(names) / sizeof(*names) ? names[s] : "UNKNOWN";
}
Config default_config(void) {
    Config c = {CUDA, 0, 10, ALL_RULES, 1000., 1e-6, 1e-6, 1e-10, 20, 128, 2., 0};
    return c;
}
ExecutionOptions default_execution_options(void) {
    ExecutionOptions o = {SCHEDULE_FIXED, 65536, 3, 3};
    return o;
}
int checked_value(long double value, double *result) {
    *result = (double)value;
    return isfinite(value) && isfinite(*result) && (value == 0. || *result != 0.);
}
int checked_product(const double *factors, int count, double *value) {
    double mantissa = 1.;
    int exponent = 0;
    for (int i = 0; i < count; ++i) {
        if (!isfinite(factors[i]))
            return 0;
        if (factors[i] == 0.) {
            *value = 0.;
            return 1;
        }
        int e;
        mantissa *= frexp(factors[i], &e);
        exponent += e;
        mantissa = frexp(mantissa, &e);
        exponent += e;
    }
    *value = scalbn(mantissa, exponent);
    return isfinite(*value) && *value != 0.;
}
void matrix_free(Matrix *a) {
    if (!a)
        return;
    free(a->row_ptr);
    free(a->col_ind);
    free(a->values);
    memset(a, 0, sizeof(*a));
}
void problem_free(Problem *p) {
    if (!p)
        return;
    matrix_free(&p->A);
    matrix_free(&p->Q);
    free(p->c);
    free(p->lower);
    free(p->upper);
    free(p->row_lower);
    free(p->row_upper);
    memset(p, 0, sizeof(*p));
}
Status problem_alloc(Problem *p, int m, int n) {
    if (!p || m < 0 || n < 0 || m == INT_MAX || n == INT_MAX)
        return INVALID_ARGUMENT;
    Problem t = {0};
    t.A.rows = m;
    t.A.cols = t.Q.rows = t.Q.cols = n;
    t.A.row_ptr = checked_calloc((size_t)m + 1, sizeof(int));
    t.Q.row_ptr = checked_calloc((size_t)n + 1, sizeof(int));
    t.c = checked_calloc((size_t)n, sizeof(double));
    t.lower = checked_calloc((size_t)n, sizeof(double));
    t.upper = checked_calloc((size_t)n, sizeof(double));
    t.row_lower = checked_calloc((size_t)m, sizeof(double));
    t.row_upper = checked_calloc((size_t)m, sizeof(double));
    if (!t.A.row_ptr || !t.Q.row_ptr || !t.c || !t.lower || !t.upper || !t.row_lower ||
        !t.row_upper) {
        problem_free(&t);
        return OUT_OF_MEMORY;
    }
    for (int j = 0; j < n; ++j) {
        t.lower[j] = -INFINITY;
        t.upper[j] = INFINITY;
    }
    for (int i = 0; i < m; ++i) {
        t.row_lower[i] = -INFINITY;
        t.row_upper[i] = INFINITY;
    }
    problem_free(p);
    *p = t;
    return OK;
}
void coo_add(COO *b, int i, int j, double v) {
    if (b->failed || v == 0.)
        return;
    if (!isfinite(v) || b->size >= INT_MAX) {
        b->failed = 2;
        return;
    }
    if (b->size == b->capacity) {
        size_t cap = b->capacity ? 2 * b->capacity : 64;
        if (cap > SIZE_MAX / sizeof(Entry)) {
            b->failed = 1;
            return;
        }
        Entry *p = realloc(b->data, cap * sizeof(*p));
        if (!p) {
            b->failed = 1;
            return;
        }
        b->data = p;
        b->capacity = cap;
    }
    b->data[b->size++] = (Entry){i, j, v};
}
static int compare_entry(const void *a, const void *b) {
    const Entry *x = a, *y = b;
    if (x->i != y->i)
        return x->i < y->i ? -1 : 1;
    if (x->j != y->j)
        return x->j < y->j ? -1 : 1;
    /* Deterministic accumulation order for duplicate triplets. */
    return x->v < y->v ? -1 : x->v > y->v;
}
Status coo_finish(COO *b, int m, int n, Matrix *a) {
    Status status = b->failed == 2 ? NUMERICAL_ERROR : OUT_OF_MEMORY;
    Matrix t = {0};
    t.rows = m;
    t.cols = n;
    if (b->failed)
        goto done;
    if (b->size)
        qsort(b->data, b->size, sizeof(Entry), compare_entry);
    t.row_ptr = checked_calloc((size_t)m + 1, sizeof(int));
    t.col_ind = checked_calloc(b->size, sizeof(int));
    t.values = checked_calloc(b->size, sizeof(double));
    if (!t.row_ptr || !t.col_ind || !t.values)
        goto done;
    for (size_t k = 0; k < b->size;) {
        int i = b->data[k].i, j = b->data[k].j;
        if (i < 0 || i >= m || j < 0 || j >= n) {
            status = INVALID_ARGUMENT;
            goto done;
        }
        long double sum = 0., correction = 0.;
        do {
            long double value = (long double)b->data[k++].v;
            long double next = sum + value;
            if (!isfinite(next)) {
                status = NUMERICAL_ERROR;
                goto done;
            }
            correction += fabsl(sum) >= fabsl(value) ? (sum - next) + value : (value - next) + sum;
            sum = next;
        } while (k < b->size && b->data[k].i == i && b->data[k].j == j);
        double v;
        if (!checked_value(sum + correction, &v)) {
            status = NUMERICAL_ERROR;
            goto done;
        }
        if (v != 0.) {
            t.row_ptr[i + 1]++;
            t.col_ind[t.nnz] = j;
            t.values[t.nnz++] = v;
        }
    }
    for (int i = 0; i < m; i++)
        t.row_ptr[i + 1] += t.row_ptr[i];
    matrix_free(a);
    *a = t;
    /* Ownership has moved to the caller. Success must not free the temporary
     * matrix buffers through the error cleanup below. */
    free(b->data);
    memset(b, 0, sizeof(*b));
    return OK;
done:
    matrix_free(&t);
    free(b->data);
    memset(b, 0, sizeof(*b));
    return status;
}
Status matrix_from_triplets(Matrix *a, int m, int n, int nz, const int *i, const int *j,
                                   const double *v) {
    if (!a || m < 0 || n < 0 || m == INT_MAX || n == INT_MAX || nz < 0 || (nz && (!i || !j || !v)))
        return INVALID_ARGUMENT;
    COO b = {0};
    for (int k = 0; k < nz; k++) {
        if (!isfinite(v[k]) || i[k] < 0 || i[k] >= m || j[k] < 0 || j[k] >= n) {
            free(b.data);
            return INVALID_ARGUMENT;
        }
        coo_add(&b, i[k], j[k], v[k]);
    }
    return coo_finish(&b, m, n, a);
}
Status problem_clone(const Problem *s, Problem *d) {
    if (!s || !d || s == d)
        return INVALID_ARGUMENT;
    Status st = validate(s, NULL, 0);
    if (st != OK)
        return st;
    Problem t = {0};
    st = problem_alloc(&t, s->A.rows, s->A.cols);
    if (st)
        return st;
    const Matrix *src[] = {&s->A, &s->Q};
    Matrix *dst[] = {&t.A, &t.Q};
    for (int k = 0; k < 2; k++) {
        const Matrix *a = src[k];
        Matrix *b = dst[k];
        b->nnz = a->nnz;
        memcpy(b->row_ptr, a->row_ptr, ((size_t)a->rows + 1) * sizeof(int));
        if (a->nnz) {
            b->col_ind = checked_calloc((size_t)a->nnz, sizeof(int));
            b->values = checked_calloc((size_t)a->nnz, sizeof(double));
            if (!b->col_ind || !b->values) {
                problem_free(&t);
                return OUT_OF_MEMORY;
            }
            memcpy(b->col_ind, a->col_ind, (size_t)a->nnz * sizeof(int));
            memcpy(b->values, a->values, (size_t)a->nnz * sizeof(double));
        }
    }
    size_t n = (size_t)s->A.cols * sizeof(double), m = (size_t)s->A.rows * sizeof(double);
    if (n) {
        memcpy(t.c, s->c, n);
        memcpy(t.lower, s->lower, n);
        memcpy(t.upper, s->upper, n);
    }
    if (m) {
        memcpy(t.row_lower, s->row_lower, m);
        memcpy(t.row_upper, s->row_upper, m);
    }
    t.objective_offset = s->objective_offset;
    problem_free(d);
    *d = t;
    return OK;
}
double matrix_get(const Matrix *a, int i, int j) {
    int l = a->row_ptr[i], h = a->row_ptr[i + 1];
    while (l < h) {
        int k = l + (h - l) / 2;
        if (a->col_ind[k] < j)
            l = k + 1;
        else
            h = k;
    }
    return l < a->row_ptr[i + 1] && a->col_ind[l] == j ? a->values[l] : 0.;
}
void matrix_matvec(const Matrix *a, const double *x, double *y) {
    for (int i = 0; i < a->rows; i++) {
        double s = 0.;
        for (int k = a->row_ptr[i]; k < a->row_ptr[i + 1]; k++)
            s += a->values[k] * x[a->col_ind[k]];
        y[i] = s;
    }
}
void matrix_transpose_matvec(const Matrix *a, const double *x, double *y) {
    for (int j = 0; j < a->cols; j++)
        y[j] = 0.;
    for (int i = 0; i < a->rows; i++)
        for (int k = a->row_ptr[i]; k < a->row_ptr[i + 1]; k++)
            y[a->col_ind[k]] += a->values[k] * x[i];
}
Status matrix_transpose(const Matrix *a, Matrix *out) {
    Matrix t = {0};
    t.rows = a->cols;
    t.cols = a->rows;
    t.nnz = a->nnz;
    t.row_ptr = checked_calloc((size_t)t.rows + 1, sizeof(int));
    t.col_ind = checked_calloc((size_t)t.nnz, sizeof(int));
    t.values = checked_calloc((size_t)t.nnz, sizeof(double));
    int *cursor = checked_calloc((size_t)t.rows, sizeof(int));
    if (!t.row_ptr || !t.col_ind || !t.values || !cursor) {
        matrix_free(&t);
        free(cursor);
        return OUT_OF_MEMORY;
    }
    for (int k = 0; k < a->nnz; ++k)
        ++t.row_ptr[a->col_ind[k] + 1];
    for (int j = 0; j < t.rows; ++j)
        t.row_ptr[j + 1] += t.row_ptr[j];
    if (t.rows)
        memcpy(cursor, t.row_ptr, (size_t)t.rows * sizeof(int));
    for (int i = 0; i < a->rows; ++i)
        for (int k = a->row_ptr[i]; k < a->row_ptr[i + 1]; ++k) {
            int pos = cursor[a->col_ind[k]]++;
            t.col_ind[pos] = i;
            t.values[pos] = a->values[k];
        }
    free(cursor);
    matrix_free(out);
    *out = t;
    return OK;
}
static int valid_matrix(const Matrix *a) {
    if (a->rows < 0 || a->cols < 0 || a->rows == INT_MAX || a->cols == INT_MAX || a->nnz < 0 ||
        !a->row_ptr || a->row_ptr[0] != 0 || a->row_ptr[a->rows] != a->nnz ||
        (a->nnz && (!a->values || !a->col_ind)))
        return 0;
    for (int i = 0; i < a->rows; i++) {
        if (a->row_ptr[i] < 0 || a->row_ptr[i] > a->row_ptr[i + 1] || a->row_ptr[i + 1] > a->nnz)
            return 0;
        int last = -1;
        for (int k = a->row_ptr[i]; k < a->row_ptr[i + 1]; k++) {
            int j = a->col_ind[k];
            if (j <= last || j >= a->cols || !isfinite(a->values[k]) || a->values[k] == 0.)
                return 0;
            last = j;
        }
    }
    return 1;
}
Status validate(const Problem *p, char *msg, size_t cap) {
    const char *error = NULL;
    if (!p || !valid_matrix(&p->A) || !valid_matrix(&p->Q))
        error = "Invalid CSR storage.";
    else if (p->Q.rows != p->A.cols || p->Q.cols != p->A.cols)
        error = "Hessian dimensions do not match A.";
    else if ((p->A.cols && (!p->c || !p->lower || !p->upper)) ||
             (p->A.rows && (!p->row_lower || !p->row_upper)) || !isfinite(p->objective_offset))
        error = "Missing vectors or invalid objective offset.";
    if (error) {
        set_message(msg, cap, error);
        return INVALID_ARGUMENT;
    }
    for (int j = 0; j < p->A.cols; j++) {
        if (!isfinite(p->c[j]) || isnan(p->lower[j]) || isnan(p->upper[j]) ||
            p->lower[j] == INFINITY || p->upper[j] == -INFINITY) {
            error = "Invalid objective or variable bounds.";
            break;
        }
        for (int k = p->Q.row_ptr[j]; k < p->Q.row_ptr[j + 1]; k++)
            if (p->Q.values[k] != matrix_get(&p->Q, p->Q.col_ind[k], j)) {
                error = "Q must be fully symmetric.";
                break;
            }
        if (matrix_get(&p->Q, j, j) < 0.)
            error = "Negative Hessian diagonal in a convex QP.";
    }
    for (int i = 0; i < p->A.rows; i++)
        if (isnan(p->row_lower[i]) || isnan(p->row_upper[i]) || p->row_lower[i] == INFINITY ||
            p->row_upper[i] == -INFINITY)
            error = "Invalid row bounds.";
    set_message(msg, cap, error ? error : "");
    return error ? INVALID_ARGUMENT : OK;
}


void result_free(Result *r) {
    if (!r)
        return;
    state_free(&r->state);
    problem_free(&r->reduced);
    memset(r, 0, sizeof(*r));
}
