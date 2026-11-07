#include "internal.h"

static double dot(const Vector *a, const double *x) {
    double v = 0.;
    for (int k = 0; k < a->size; k++)
        v += a->value[k] * x[a->index[k]];
    return v;
}
static int at(double x, double bound, double tol) {
    return isfinite(bound) && fabs(x - bound) <= tol * (1. + fabs(bound));
}
static int valid_bound_dual(double x, double z, double lower, double upper, double tol) {
    return z == 0. || (z > 0. && at(x, lower, tol)) || (z < 0. && at(x, upper, tol));
}
static void undo_bound(const Record *r, const double *x, double *y, double *z, double tol) {
    /* An unchanged bound side keeps its multiplier even if the solver's
     * primal error exceeds replay_tol. Transferring it to the opposite inducing
     * row side would violate dual signs/complementarity. */
    if ((z[r->j] > 0. && r->lower2 == r->lower) || (z[r->j] < 0. && r->upper2 == r->upper))
        return;
    if (valid_bound_dual(x[r->j], z[r->j], r->lower, r->upper, tol))
        return;
    double delta = z[r->j] / r->pivot;
    y[r->row] += delta;
    for (int k = 0; k < r->coefficients.size; k++)
        if (r->coefficients.index[k] != r->j)
            z[r->coefficients.index[k]] -= r->coefficients.value[k] * delta;
    /* The inducing pivot consumes this multiplier exactly. A floating
     * subtraction can leave an illegal nonzero dual at an infinite bound. */
    z[r->j] = 0.;
}
static int valid_inputs(const State *s, const double *xr, int nx, const double *yr, int ny,
                        const double *zr, int nz, double *x, double *y, double *z) {
    return s && nx == s->reduced_n && nz == nx && ny == s->reduced_m && (!nx || (xr && zr)) &&
           (!ny || yr) && (!s->original_n || (x && z)) && (!s->original_m || y);
}
Status postsolve_workspace_create(const State *s, PostsolveWorkspace **out) {
    return postsolve_workspace_create_backend(s, s ? s->backend : CPU, s ? s->device : 0,
                                                  out);
}
Status postsolve_workspace_create_backend(const State *s, Backend backend, int device,
                                                 PostsolveWorkspace **out) {
    if (!s || !out || *out || backend < CPU || backend > AUTO || device < 0)
        return INVALID_ARGUMENT;
    if (backend == AUTO)
        backend = cuda_available(device) ? CUDA : CPU;
    if (backend == CUDA && !cuda_available(device))
        return CUDA_ERROR;
    size_t n = (size_t)s->original_n, m = (size_t)s->original_m;
    if (n > (SIZE_MAX - m) / 2)
        return OUT_OF_MEMORY;
    size_t count = 2 * n + m;
    if (count > (SIZE_MAX - sizeof(PostsolveWorkspace)) / sizeof(double))
        return OUT_OF_MEMORY;
    PostsolveWorkspace *w = malloc(sizeof(*w));
    if (!w)
        return OUT_OF_MEMORY;
    w->cols = s->original_n;
    w->rows = s->original_m;
    w->backend = backend;
    w->device = device;
    w->gpu = NULL;
    w->host_state = NULL;
    w->host_generation = 0;
    w->data = backend == CPU ? checked_calloc(count, sizeof(double)) : NULL;
    if (backend == CPU && !w->data) {
        free(w);
        return OUT_OF_MEMORY;
    }
    if (backend == CUDA) {
        Status status = cuda_postsolve_create(s, device, &w->gpu);
        if (status) {
            free(w);
            return status;
        }
    }
    *out = w;
    return OK;
}
void postsolve_workspace_free(PostsolveWorkspace **w) {
    if (!w)
        return;
    if (*w)
        cuda_postsolve_free(&(*w)->gpu);
    if (*w)
        state_free(&(*w)->host_state);
    if (*w)
        free((*w)->data);
    free(*w);
    *w = NULL;
}
Status postsolve(const State *s, const double *xr, int nx, const double *yr, int ny,
                        const double *zr, int nz, double *x, double *y, double *z) {
    if (!valid_inputs(s, xr, nx, yr, ny, zr, nz, x, y, z))
        return INVALID_ARGUMENT;
    PostsolveWorkspace *w = NULL;
    Status status = postsolve_workspace_create(s, &w);
    if (!status)
        status = postsolve_with_workspace(s, w, xr, nx, yr, ny, zr, nz, x, y, z);
    postsolve_workspace_free(&w);
    return status;
}
Status postsolve_with_workspace(const State *s, PostsolveWorkspace *w,
                                       const double *xr, int nx, const double *yr, int ny,
                                       const double *zr, int nz, double *x, double *y, double *z) {
    if (!valid_inputs(s, xr, nx, yr, ny, zr, nz, x, y, z) || !w || w->cols < s->original_n ||
        w->rows < s->original_m)
        return INVALID_ARGUMENT;
    int n0 = s->original_n, m0 = s->original_m;
    if (!w->data) {
        w->data = checked_calloc((size_t)2 * w->cols + w->rows, sizeof(double));
        if (!w->data)
            return OUT_OF_MEMORY;
    }
    double *xx = w->data, *zz = xx + w->cols, *yy = zz + w->cols;
    if (w->backend == CUDA) {
        Status status = cuda_postsolve(s, w->gpu, xr, yr, zr, xx);
        if (!status) {
            if (n0) {
                memcpy(x, xx, (size_t)n0 * sizeof(double));
                memcpy(z, xx + n0, (size_t)n0 * sizeof(double));
            }
            if (m0)
                memcpy(y, xx + (size_t)2 * n0, (size_t)m0 * sizeof(double));
        }
        return status;
    }
    if (s->device_tape && !s->records) {
        if (w->host_generation != s->generation) {
            State *copy = NULL;
            Status code = cuda_state_host_clone(s, &copy);
            if (code)
                return code;
            state_free(&w->host_state);
            w->host_state = copy;
            w->host_generation = s->generation;
        }
        s = w->host_state;
    }
    memset(xx, 0, (size_t)n0 * sizeof(double));
    memset(zz, 0, (size_t)n0 * sizeof(double));
    memset(yy, 0, (size_t)m0 * sizeof(double));
    Status status = OK;
    for (int j = 0; j < nx; j++) {
        xx[s->col_ids[j]] = xr[j];
        zz[s->col_ids[j]] = zr[j];
        if (!isfinite(xr[j]) || !isfinite(zr[j]))
            status = INVALID_ARGUMENT;
    }
    for (int i = 0; i < ny; i++) {
        yy[s->row_ids[i]] = yr[i];
        if (!isfinite(yr[i]))
            status = INVALID_ARGUMENT;
    }
    if (status)
        goto done;
    for (size_t step = s->count; step > 0; step--) {
        const Record *r = &s->records[step - 1];
        if (r->type == REC_FOLD_COL) {
            double value = xx[r->j], dual = zz[r->j] / r->coefficients.size;
            for (int k = 0; k < r->coefficients.size; ++k) {
                int j = r->coefficients.index[k];
                xx[j] = value;
                zz[j] = dual;
            }
        } else if (r->type == REC_FOLD_ROW) {
            double dual = yy[r->row] / r->coefficients.size;
            for (int k = 0; k < r->coefficients.size; ++k)
                yy[r->coefficients.index[k]] = dual;
        } else if (r->type == REC_FIXED || r->type == REC_SUB) {
            xx[r->j] = r->alpha + dot(&r->coefficients, xx);
            if (r->type == REC_SUB && !r->retained_row)
                yy[r->row] = 0.;
            double gradient = r->c + dot(&r->qrow, xx), other = 0.;
            for (int k = 0; k < r->acol.size; k++)
                if (r->acol.index[k] != r->row)
                    other += r->acol.value[k] * yy[r->acol.index[k]];
            if (r->type == REC_FIXED)
                zz[r->j] = gradient - other;
            else {
                zz[r->j] = r->retained_row ? yy[r->row] : 0.;
                yy[r->row] = (gradient - other - zz[r->j]) / r->pivot;
            }
        } else if (r->type == REC_ROW)
            yy[r->row] = 0.;
        else if (r->type == REC_SINGLE_ROW) {
            yy[r->row] = 0.;
            undo_bound(r, xx, yy, zz, s->tolerance);
        } else if (r->type == REC_BOUND)
            undo_bound(r, xx, yy, zz, s->tolerance);
        else if (r->type == REC_PAR_ROW) {
            yy[r->row] = 0.;
            double v = dot(&r->coefficients, xx), dual = yy[r->k];
            if (!valid_bound_dual(v, dual, r->lower, r->upper, s->tolerance)) {
                yy[r->row] = dual / r->ratio;
                yy[r->k] = 0.;
            }
        } else if (r->type == REC_PAR_COL) {
            double sum = xx[r->k], lo = r->lower2, hi = r->upper2;
            double a = (sum - r->upper) / r->ratio, b = (sum - r->lower) / r->ratio;
            if (r->ratio < 0.) {
                double t = a;
                a = b;
                b = t;
            }
            lo = fmax(lo, a);
            hi = fmin(hi, b);
            if (lo > hi + s->tolerance * (1. + fabs(sum))) {
                status = NUMERICAL_ERROR;
                goto done;
            }
            double value =
                isfinite(lo) && isfinite(hi) ? .5 * lo + .5 * hi : fmin(fmax(0., lo), hi);
            xx[r->j] = value;
            xx[r->k] = sum - r->ratio * value;
            zz[r->j] = r->ratio * zz[r->k];
        }
    }
    for (int j = 0; j < n0; j++)
        if (!isfinite(xx[j]) || !isfinite(zz[j]))
            status = NUMERICAL_ERROR;
    for (int i = 0; i < m0; i++)
        if (!isfinite(yy[i]))
            status = NUMERICAL_ERROR;
    if (!status) {
        if (n0) {
            memcpy(x, xx, (size_t)n0 * sizeof(double));
            memcpy(z, zz, (size_t)n0 * sizeof(double));
        }
        if (m0)
            memcpy(y, yy, (size_t)m0 * sizeof(double));
    }
done:
    return status;
}
Status postsolve_device(const State *s, PostsolveWorkspace *w, const double *xr,
                               int nx, const double *yr, int ny, const double *zr, int nz,
                               double *x, double *y, double *z) {
    if (!valid_inputs(s, xr, nx, yr, ny, zr, nz, x, y, z) || !w || !s->device_tape ||
        w->backend != CUDA || w->device != s->device || w->cols < s->original_n ||
        w->rows < s->original_m)
        return INVALID_ARGUMENT;
    return cuda_postsolve_device(s, w->gpu, xr, yr, zr, x, y, z);
}
