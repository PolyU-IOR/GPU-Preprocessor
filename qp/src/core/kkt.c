#include "internal.h"

double objective(const Problem *p, const double *x) {
    if (!p || (p->A.cols && !x))
        return NAN;
    double value = p->objective_offset;
    for (int j = 0; j < p->A.cols; j++) {
        value += p->c[j] * x[j];
        for (int k = p->Q.row_ptr[j]; k < p->Q.row_ptr[j + 1]; k++)
            value += .5 * x[j] * p->Q.values[k] * x[p->Q.col_ind[k]];
    }
    return value;
}
static double box_value(const double *dual, const double *lo, const double *hi, int n,
                        int *overflow) {
    double value = 0.;
    for (int i = 0; i < n; i++)
        if (dual[i] != 0.) {
            double bound = dual[i] > 0. ? lo[i] : hi[i];
            if (!isfinite(bound))
                return -INFINITY;
            value += dual[i] * bound;
            if (!isfinite(value))
                *overflow = 1;
        }
    return value;
}
Status compute_kkt_backend(const Problem *p, const double *x, const double *y,
                                  const double *z, Backend backend, int device,
                                  KKT *metrics) {
    if (metrics)
        *metrics = (KKT){NAN, NAN, INFINITY, INFINITY, INFINITY};
    if (!metrics || validate(p, NULL, 0) != OK || (p->A.cols && (!x || !z)) ||
        (p->A.rows && !y) || backend < CPU || backend > AUTO || device < 0)
        return INVALID_ARGUMENT;
    int m = p->A.rows, n = p->A.cols;
    for (int j = 0; j < n; j++)
        if (!isfinite(x[j]) || !isfinite(z[j])) {
            *metrics = (KKT){NAN, NAN, INFINITY, INFINITY, INFINITY};
            return OK;
        }
    for (int i = 0; i < m; i++)
        if (!isfinite(y[i])) {
            *metrics = (KKT){NAN, NAN, INFINITY, INFINITY, INFINITY};
            return OK;
        }
    double *Ax = checked_calloc((size_t)m, sizeof(double)), *Qx = checked_calloc((size_t)n, sizeof(double)),
           *ATy = checked_calloc((size_t)n, sizeof(double));
    if (!Ax || !Qx || !ATy) {
        free(Ax);
        free(Qx);
        free(ATy);
        return OUT_OF_MEMORY;
    }
    if (backend == AUTO)
        backend = cuda_available(device) ? CUDA : CPU;
    Status status = OK;
    if (backend == CUDA)
        status = cuda_products(p, device, x, y, Ax, Qx, ATy);
    else {
        matrix_matvec(&p->A, x, Ax);
        matrix_matvec(&p->Q, x, Qx);
        matrix_transpose_matvec(&p->A, y, ATy);
    }
    if (status) {
        free(Ax);
        free(Qx);
        free(ATy);
        return status;
    }
    /* fmax ignores NaNs. Reject overflowing products before they can be
     * mistaken for small residuals, or turn normalization into inf/inf. */
    for (int i = 0; i < m; ++i)
        if (!isfinite(Ax[i]))
            status = NUMERICAL_ERROR;
    for (int j = 0; j < n; ++j)
        if (!isfinite(Qx[j]) || !isfinite(ATy[j]))
            status = NUMERICAL_ERROR;
    if (status) {
        free(Ax);
        free(Qx);
        free(ATy);
        return status;
    }
    double rhs = 0., axnorm = 0., rows = 0., bounds = 0., dnorm = 0., cnorm = 0., qnorm = 0.,
           anorm = 0., quadratic = 0., linear = 0.;
    for (int i = 0; i < m; i++) {
        if (isfinite(p->row_lower[i]))
            rhs = fmax(rhs, fabs(p->row_lower[i]));
        if (isfinite(p->row_upper[i]))
            rhs = fmax(rhs, fabs(p->row_upper[i]));
        axnorm = fmax(axnorm, fabs(Ax[i]));
        rows = fmax(rows, fmax(p->row_lower[i] - Ax[i], Ax[i] - p->row_upper[i]));
        if (!isfinite(p->row_lower[i]))
            dnorm = fmax(dnorm, y[i]);
        if (!isfinite(p->row_upper[i]))
            dnorm = fmax(dnorm, -y[i]);
    }
    for (int j = 0; j < n; j++) {
        bounds = fmax(bounds, fmax(p->lower[j] - x[j], x[j] - p->upper[j]));
        dnorm = fmax(dnorm, fabs(Qx[j] + p->c[j] - ATy[j] - z[j]));
        if (!isfinite(p->lower[j]))
            dnorm = fmax(dnorm, z[j]);
        if (!isfinite(p->upper[j]))
            dnorm = fmax(dnorm, -z[j]);
        cnorm = fmax(cnorm, fabs(p->c[j]));
        qnorm = fmax(qnorm, fabs(Qx[j]));
        anorm = fmax(anorm, fabs(ATy[j]));
        quadratic += .5 * x[j] * Qx[j];
        linear += p->c[j] * x[j];
    }
    int overflow = 0;
    double pobj = quadratic + linear + p->objective_offset;
    double dobj = -quadratic + box_value(y, p->row_lower, p->row_upper, m, &overflow) +
                  box_value(z, p->lower, p->upper, n, &overflow) + p->objective_offset;
    if (overflow || !isfinite(quadratic) || !isfinite(linear) || !isfinite(pobj) ||
        !isfinite(rows) || !isfinite(bounds) || !isfinite(dnorm)) {
        free(Ax);
        free(Qx);
        free(ATy);
        return NUMERICAL_ERROR;
    }
    double gap = isfinite(pobj) && isfinite(dobj)
                     ? fabs(pobj / (1. + fmax(fabs(pobj), fabs(dobj))) -
                            dobj / (1. + fmax(fabs(pobj), fabs(dobj))))
                     : INFINITY;
    *metrics = (KKT){pobj, dobj, fmax(rows / (1. + fmax(rhs, axnorm)), bounds),
                        dnorm / (1. + fmax(cnorm, fmax(qnorm, anorm))), gap};
    free(Ax);
    free(Qx);
    free(ATy);
    return OK;
}
Status compute_kkt(const Problem *p, const double *x, const double *y, const double *z,
                          KKT *metrics) {
    return compute_kkt_backend(p, x, y, z, CPU, 0, metrics);
}
