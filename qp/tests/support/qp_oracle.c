#include "qp_oracle.h"
#include <float.h>
#include <math.h>
#include <string.h>

static double get(const Matrix *a, int i, int j) {
    for (int k = a->row_ptr[i]; k < a->row_ptr[i + 1]; k++)
        if (a->col_ind[k] == j)
            return a->values[k];
    return 0.;
}
/* Independent small-QP oracle: enumerate active inequality faces and solve the
 * resulting KKT systems using long-double Gaussian elimination. No presolver
 * routines or reduction thresholds are used to compute the expected solution. */
typedef struct {
    double a[12], b;
    int kind, index, sign;
} Constraint;
typedef struct {
    const Problem *p;
    Constraint eq[32], ineq[32];
    int ne, ni, chosen[12];
    double *x, *y, *z;
} Oracle;
static int linear_solve(int n, long double a[32][33], double *solution) {
    int pivots[32], rank = 0;
    memset(solution, 0, (size_t)n * sizeof(double));
    for (int i = 0; i < n; ++i) {
        long double scale = 0.;
        for (int j = 0; j < n; ++j)
            scale = fmaxl(scale, fabsl(a[i][j]));
        if (scale > 0.)
            for (int j = 0; j <= n; ++j)
                a[i][j] /= scale;
    }
    for (int col = 0; col < n && rank < n; col++) {
        int row = rank;
        for (int i = rank + 1; i < n; i++)
            if (fabsl(a[i][col]) > fabsl(a[row][col]))
                row = i;
        if (fabsl(a[row][col]) < 64.L * LDBL_EPSILON)
            continue;
        for (int j = 0; j <= n; j++) {
            long double t = a[rank][j];
            a[rank][j] = a[row][j];
            a[row][j] = t;
        }
        long double v = a[rank][col];
        for (int j = col; j <= n; j++)
            a[rank][j] /= v;
        for (int i = 0; i < n; i++)
            if (i != rank) {
                v = a[i][col];
                for (int j = col; j <= n; j++)
                    a[i][j] -= v * a[rank][j];
            }
        pivots[rank++] = col;
    }
    for (int i = rank; i < n; i++)
        if (fabsl(a[i][n]) > 1e-12L)
            return 0;
    for (int i = 0; i < rank; i++)
        solution[pivots[i]] = (double)a[i][n];
    return 1;
}
static int oracle_face(Oracle *o, int active) {
    int n = o->p->A.cols, m = o->p->A.rows, nc = o->ne + active, dim = n + nc;
    if (dim > 32)
        return 0;
    long double matrix[32][33] = {{0}};
    double solution[32];
    for (int i = 0; i < n; i++) {
        matrix[i][dim] = -o->p->c[i];
        for (int j = 0; j < n; j++)
            matrix[i][j] = get(&o->p->Q, i, j);
    }
    for (int k = 0; k < nc; k++) {
        Constraint *r = k < o->ne ? &o->eq[k] : &o->ineq[o->chosen[k - o->ne]];
        for (int j = 0; j < n; j++) {
            matrix[j][n + k] = -r->a[j];
            matrix[n + k][j] = r->a[j];
        }
        matrix[n + k][dim] = r->b;
    }
    if (!linear_solve(dim, matrix, solution))
        return 0;
    for (int k = 0; k < o->ni; k++) {
        Constraint *r = &o->ineq[k];
        double v = 0.;
        for (int j = 0; j < n; j++)
            v += r->a[j] * solution[j];
        if (r->sign * (v - r->b) < -1e-8 * (1 + fabs(r->b)))
            return 0;
    }
    for (int k = o->ne; k < nc; k++)
        if (solution[n + k] * o->ineq[o->chosen[k - o->ne]].sign < -1e-8)
            return 0;
    for (int j = 0; j < n; j++) {
        o->x[j] = solution[j];
        o->z[j] = 0.;
    }
    for (int i = 0; i < m; i++)
        o->y[i] = 0.;
    for (int k = 0; k < nc; k++) {
        Constraint *r = k < o->ne ? &o->eq[k] : &o->ineq[o->chosen[k - o->ne]];
        if (r->kind)
            o->z[r->index] += solution[n + k];
        else
            o->y[r->index] += solution[n + k];
    }
    /* Dependent active faces can yield enormous cancelling multipliers after
     * elimination. Certify the returned double vectors in the original KKT
     * equations before accepting a face; do not scale by dual magnitudes. */
    double scale = 1., residual = 0.;
    for (int j = 0; j < n; ++j) {
        double gradient = o->p->c[j];
        for (int k = o->p->Q.row_ptr[j]; k < o->p->Q.row_ptr[j + 1]; ++k)
            gradient += o->p->Q.values[k] * o->x[o->p->Q.col_ind[k]];
        double value = gradient;
        for (int i = 0; i < m; ++i)
            value -= get(&o->p->A, i, j) * o->y[i];
        value -= o->z[j];
        if (!isfinite(value) || !isfinite(gradient) || !isfinite(o->x[j]))
            return 0;
        scale = fmax(scale, fabs(gradient));
        residual = fmax(residual, fabs(value));
    }
    return residual <= 1e-8 * scale;
}
static int enumerate(Oracle *o, int start, int depth, int target) {
    if (depth == target)
        return oracle_face(o, target);
    for (int i = start; i <= o->ni - (target - depth); i++) {
        o->chosen[depth] = i;
        if (enumerate(o, i + 1, depth + 1, target))
            return 1;
    }
    return 0;
}
int test_solve(const Problem *p, double *x, double *y, double *z) {
    Oracle o = {0};
    o.p = p;
    o.x = x;
    o.y = y;
    o.z = z;
    int n = p->A.cols;
    if (n > 12 || p->A.rows > 12)
        return 0;
    for (int kind = 0; kind < 2; kind++)
        for (int i = 0; i < (kind ? n : p->A.rows); i++) {
            double l = kind ? p->lower[i] : p->row_lower[i],
                   u = kind ? p->upper[i] : p->row_upper[i];
            Constraint r = {0};
            r.kind = kind;
            r.index = i;
            for (int j = 0; j < n; j++)
                r.a[j] = kind ? (i == j) : get(&p->A, i, j);
            if (isfinite(l) && l == u) {
                r.b = l;
                if (o.ne == 32)
                    return 0;
                o.eq[o.ne++] = r;
            } else {
                if (isfinite(l)) {
                    r.b = l;
                    r.sign = 1;
                    if (o.ni == 32)
                        return 0;
                    o.ineq[o.ni++] = r;
                }
                if (isfinite(u)) {
                    r.b = u;
                    r.sign = -1;
                    if (o.ni == 32)
                        return 0;
                    o.ineq[o.ni++] = r;
                }
            }
        }
    if (o.ne > n + 12 || o.ni > 32)
        return 0;
    for (int k = 0; k <= n && k <= o.ni; k++)
        if (enumerate(&o, 0, 0, k))
            return 1;
    return 0;
}
