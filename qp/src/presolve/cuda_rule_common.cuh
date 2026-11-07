#pragma once

/* Upper bound on one pivot's Hessian growth. Count potential new positions,
 * including cancellation-prone terms, and credit only its removed diagonal.
 * Off-diagonal deletions can only make the true growth smaller. Unlike a batch
 * cap, this bound does not reject thousands of individually safe pivots. */
__device__ static bool substitution_fill_allowed(View a, View q, int row, int pivot,
                                                 int maximum) {
    if (maximum < 0 || q.ptr[pivot] == q.ptr[pivot + 1])
        return true;
    bool diagonal = q.get(pivot, pivot) != 0.;
    long long fill = diagonal ? -1 : 0;
    for (int x = a.ptr[row]; x < a.ptr[row + 1]; ++x) {
        int i = a.col[x];
        if (i == pivot)
            continue;
        bool cross = q.get(pivot, i) != 0.;
        for (int y = a.ptr[row]; y < a.ptr[row + 1]; ++y) {
            int j = a.col[y];
            if (j != pivot && (diagonal || cross || q.get(pivot, j) != 0.) && q.get(i, j) == 0.)
                if (++fill > maximum)
                    return false;
        }
    }
    for (int x = q.ptr[pivot]; x < q.ptr[pivot + 1]; ++x) {
        int i = q.col[x];
        if (i == pivot || a.get(row, i) != 0.)
            continue;
        for (int y = a.ptr[row]; y < a.ptr[row + 1]; ++y) {
            int j = a.col[y];
            if (j != pivot && q.get(i, j) == 0.) {
                fill += 2;
                if (fill > maximum)
                    return false;
            }
        }
    }
    return true;
}
__device__ static void activity(View a, int i, int omit, const double *l, const double *u,
                                double &lo, double &hi) {
    lo = hi = 0.;
    for (int k = a.ptr[i]; k < a.ptr[i + 1]; ++k) {
        int j = a.col[k];
        if (j == omit)
            continue;
        double v = a.val[k];
        lo = add_down(lo, mul_down(v, v > 0 ? l[j] : u[j]));
        hi = add_up(hi, mul_up(v, v > 0 ? u[j] : l[j]));
    }
}
__device__ static bool scaled(double a, double b, double ratio) {
    return coefficient_exact_scaled(a, b, ratio);
}
__device__ static unsigned long long mix_hash(unsigned long long hash, unsigned long long value) {
    return (hash ^ value) * 1099511628211ull;
}
__global__ static void pair_hashes(View A, View Q, const double *c, bool columns,
                                   unsigned long long *hashes, int *order) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= A.rows)
        return;
    unsigned long long hash = 1469598103934665603ull;
    int first = A.ptr[i], last = A.ptr[i + 1];
    double pivot = first < last ? A.val[first] : 1.;
    hash = mix_hash(hash, (unsigned)(last - first));
    for (int k = first; k < last; ++k) {
        hash = mix_hash(hash, (unsigned)A.col[k]);
        hash = mix_hash(hash, (unsigned long long)__double_as_longlong(A.val[k] / pivot));
    }
    if (columns) {
        hash = mix_hash(hash, (unsigned)(Q.ptr[i + 1] - Q.ptr[i]));
        for (int k = Q.ptr[i]; k < Q.ptr[i + 1]; ++k) {
            hash = mix_hash(hash, (unsigned)Q.col[k]);
            hash = mix_hash(hash, (unsigned long long)__double_as_longlong(Q.val[k] / pivot));
        }
        double value = c[i] / pivot;
        if (value == 0.)
            value = 0.;
        hash = mix_hash(hash, (unsigned long long)__double_as_longlong(value));
    }
    hashes[i] = hash;
    order[i] = i;
}
__device__ static __forceinline__ void detect_equality_shared_rule(View A, View Q, View AT, const double * rl, const Config & cfg, uint32_t rule, int i, int first, int count, Candidate &r) {
    if (cfg.max_substitution_support < 0 || count - 1 <= cfg.max_substitution_support) {
        for (int t = first; t < A.ptr[i + 1]; ++t) {
            int j = A.col[t];
            double a = A.val[t];
            if (fabs(a) <= cfg.zero_tol ||
                (rule == SINGLETON_EQ && AT.ptr[j + 1] - AT.ptr[j] != 1))
                continue;
            if (!substitution_fill_allowed(A, Q, i, j, cfg.max_q_fill))
                continue;
            r.action = GPU_SUB;
            r.j = j;
            r.value = rl[i] / a;
            if (!isfinite(r.value) || (rl[i] != 0. && r.value == 0.))
                r.status = NUMERICAL_ERROR;
            for (int h = first; h < A.ptr[i + 1]; ++h)
                if (h != t) {
                    double gamma = -A.val[h] / a;
                    if (!isfinite(gamma) || gamma == 0.)
                        r.status = NUMERICAL_ERROR;
                }
            break;
        }
    }
}

__device__ static __forceinline__ void detect_dual_fix_shared_rule(View Q, View AT, const double * l, const double * u, const double * c, const double * rl, const double * ru, const Config & cfg, uint32_t rule, int j, int count, double diag, bool separable, Candidate &r) {
    if (rule == EMPTY_COLS && count) {
        return;
    }
    if (!count && separable) {
        r.value = diag > 0.   ? fmin(fmax(-c[j] / diag, l[j]), u[j])
                  : c[j] > 0. ? l[j]
                  : c[j] < 0. ? u[j]
                              : fmin(fmax(0., l[j]), u[j]);
        if (isfinite(r.value))
            r.action = GPU_FIXED;
        else
            r.status = diag > 0. ? NUMERICAL_ERROR : UNBOUNDED;
    } else if (rule != EMPTY_COLS || cfg.monotone_empty_columns) {
        bool lower_safe = true, upper_safe = true;
        for (int t = AT.ptr[j]; t < AT.ptr[j + 1]; ++t) {
            double a = AT.val[t];
            int i = AT.col[t];
            if ((a > 0 && isfinite(rl[i])) || (a < 0 && isfinite(ru[i])))
                lower_safe = false;
            if ((a > 0 && isfinite(ru[i])) || (a < 0 && isfinite(rl[i])))
                upper_safe = false;
        }
        double gl, gu;
        activity(Q, j, -1, l, u, gl, gu);
        gl = add_down(c[j], gl);
        gu = add_up(c[j], gu);
        if (lower_safe && gl >= 0. && isfinite(l[j])) {
            r.action = GPU_FIXED;
            r.value = l[j];
        } else if (upper_safe && gu <= 0. && isfinite(u[j])) {
            r.action = GPU_FIXED;
            r.value = u[j];
        } else if (Q.ptr[j] == Q.ptr[j + 1] && ((lower_safe && c[j] > 0. && !isfinite(l[j])) ||
                                                (upper_safe && c[j] < 0. && !isfinite(u[j]))))
            r.status = UNBOUNDED;
    }
}
