#pragma once

__device__ static __forceinline__ void detect_singleton_rows_rule(View A, const double * l, const double * u, const double * rl, const double * ru, const Config & cfg, int i, int first, Candidate &r) {
    int j = A.col[first];
    double a = A.val[first];
    if (fabs(a) > cfg.zero_tol) {
        r.action = GPU_SINGLE_ROW;
        r.j = j;
        r.lower = fmax(l[j], div_down(a > 0 ? rl[i] : ru[i], a));
        r.upper = fmin(u[j], div_up(a > 0 ? ru[i] : rl[i], a));
        if (r.lower > r.upper)
            r.status = INFEASIBLE;
    }
}
