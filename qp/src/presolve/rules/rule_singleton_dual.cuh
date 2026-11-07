#pragma once

__device__ static __forceinline__ void detect_singleton_dual_rule(View A, View AT, const double * l, const double * u, const double * c, const double * rl, const double * ru, const Config & cfg, int j, double diag, Candidate &r) {
    int i = AT.col[AT.ptr[j]];
    double a = AT.val[AT.ptr[j]], lo, hi;
    if (rl[i] != ru[i] && fabs(a) > cfg.zero_tol) {
        activity(A, i, j, l, u, lo, hi);
        double il = a > 0 ? div_down(add_down(rl[i], -hi), a)
                          : div_down(add_up(ru[i], -lo), a);
        double iu = a > 0 ? div_up(add_up(ru[i], -lo), a)
                          : div_up(add_down(rl[i], -hi), a);
        if (isfinite(il) && isfinite(iu) && il > l[j] && iu < u[j]) {
            double dlo = diag == 0. ? c[j] : add_down(c[j], mul_down(diag, il));
            double dhi = diag == 0. ? c[j] : add_up(c[j], mul_up(diag, iu));
            double side = dlo >= 0.   ? (a > 0 ? rl[i] : ru[i])
                          : dhi <= 0. ? (a > 0 ? ru[i] : rl[i])
                                      : INFINITY;
            if (isfinite(side)) {
                r.action = GPU_DUAL;
                r.row = i;
                r.value = side;
            }
        }
    }
}
