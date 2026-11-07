#pragma once

__device__ static __forceinline__ void detect_fixed_cols_rule(const double * l, const double * u, int j, Candidate &r) {
    if (isfinite(l[j]) && isfinite(u[j]) && u[j] == l[j]) {
        r.action = GPU_FIXED;
        r.value = .5 * l[j] + .5 * u[j];
    }
}
