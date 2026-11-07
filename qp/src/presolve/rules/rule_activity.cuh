#pragma once

__device__ static __forceinline__ void detect_activity_rule(View A, const double * l, const double * u, const double * rl, const double * ru, const Config & cfg, int i, int count, const RowActivity * activity_cache, Candidate &r) {
    double lo, hi;
    if (activity_cache && count >= ROW_ACTIVITY_MIN) {
        RowActivity range = activity_cache[i];
        lo = range.lower_infinite ? -INFINITY : range.lower;
        hi = range.upper_infinite ? INFINITY : range.upper;
    } else
        activity(A, i, -1, l, u, lo, hi);
    if (hi < rl[i] - cfg.feasibility_tol || lo > ru[i] + cfg.feasibility_tol)
        r.status = INFEASIBLE;
    else if (lo >= rl[i] && hi <= ru[i])
        r.action = GPU_ROW;
}
