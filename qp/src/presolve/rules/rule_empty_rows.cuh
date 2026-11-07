#pragma once

__device__ static __forceinline__ void detect_empty_rows_rule(const double * rl, const double * ru, const Config & cfg, int i, Candidate &r) {
    if (rl[i] > cfg.feasibility_tol || ru[i] < -cfg.feasibility_tol)
        r.status = INFEASIBLE;
    else
        r.action = GPU_ROW;
}
