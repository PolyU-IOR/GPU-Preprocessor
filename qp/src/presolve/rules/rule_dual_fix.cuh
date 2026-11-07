#pragma once

__device__ static __forceinline__ void detect_dual_fix_rule(View Q, View AT, const double * l, const double * u, const double * c, const double * rl, const double * ru, const Config & cfg, int j, int count, double diag, bool separable, Candidate &r) {
    detect_dual_fix_shared_rule(Q, AT, l, u, c, rl, ru, cfg, DUAL_FIX, j, count, diag, separable, r);
}
