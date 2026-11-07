#pragma once

__device__ static __forceinline__ void detect_singleton_eq_rule(View A, View Q, View AT, const double * rl, const Config & cfg, int i, int first, int count, Candidate &r) {
    detect_equality_shared_rule(A, Q, AT, rl, cfg, SINGLETON_EQ, i, first, count, r);
}
