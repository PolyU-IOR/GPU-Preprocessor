#pragma once

__device__ static __forceinline__ void detect_parallel_cols_rule(View Q, View AT, const double * c, int j, int count, int position, const int * order, const unsigned long long * hashes, Candidate &r) {
    int begin = 0, end = position;
    while (begin < end) {
        int mid = begin + (end - begin) / 2;
        if (hashes[mid] < hashes[position])
            begin = mid + 1;
        else
            end = mid;
    }
    for (int slot = begin; slot < position; ++slot) {
        int keep = order[slot];
        int first = AT.ptr[keep];
        if (AT.ptr[keep + 1] - first != count)
            continue;
        double ratio = AT.val[AT.ptr[j]] / AT.val[first];
        bool match = ratio != 0. && scaled(c[keep], c[j], ratio);
        for (int t = 0; t < count && match; ++t)
            match = AT.col[AT.ptr[j] + t] == AT.col[first + t] &&
                    scaled(AT.val[first + t], AT.val[AT.ptr[j] + t], ratio);
        if (Q.ptr[keep + 1] - Q.ptr[keep] != Q.ptr[j + 1] - Q.ptr[j])
            match = false;
        for (int t = 0; t < Q.ptr[j + 1] - Q.ptr[j] && match; ++t)
            match = Q.col[Q.ptr[j] + t] == Q.col[Q.ptr[keep] + t] &&
                    scaled(Q.val[Q.ptr[keep] + t], Q.val[Q.ptr[j] + t], ratio);
        if (match) {
            r.action = GPU_PAR_COL;
            r.k = keep;
            r.ratio = ratio;
            break;
        }
    }
}

/* One cooperative block per target. Loads/stores are parallel; thread zero
 * preserves the exact chronological addition order within the shared tile. */
__global__ static void aggregate_parallel_columns(const Candidate *candidates,
                                                  const int *selected, int count,
                                                  const unsigned long long *keys, const int *order,
                                                  const int *groups, int group_count, double *l,
                                                  double *u, NewBounds *latest, int *error,
                                                  NewBounds *previous) {
    constexpr int tile = 128;
    __shared__ double lower[tile], upper[tile];
    __shared__ int finite_lower[tile], finite_upper[tile], bad;
    __shared__ NewBounds result[tile];
    int group = blockIdx.x, lane = threadIdx.x;
    int begin = groups[group], end = group + 1 < group_count ? groups[group + 1] : count;
    int target = (int)keys[begin];
    double lo = 0., hi = 0.;
    if (!lane) {
        lo = l[target];
        hi = u[target];
        bad = 0;
    }
    for (int first = begin; first < end; first += tile) {
        int pos = first + lane;
        if (pos < end) {
            Candidate a = candidates[selected[order[pos]]];
            double al = a.ratio > 0 ? l[a.j] : u[a.j];
            double au = a.ratio > 0 ? u[a.j] : l[a.j];
            lower[lane] = a.ratio * al;
            upper[lane] = a.ratio * au;
            finite_lower[lane] = isfinite(al);
            finite_upper[lane] = isfinite(au);
        }
        __syncthreads();
        if (!lane) {
            int width = min(tile, end - first);
            for (int t = 0; t < width; ++t) {
                previous[order[first + t]] = NewBounds{lo, hi};
                double nl = lo + lower[t], nu = hi + upper[t];
                if (isnan(nl) || isnan(nu) || nl == INFINITY || nu == -INFINITY ||
                    (isfinite(lo) && finite_lower[t] && !isfinite(nl)) ||
                    (isfinite(hi) && finite_upper[t] && !isfinite(nu))) {
                    atomicOr(error, 2);
                    bad = 1;
                    break;
                }
                lo = nl;
                hi = nu;
                if (lo > hi) {
                    atomicOr(error, 1);
                    bad = 1;
                    break;
                }
                result[t] = NewBounds{lo, hi};
            }
        }
        __syncthreads();
        if (bad)
            return;
        if (pos < end)
            latest[order[pos]] = result[lane];
        __syncthreads();
    }
    if (!lane) {
        l[target] = lo;
        u[target] = hi;
    }
}
