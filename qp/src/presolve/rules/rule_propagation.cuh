#pragma once

__device__ static Candidate propagation_row(View A, int i, int j, double a,
                                              const double *l, const double *u,
                                              const double *rl, const double *ru,
                                              const Config &cfg, const RowActivity *row_activity) {
    Candidate r = {}; r.j = j; r.row = -1; r.k = -1;
    double lo, hi;
    if (fabs(a) <= cfg.zero_tol || (rl[i] == -INFINITY && ru[i] == INFINITY)) return r;
    if (A.ptr[i + 1] - A.ptr[i] >= ROW_ACTIVITY_MIN && row_activity)
        excluded_row_activity(row_activity[i], a, l[j], u[j], lo, hi);
    else activity(A, i, j, l, u, lo, hi);
    double nl = -INFINITY, nu = INFINITY;
    if (isfinite(rl[i]) && isfinite(hi)) {
        double v = add_down(rl[i], -hi);
        if (a > 0) nl = div_down(v, a); else nu = div_up(v, a);
    }
    if (isfinite(ru[i]) && isfinite(lo)) {
        double v = add_up(ru[i], -lo);
        if (a > 0) nu = div_up(v, a); else nl = div_down(v, a);
    }
    nl = fmax(nl, l[j]); nu = fmin(nu, u[j]);
    if (nl > nu) {
        if (!small_bound_conflict(nl, nu, cfg.feasibility_tol)) r.status = INFEASIBLE;
        return r;
    }
    bool bl = nl > l[j] && (!isfinite(l[j]) || nl - l[j] > cfg.bound_tol * (1. + fabs(nl)));
    bool bu = nu < u[j] && (!isfinite(u[j]) || u[j] - nu > cfg.bound_tol * (1. + fabs(nu)));
    if (bl || bu) {
        r.action = GPU_BOUND; r.row = i;
        r.lower = bl ? nl : l[j]; r.upper = bu ? nu : u[j];
    }
    return r;
}
/* One warp reads contiguous entries of AT. Only its earliest eligible row is
 * published, preserving the scalar detector's chronological choice. */
__global__ static void detect_propagation(View a, View at, const double *l, const double *u,
                                          const double *rl, const double *ru, Config cfg,
                                          const int *active, int count,
                                          const RowActivity *activity_cache, Candidate *out) {
    unsigned lane = threadIdx.x & 31;
    size_t warp = ((size_t)blockIdx.x * blockDim.x + threadIdx.x) / 32;
    size_t stride = (size_t)gridDim.x * blockDim.x / 32;
    for (size_t pos = warp; pos < (size_t)count; pos += stride) {
        int j = active[pos], end = at.ptr[j + 1];
        for (size_t first = at.ptr[j]; first < (size_t)end; first += 32) {
            size_t t = first + lane;
            Candidate c = {};
            if (t < (size_t)end)
                c = propagation_row(a, at.col[t], j, at.val[t], l, u, rl, ru, cfg, activity_cache);
            unsigned mask = __ballot_sync(0xffffffffu, c.action || c.status);
            if (mask) {
                if (lane == (unsigned)(__ffs(mask) - 1)) out[pos] = c;
                break;
            }
        }
    }
}
__global__ static void initialize_propagation(int n, int *queue, int *seen, int *count) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (!j)
        *count = n;
    if (j < n) {
        queue[j] = j;
        seen[j] = 1;
    }
}
/* Detection emits at most one propagation proposal per variable. Each thread
 * intersects its own target with the source-snapshot proposal. */
__global__ static void apply_propagation_bounds(const Candidate *candidates, const int *selected,
                                                int count, const double *l, const double *u,
                                                NewBounds *latest, int *error) {
    int z = blockIdx.x * blockDim.x + threadIdx.x;
    if (z >= count)
        return;
    Candidate a = candidates[selected[z]];
    double lo = fmax(l[a.j], a.lower), hi = fmin(u[a.j], a.upper);
    if (lo > hi)
        atomicOr(error, 1);
    latest[z] = NewBounds{lo, hi};
}
__device__ static void enqueue_propagation(int j, int epoch, int *seen, int *queue, int *count) {
    if (atomicExch(seen + j, epoch) != epoch)
        queue[atomicAdd(count, 1)] = j;
}
__global__ static void publish_propagation_values(const Candidate *candidates,
                                                  const int *selected, const NewBounds *latest,
                                                  int count, double *l, double *u) {
    int z = blockIdx.x * blockDim.x + threadIdx.x;
    if (z < count) {
        int j = candidates[selected[z]].j;
        l[j] = latest[z].lower;
        u[j] = latest[z].upper;
    }
}
__global__ static void retain_propagation_proposals(const Candidate *candidates, int begin,
                                                    int end, int epoch, int *seen, int *queue,
                                                    int *count) {
    int z = begin + blockIdx.x * blockDim.x + threadIdx.x;
    if (z < end)
        enqueue_propagation(candidates[z].j, epoch, seen, queue, count);
}
__device__ static void mark_dirty_row(int i, int *dirty, int *rows, int *count) {
    if (!atomicExch(dirty + i, 1))
        rows[atomicAdd(count, 1)] = i;
}
__global__ static void mark_affected_rows(View at,
                                                  const Candidate *candidates,
                                                  const int *selected, int count, bool dual,
                                                  int *dirty, int *rows, int *size) {
    unsigned lane = threadIdx.x & 31;
    size_t warp = ((size_t)blockIdx.x * blockDim.x + threadIdx.x) / 32;
    size_t stride = (size_t)gridDim.x * blockDim.x / 32;
    for (size_t z = warp; z < (size_t)count; z += stride) {
        Candidate c = candidates[selected[z]];
        if (dual) {
            if (!lane) mark_dirty_row(c.row, dirty, rows, size);
        } else
            for (size_t t = (size_t)at.ptr[c.j] + lane; t < (size_t)at.ptr[c.j + 1]; t += 32)
                mark_dirty_row(at.col[t], dirty, rows, size);
    }
}
__global__ static void enqueue_dirty_variables(View a, const int *rows, const int *count, int epoch,
                                               int *seen, int *queue, int *size) {
    for (int slot = blockIdx.x; slot < *count; slot += gridDim.x) {
        int i = rows[slot];
        for (size_t k = (size_t)a.ptr[i] + threadIdx.x; k < (size_t)a.ptr[i + 1];
             k += blockDim.x)
            enqueue_propagation(a.col[k], epoch, seen, queue, size);
    }
}
__device__ static __forceinline__ void detect_propagation_rule(View A, View AT, const double * l, const double * u, const double * c, const double * rl, const double * ru, const Config & cfg, int j, const RowActivity * row_activity, Candidate &r) {
    for (int t = AT.ptr[j]; t < AT.ptr[j + 1]; ++t) {
        Candidate c = propagation_row(A, AT.col[t], j, AT.val[t], l, u, rl, ru, cfg, row_activity);
        if (c.action || c.status) { r = c; break; }
    }
}
