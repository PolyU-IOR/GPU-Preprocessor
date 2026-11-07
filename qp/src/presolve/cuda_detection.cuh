#pragma once

__global__ static void detect_rows(View A, View Q, View AT, const double *l,
                                   const double *u, const double *rl, const double *ru,
                                   Config cfg, uint32_t rule, Candidate *out,
                                   const int *order, const unsigned long long *hashes,
                                   const RowActivity *activity_cache) {
    int position = blockIdx.x * blockDim.x + threadIdx.x;
    if (position >= A.rows)
        return;
    int i = rule == PARALLEL_ROWS ? order[position] : position;
    Candidate r = {};
    r.row = i;
    r.j = -1;
    r.k = -1;
    int first = A.ptr[i], count = A.ptr[i + 1] - first;
    if (rule == EMPTY_ROWS && !count) {
        detect_empty_rows_rule(rl, ru, cfg, i, r);
    } else if (rule == SINGLETON_ROWS && count == 1) {
        detect_singleton_rows_rule(A, l, u, rl, ru, cfg, i, first, r);
    } else if (rule == ACTIVITY) {
        detect_activity_rule(A, l, u, rl, ru, cfg, i, count, activity_cache, r);
    } else if (rule == PARALLEL_ROWS && count) {
        detect_parallel_rows_rule(A, first, count, position, order, hashes, r);
    } else if ((rule == SINGLETON_EQ || rule == DOUBLETON_EQ) && isfinite(rl[i]) &&
               rl[i] == ru[i] && (rule != DOUBLETON_EQ || count == 2)) {
        if (rule == SINGLETON_EQ) detect_singleton_eq_rule(A, Q, AT, rl, cfg, i, first, count, r);
        else detect_doubleton_eq_rule(A, Q, AT, rl, cfg, i, first, count, r);
    }
    out[i] = r;
}
__global__ static void detect_cols(View A, View Q, View AT, const double *l,
                                   const double *u, const double *c, const double *rl,
                                   const double *ru, Config cfg, uint32_t rule,
                                   Candidate *out, const int *order,
                                   const unsigned long long *hashes, const int *active,
                                   int active_size, const RowActivity *row_activity) {
    int position = blockIdx.x * blockDim.x + threadIdx.x;
    if (position >= (rule == PROPAGATION ? active_size : A.cols))
        return;
    int j = rule == PROPAGATION     ? active[position]
            : rule == PARALLEL_COLS ? order[position]
                                        : position;
    int output = rule == PROPAGATION ? position : j;
    Candidate r = {};
    r.j = j;
    r.row = -1;
    r.k = -1;
    int count = AT.ptr ? AT.ptr[j + 1] - AT.ptr[j] : 0;
    double diag = Q.get(j, j);
    bool separable = Q.ptr[j + 1] - Q.ptr[j] == (diag != 0.);
    if (rule == FIXED_COLS) {
        detect_fixed_cols_rule(l, u, j, r);
    } else if (rule == PROPAGATION) {
        detect_propagation_rule(A, AT, l, u, c, rl, ru, cfg, j, row_activity, r);
    } else if (rule == SINGLETON_DUAL && count == 1 && separable) {
        detect_singleton_dual_rule(A, AT, l, u, c, rl, ru, cfg, j, diag, r);
    } else if (rule == EMPTY_COLS || rule == DUAL_FIX) {
        if (rule == EMPTY_COLS) detect_empty_cols_rule(Q, AT, l, u, c, rl, ru, cfg, j, count, diag, separable, r);
        else detect_dual_fix_rule(Q, AT, l, u, c, rl, ru, cfg, j, count, diag, separable, r);
    } else if (rule == PARALLEL_COLS && count) {
        detect_parallel_cols_rule(Q, AT, c, j, count, position, order, hashes, r);
    }
    out[output] = r;
}
struct ActiveCandidate {
    __host__ __device__ bool operator()(const Candidate &c) const {
        return c.action != GPU_NONE || c.status != 0;
    }
};
static Status cuda_detect(void *opaque, const Config *cfg, uint32_t rule,
                                     int capacity, int *found) {
    int count = capacity;
    *found = 0;
    CudaPlan *p = (CudaPlan *)opaque;
    DeviceGuard guard(p->device);
    TransferScope traffic(p->transfers);
    GPU_CHECK(guard.status);
    if (!count)
        return OK;
    bool rows = rule == EMPTY_ROWS || rule == SINGLETON_ROWS || rule == ACTIVITY ||
                rule == PARALLEL_ROWS || rule == SINGLETON_EQ || rule == DOUBLETON_EQ;
    if (rule != EMPTY_ROWS && rule != SINGLETON_ROWS && rule != ACTIVITY &&
        rule != PARALLEL_ROWS && rule != FIXED_COLS) {
        Status status = ensure_transpose(p);
        if (status) return status;
    }
    View a = view(p->A, p->m, p->n), q = view(p->Q, p->n, p->n),
            at = view(p->AT, p->n, p->m);
    /* Recompute wide-row activity cooperatively at this bound snapshot.
     * Sparse models keep their existing path and summation order. */
    const RowActivity *activity_cache = nullptr;
    if (rule == ACTIVITY && p->m && p->annz / p->m >= 256) {
        GPU_CHECK(p->propagation_activity.reserve(p->m));
        cache_row_activity<<<std::min(p->m, 65535), 256>>>(a, p->l.data, p->u.data,
            p->rl.data, p->ru.data, p->propagation_activity.data);
        activity_cache = p->propagation_activity.data;
    }
    try {
        if (rule == PROPAGATION) {
            if (!p->propagation_ready) {
                GPU_CHECK(p->propagation_queue.reserve(p->n));
                GPU_CHECK(p->propagation_seen.reserve(p->n));
                GPU_CHECK(p->propagation_count.reserve(1));
                initialize_propagation<<<((unsigned)p->n + 255) / 256, 256>>>(
                    p->n, p->propagation_queue.data, p->propagation_seen.data,
                    p->propagation_count.data);
                p->propagation_size = p->n;
                if (p->annz >= ROW_ACTIVITY_MIN && p->m) {
                    GPU_CHECK(p->propagation_long_rows.reserve(p->m));
                    auto begin = thrust::make_counting_iterator(0);
                    p->propagation_long_size = (int)(thrust::copy_if(thrust::device, begin,
                        begin + p->m, p->propagation_long_rows.data, LongActivityRow{a.ptr}) -
                        p->propagation_long_rows.data);
                }
                p->propagation_ready = true;
            }
            count = p->propagation_size;
            if (!count)
                return OK;
            /* Stable variable order preserves chronological recovery records. */
            thrust::sort(thrust::device, p->propagation_queue.data,
                         p->propagation_queue.data + count);
            if (p->propagation_long_size) {
                GPU_CHECK(p->propagation_activity.reserve(p->m));
                cache_row_activity<<<std::min(p->propagation_long_size, 65535), 256>>>(a,
                    p->l.data, p->u.data, p->rl.data, p->ru.data, p->propagation_activity.data,
                    p->propagation_long_rows.data, p->propagation_long_size);
            }
        }
        if (rule == PARALLEL_ROWS && !p->row_groups) {
            GPU_CHECK(p->row_hash.reserve(count));
            GPU_CHECK(p->row_order.reserve(count));
            pair_hashes<<<((unsigned)count + 255) / 256, 256>>>(
                a, q, p->c.data, false, p->row_hash.data, p->row_order.data);
            thrust::stable_sort_by_key(thrust::device, p->row_hash.data, p->row_hash.data + count,
                                       p->row_order.data);
            p->row_groups = true;
        }
        if (rule == PARALLEL_COLS && !p->col_groups) {
            GPU_CHECK(p->col_hash.reserve(count));
            GPU_CHECK(p->col_order.reserve(count));
            pair_hashes<<<((unsigned)count + 255) / 256, 256>>>(
                at, q, p->c.data, true, p->col_hash.data, p->col_order.data);
            thrust::stable_sort_by_key(thrust::device, p->col_hash.data, p->col_hash.data + count,
                                       p->col_order.data);
            p->col_groups = true;
        }
    } catch (const std::bad_alloc &) {
        return OUT_OF_MEMORY;
    } catch (const thrust::system_error &) {
        return CUDA_ERROR;
    }
    GPU_CHECK(p->candidates.reserve(count));
    GPU_CHECK(p->compact.reserve(count));
    GPU_CHECK(cudaMemset(p->compact.data, 0, (size_t)count * sizeof(Candidate)));
    /* Initialize candidate padding before device compaction. */
    GPU_CHECK(cudaMemset(p->candidates.data, 0, (size_t)count * sizeof(Candidate)));
    if (rows)
        detect_rows<<<((unsigned)count + 255) / 256, 256>>>(
            a, q, at, p->l.data, p->u.data, p->rl.data, p->ru.data, *cfg, rule, p->candidates.data,
            p->row_order.data, p->row_hash.data, activity_cache);
    else if (rule == PROPAGATION && (int64_t)p->annz >= (int64_t)32 * p->n)
        detect_propagation<<<(unsigned)std::min(((size_t)count + 7) / 8, (size_t)65535), 256>>>(
            a, at, p->l.data, p->u.data, p->rl.data, p->ru.data, *cfg,
            p->propagation_queue.data, count, p->propagation_activity.data, p->candidates.data);
    else
        detect_cols<<<((unsigned)count + 255) / 256, 256>>>(
            a, q, at, p->l.data, p->u.data, p->c.data, p->rl.data, p->ru.data, *cfg, rule,
            p->candidates.data, p->col_order.data, p->col_hash.data, p->propagation_queue.data,
            count, p->propagation_activity.data);
    GPU_CHECK(cudaGetLastError());
    try {
        auto end = thrust::copy_if(thrust::device, p->candidates.data, p->candidates.data + count,
                                   p->compact.data, ActiveCandidate());
        *found = (int)(end - p->compact.data);
        p->detected = *found;
        if (rule == PROPAGATION && !*found) {
            p->propagation_size = 0;
            GPU_CHECK(cudaMemset(p->propagation_count.data, 0, sizeof(int)));
            if (p->propagation_epoch == INT_MAX) {
                GPU_CHECK(cudaMemset(p->propagation_seen.data, 0, (size_t)p->n * sizeof(int)));
                p->propagation_epoch = 1;
            } else
                ++p->propagation_epoch;
        }
    } catch (const std::bad_alloc &) {
        return OUT_OF_MEMORY;
    } catch (const thrust::system_error &) {
        return CUDA_ERROR;
    }
    p->transfers->device_candidates += (uint64_t)count;
    return OK;
}
