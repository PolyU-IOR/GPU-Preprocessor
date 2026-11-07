#pragma once

__global__ static void initialize_batch(int m, int n, const double *l, const double *u,
                                        const double *rl, const double *ru, int *kr, int *kc,
                                        int *owner, double *alpha, double *bl, double *bu,
                                        double *brl, double *bru) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < m) {
        kr[i] = 1;
        owner[i] = -1;
        brl[i] = rl[i];
        bru[i] = ru[i];
    }
    if (i < n) {
        kc[i] = 1;
        alpha[i] = 0.;
        bl[i] = l[i];
        bu[i] = u[i];
    }
}
__global__ static void mark_batch(int n, const Candidate *candidates, const int *selected,
                                  int count, int *kr, int *kc, int *owner, double *alpha,
                                  const double *l, const double *u, double *rl, double *ru,
                                  unsigned long long *keys, int *order) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= count)
        return;
    Candidate a = candidates[selected[t]];
    order[t] = t;
    unsigned long long key = ULLONG_MAX;
    if (a.action == GPU_FIXED || a.action == GPU_SUB) {
        kc[a.j] = 0;
        alpha[a.j] = a.value;
        if (a.action == GPU_SUB) {
            owner[a.row] = a.j;
            kr[a.row] = isfinite(l[a.j]) || isfinite(u[a.j]);
        }
    } else if (a.action == GPU_PAR_COL) {
        kc[a.j] = 0;
        key = (unsigned)a.k;
    } else if (a.action == GPU_PAR_ROW) {
        kr[a.row] = 0;
        key = (unsigned long long)(unsigned)n + (unsigned)a.k;
    } else if (a.action == GPU_ROW || a.action == GPU_SINGLE_ROW) {
        kr[a.row] = 0;
        if (a.action == GPU_SINGLE_ROW)
            key = (unsigned)a.j;
    } else if (a.action == GPU_BOUND)
        key = (unsigned)a.j;
    else if (a.action == GPU_DUAL)
        rl[a.row] = ru[a.row] = a.value;
    keys[t] = key;
}
struct BoundGroupStart {
    const unsigned long long *keys;
    __host__ __device__ bool operator()(int i) const {
        return keys[i] != ULLONG_MAX && (!i || keys[i] != keys[i - 1]);
    }
};
struct IntersectBounds {
    __host__ __device__ NewBounds operator()(NewBounds a, NewBounds b) const {
        return NewBounds{fmax(a.lower, b.lower), fmin(a.upper, b.upper)};
    }
};
__global__ static void intersection_previous(int n, int count, const unsigned long long *keys,
                                             const int *order, const NewBounds *prefix,
                                             const double *l, const double *u,
                                             NewBounds *previous) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t < count) {
        int j = (int)(keys[t] >= (unsigned)n ? keys[t] - (unsigned)n : keys[t]);
        previous[order[t]] = t && keys[t] == keys[t - 1] ? prefix[t - 1] : NewBounds{l[j], u[j]};
    }
}
__global__ static void intersection_proposals(int n, const Candidate *candidates,
                                              const int *selected, int count,
                                              const unsigned long long *keys, const int *order,
                                              const double *rl, const double *ru,
                                              NewBounds *out) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= count)
        return;
    int target = (int)(keys[t] >= (unsigned)n ? keys[t] - (unsigned)n : keys[t]);
    Candidate a = candidates[selected[order[t]]];
    double lo = a.action == GPU_PAR_ROW ? div_down(a.ratio > 0 ? rl[a.row] : ru[a.row], a.ratio)
                                        : a.lower;
    double hi = a.action == GPU_PAR_ROW ? div_up(a.ratio > 0 ? ru[a.row] : rl[a.row], a.ratio)
                                        : a.upper;
    out[t] = NewBounds{fmax(rl[target], lo), fmin(ru[target], hi)};
}
__global__ static void publish_intersection_prefix(int n, int count, const unsigned long long *keys,
                                                   const int *order, const NewBounds *prefix,
                                                   double *rl, double *ru, NewBounds *latest,
                                                   int *error) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= count)
        return;
    int target = (int)(keys[t] >= (unsigned)n ? keys[t] - (unsigned)n : keys[t]);
    NewBounds after = prefix[t];
    latest[order[t]] = after;
    if (after.lower > after.upper)
        atomicOr(error, 1);
    if (t + 1 == count || keys[t + 1] != keys[t]) {
        rl[target] = after.lower;
        ru[target] = after.upper;
    }
}
static Status cuda_prepare_batch(void *opaque, uint32_t rule, int count) {
    CudaPlan *p = (CudaPlan *)opaque;
    DeviceGuard guard(p->device);
    TransferScope traffic(p->transfers);
    GPU_CHECK(guard.status);
    DeviceBatch &b = p->batch;
    b.rule = rule;
    b.count = count;
    if (rule == PROPAGATION) {
        GPU_CHECK(b.latest.reserve(count));
        GPU_CHECK(b.error.reserve(1));
        GPU_CHECK(cudaMemset(b.error.data, 0, sizeof(int)));
        apply_propagation_bounds<<<((unsigned)count + 255) / 256, 256>>>(
            p->compact.data, p->selected.data, count, p->l.data, p->u.data, b.latest.data,
            b.error.data);
        int error;
        GPU_CHECK(b.error.download(&error, 1));
        if (error)
            return INFEASIBLE;
        return OK;
    }
    GPU_CHECK(b.kr.reserve((size_t)p->m + 1));
    GPU_CHECK(b.kc.reserve((size_t)p->n + 1));
    GPU_CHECK(b.owner.reserve((size_t)p->m + 1));
    GPU_CHECK(b.alpha.reserve((size_t)p->n + 1));
    GPU_CHECK(b.l.reserve((size_t)p->n + 1));
    GPU_CHECK(b.u.reserve((size_t)p->n + 1));
    GPU_CHECK(b.rl.reserve((size_t)p->m + 1));
    GPU_CHECK(b.ru.reserve((size_t)p->m + 1));
    GPU_CHECK(b.keys.reserve(count));
    GPU_CHECK(b.order.reserve(count));
    int width = std::max(p->m, p->n);
    if (width)
        initialize_batch<<<((unsigned)width + 255) / 256, 256>>>(
            p->m, p->n, p->l.data, p->u.data, p->rl.data, p->ru.data, b.kr.data, b.kc.data,
            b.owner.data, b.alpha.data, b.l.data, b.u.data, b.rl.data, b.ru.data);
    mark_batch<<<((unsigned)count + 255) / 256, 256>>>(
        p->n, p->compact.data, p->selected.data, count, b.kr.data, b.kc.data, b.owner.data,
        b.alpha.data, p->l.data, p->u.data, b.rl.data, b.ru.data, b.keys.data, b.order.data);
    bool bounds =
        rule == SINGLETON_ROWS || rule == PARALLEL_ROWS || rule == PARALLEL_COLS;
    if (bounds) {
        GPU_CHECK(b.latest.reserve(count));
        GPU_CHECK(b.error.reserve(1));
        GPU_CHECK(b.previous.reserve(count));
        GPU_CHECK(cudaMemset(b.error.data, 0, sizeof(int)));
        {
            try {
                thrust::stable_sort_by_key(thrust::device, b.keys.data, b.keys.data + count,
                                           b.order.data);
            } catch (const std::bad_alloc &) {
                return OUT_OF_MEMORY;
            } catch (const thrust::system_error &) {
                return CUDA_ERROR;
            }
        }
        if (rule == PARALLEL_ROWS || rule == SINGLETON_ROWS) {
            const double *base_l = rule == PARALLEL_ROWS ? p->rl.data : p->l.data;
            const double *base_u = rule == PARALLEL_ROWS ? p->ru.data : p->u.data;
            double *new_l = rule == PARALLEL_ROWS ? b.rl.data : b.l.data;
            double *new_u = rule == PARALLEL_ROWS ? b.ru.data : b.u.data;
            GPU_CHECK(b.proposals.reserve(count));
            GPU_CHECK(b.prefix.reserve(count));
            intersection_proposals<<<((unsigned)count + 255) / 256, 256>>>(
                p->n, p->compact.data, p->selected.data, count, b.keys.data, b.order.data, base_l,
                base_u, b.proposals.data);
            /* Multiword look-back payloads can be loaded speculatively before
             * their ready flags. Own and initialize scratch for scan-by-key,
             * just as for the segmented tape and support scans. */
            size_t scratch_bytes = 0;
            GPU_CHECK(cub::DeviceScan::InclusiveScanByKey(nullptr, scratch_bytes, b.keys.data,
                b.proposals.data, b.prefix.data, IntersectBounds{}, count));
            Buffer<unsigned char> scratch;
            GPU_CHECK(scratch.reserve(scratch_bytes));
            GPU_CHECK(cudaMemset(scratch.data, 0, scratch_bytes));
            GPU_CHECK(cub::DeviceScan::InclusiveScanByKey(scratch.data, scratch_bytes, b.keys.data,
                b.proposals.data, b.prefix.data, IntersectBounds{}, count));
            intersection_previous<<<((unsigned)count + 255) / 256, 256>>>(
                p->n, count, b.keys.data, b.order.data, b.prefix.data, base_l, base_u,
                b.previous.data);
            publish_intersection_prefix<<<((unsigned)count + 255) / 256, 256>>>(
                p->n, count, b.keys.data, b.order.data, b.prefix.data, new_l, new_u, b.latest.data,
                b.error.data);
        } else if (rule == PARALLEL_COLS) {
            GPU_CHECK(b.groups.reserve(count));
            int groups;
            try {
                auto begin = thrust::make_counting_iterator(0);
                groups = (int)(thrust::copy_if(thrust::device, begin, begin + count, b.groups.data,
                                               BoundGroupStart{b.keys.data}) -
                               b.groups.data);
            } catch (const std::bad_alloc &) {
                return OUT_OF_MEMORY;
            } catch (const thrust::system_error &) {
                return CUDA_ERROR;
            }
            if (groups)
                aggregate_parallel_columns<<<groups, 128>>>(
                    p->compact.data, p->selected.data, count, b.keys.data, b.order.data,
                    b.groups.data, groups, b.l.data, b.u.data, b.latest.data, b.error.data,
                    b.previous.data);
        }
        int error;
        GPU_CHECK(b.error.download(&error, 1));
        if (error)
            return error & 2 ? NUMERICAL_ERROR : INFEASIBLE;
    }
    GPU_CHECK(cudaGetLastError());
    return OK;
}
static Status cuda_publish_bounds(void *opaque) {
    CudaPlan *p = (CudaPlan *)opaque;
    DeviceGuard guard(p->device);
    GPU_CHECK(guard.status);
    TransferScope traffic(p->transfers);
    if (p->batch.rule == PROPAGATION) {
        publish_propagation_values<<<((unsigned)p->batch.count + 255) / 256, 256>>>(
            p->compact.data, p->selected.data, p->batch.latest.data, p->batch.count, p->l.data,
            p->u.data);
        /* A full queue is a safe superset of affected variables. With dense
         * incidence and broad updates, rebuilding it avoids two contended
         * atomic traversals of A/AT. Detection still uses the same bound
         * snapshot, sorted variable order and first eligible source row. */
        if ((int64_t)p->annz >= (int64_t)32 * p->n &&
            (int64_t)4 * p->batch.count >= p->n) {
            initialize_propagation<<<((unsigned)p->n + 255) / 256, 256>>>(
                p->n, p->propagation_queue.data, p->propagation_seen.data,
                p->propagation_count.data);
            p->propagation_epoch = 1;
            p->propagation_size = p->n;
            GPU_CHECK(cudaGetLastError());
            ++p->transfers->bound_updates;
            return OK;
        }
        if (p->propagation_epoch == INT_MAX) {
            GPU_CHECK(cudaMemset(p->propagation_seen.data, 0, (size_t)p->n * sizeof(int)));
            p->propagation_epoch = 1;
        } else
            ++p->propagation_epoch;
        GPU_CHECK(cudaMemset(p->propagation_count.data, 0, sizeof(int)));
        int tail = p->detected - p->batch.count;
        if (tail)
            retain_propagation_proposals<<<((unsigned)tail + 255) / 256, 256>>>(
                p->compact.data, p->batch.count, p->detected, p->propagation_epoch,
                p->propagation_seen.data, p->propagation_queue.data, p->propagation_count.data);
    } else {
        GPU_CHECK(cudaStreamSynchronize(0));
        p->l.swap(p->batch.l);
        p->u.swap(p->batch.u);
        p->rl.swap(p->batch.rl);
        p->ru.swap(p->batch.ru);
    }
    if (p->propagation_ready) {
        GPU_CHECK(p->propagation_dirty_flags.reserve((size_t)p->m + 1));
        GPU_CHECK(p->propagation_dirty_rows.reserve((size_t)p->m + 1));
        GPU_CHECK(p->propagation_dirty_count.reserve(1));
        GPU_CHECK(cudaMemset(p->propagation_dirty_flags.data, 0, (size_t)p->m * sizeof(int)));
        GPU_CHECK(cudaMemset(p->propagation_dirty_count.data, 0, sizeof(int)));
        mark_affected_rows<<<(unsigned)std::min(((size_t)p->batch.count + 7) / 8, (size_t)65535), 256>>>(
            view(p->AT, p->n, p->m), p->compact.data,
            p->selected.data, p->batch.count, p->batch.rule == SINGLETON_DUAL,
            p->propagation_dirty_flags.data, p->propagation_dirty_rows.data,
            p->propagation_dirty_count.data);
        if (p->m)
            enqueue_dirty_variables<<<std::min(p->m, 1024), 256>>>(view(p->A, p->m, p->n),
                p->propagation_dirty_rows.data, p->propagation_dirty_count.data,
                p->propagation_epoch, p->propagation_seen.data, p->propagation_queue.data,
                p->propagation_count.data);
        GPU_CHECK(p->propagation_count.download(&p->propagation_size, 1));
    }
    GPU_CHECK(cudaGetLastError());
    ++p->transfers->bound_updates;
    return OK;
}
