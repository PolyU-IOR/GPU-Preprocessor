#pragma once

enum {
    GPU_NONE,
    GPU_ROW,
    GPU_SINGLE_ROW,
    GPU_BOUND,
    GPU_FIXED,
    GPU_SUB,
    GPU_PAR_ROW,
    GPU_PAR_COL,
    GPU_DUAL
};
typedef struct {
    int action, j, row, k, status;
    double value, lower, upper, ratio;
} Candidate;
typedef struct {
    double lower, upper;
} NewBounds;
#include "cuda_row_activity.cuh"

struct DeviceBatch {
    uint32_t rule = 0;
    int count = 0;
    Buffer<int> kr, kc, owner, order, error, groups;
    Buffer<unsigned long long> keys;
    Buffer<double> alpha, l, u, rl, ru;
    Buffer<NewBounds> latest, previous;
    Buffer<NewBounds> proposals, prefix;
};
struct CudaPlan {
    int device, m, n;
    int annz = 0, qnnz = 0;
    TransferInfo *transfers;
    DeviceMatrix A, Q, AT;
    Buffer<double> l, u, c, rl, ru;
    Buffer<double> offset, stages;
    Buffer<int> row_ids, col_ids;
    Buffer<Candidate> candidates, compact;
    Buffer<unsigned long long> row_hash, col_hash;
    Buffer<int> row_order, col_order, selected;
    Buffer<int> propagation_queue, propagation_seen, propagation_count;
    Buffer<int> propagation_dirty_flags, propagation_dirty_rows, propagation_dirty_count;
    Buffer<int> propagation_long_rows;
    int propagation_long_size = 0;
    Buffer<RowActivity> propagation_activity;
    int propagation_size = 0, propagation_epoch = 1, detected = 0;
    bool propagation_ready = false;
    bool row_groups = false, col_groups = false;
    DeviceBatch batch;
    int batch_caps[RULE_COUNT] = {};
    bool reuse_objective = false, row_only = false;
    CudaPlan *pending = nullptr;
    explicit CudaPlan(int d, int rows, int cols, TransferInfo *t)
        : device(d), m(rows), n(cols), transfers(t) {}
    ~CudaPlan() {
        delete pending;
    }
};
__global__ static void transpose_keys(View a, unsigned long long *keys) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < a.rows)
        for (int k = a.ptr[i]; k < a.ptr[i + 1]; ++k)
            keys[k] = ((unsigned long long)(unsigned)a.col[k] << 32) | (unsigned)i;
}
__global__ static void transpose_payload(int nnz, const unsigned long long *keys, int *ptr,
                                         int *col) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k < nnz) {
        atomicAdd(ptr + (unsigned)(keys[k] >> 32) + 1, 1);
        col[k] = (int)(unsigned)keys[k];
    }
}
static Status device_transpose(const DeviceMatrix &a, int m, int n, int nnz,
                                  DeviceMatrix &at) {
    GPU_CHECK(at.ptr.reserve((size_t)n + 1));
    GPU_CHECK(at.col.reserve(nnz ? nnz : 1));
    GPU_CHECK(at.val.reserve(nnz ? nnz : 1));
    GPU_CHECK(cudaMemset(at.ptr.data, 0, ((size_t)n + 1) * sizeof(int)));
    if (nnz) {
        Buffer<unsigned long long> keys;
        GPU_CHECK(keys.reserve(nnz));
        GPU_CHECK(cuda_memcpy(at.val.data, a.val.data, (size_t)nnz * sizeof(double),
                                  cudaMemcpyDeviceToDevice));
        transpose_keys<<<((unsigned)m + 255) / 256, 256>>>(view(a, m, n), keys.data);
        try {
            thrust::stable_sort_by_key(thrust::device, keys.data, keys.data + nnz, at.val.data);
            transpose_payload<<<((unsigned)nnz + 255) / 256, 256>>>(nnz, keys.data, at.ptr.data,
                                                                    at.col.data);
            thrust::inclusive_scan(thrust::device, at.ptr.data, at.ptr.data + n + 1, at.ptr.data);
        } catch (const std::bad_alloc &) {
            return OUT_OF_MEMORY;
        } catch (const thrust::system_error &) {
            return CUDA_ERROR;
        }
    }
    GPU_CHECK(cudaGetLastError());
    return OK;
}
static Status ensure_transpose(CudaPlan *p) {
    return p->AT.ptr.data ? OK : device_transpose(p->A, p->m, p->n, p->annz, p->AT);
}
static Status cuda_plan_create(const Problem *p, int device, TransferInfo *transfers,
                               void **out) {
    DeviceGuard guard(device);
    TransferScope traffic(transfers);
    GPU_CHECK(guard.status);
    std::unique_ptr<CudaPlan> plan(new (std::nothrow)
                                       CudaPlan(device, p->A.rows, p->A.cols, transfers));
    if (!plan)
        return OUT_OF_MEMORY;
    GPU_CHECK(plan->A.upload(p->A));
    GPU_CHECK(plan->Q.upload(p->Q));
    GPU_CHECK(plan->l.upload(p->lower, plan->n));
    GPU_CHECK(plan->u.upload(p->upper, plan->n));
    GPU_CHECK(plan->c.upload(p->c, plan->n));
    GPU_CHECK(plan->rl.upload(p->row_lower, plan->m));
    GPU_CHECK(plan->ru.upload(p->row_upper, plan->m));
    plan->annz = p->A.nnz;
    plan->qnnz = p->Q.nnz;
    ++transfers->model_uploads;
    *out = plan.release();
    return OK;
}
extern "C" void cuda_plan_free(void **out) {
    if (!out || !*out)
        return;
    CudaPlan *p = (CudaPlan *)*out;
    DeviceGuard guard(p->device);
    delete p;
}
static void cuda_plan_commit(void **out) {
    CudaPlan *old = (CudaPlan *)*out;
    CudaPlan *next = old->pending;
    old->pending = nullptr;
    std::copy(old->batch_caps, old->batch_caps + RULE_COUNT, next->batch_caps);
    if (next->row_only) {
        next->Q.ptr.swap(old->Q.ptr);
        next->Q.col.swap(old->Q.col);
        next->Q.val.swap(old->Q.val);
        next->c.swap(old->c);
        next->offset.swap(old->offset);
        next->col_ids.swap(old->col_ids);
        next->l.swap(old->batch.l);
        next->u.swap(old->batch.u);
        next->row_only = false;
    }
    if (next->reuse_objective) {
        next->Q.ptr.swap(old->Q.ptr);
        next->Q.col.swap(old->Q.col);
        next->Q.val.swap(old->Q.val);
        next->c.swap(old->c);
        next->l.swap(old->l);
        next->u.swap(old->u);
        next->offset.swap(old->offset);
        next->row_ids.swap(old->row_ids);
        next->col_ids.swap(old->col_ids);
        next->reuse_objective = false;
    }
    ++old->transfers->resident_commits;
    *out = next;
    DeviceGuard guard(old->device);
    delete old;
}
static int rule_slot(uint32_t rule) {
    int slot = 0;
    while ((rule >>= 1) != 0)
        ++slot;
    return slot;
}
static int cuda_batch_limit(void *opaque, uint32_t rule, int limit) {
    CudaPlan *p = (CudaPlan *)opaque;
    int cap = p->batch_caps[rule_slot(rule)];
    return cap ? std::min(cap, limit) : limit;
}
static void cuda_plan_reject(void *opaque, uint32_t rule, int limit) {
    CudaPlan *p = (CudaPlan *)opaque;
    DeviceGuard guard(p->device);
    delete p->pending;
    p->pending = nullptr;
    if (limit > 0)
        p->batch_caps[rule_slot(rule)] = limit;
}
