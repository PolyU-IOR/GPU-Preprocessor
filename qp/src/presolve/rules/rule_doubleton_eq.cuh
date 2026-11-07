#pragma once

__device__ static __forceinline__ void detect_doubleton_eq_rule(View A, View Q, View AT, const double * rl, const Config & cfg, int i, int first, int count, Candidate &r) {
    detect_equality_shared_rule(A, Q, AT, rl, cfg, DOUBLETON_EQ, i, first, count, r);
}

#ifndef CUDA_DOUBLETON_CUH
#define CUDA_DOUBLETON_CUH
/* Included after CudaPlan, download_matrix and device validation helpers.
 * A zero-cost Q-empty pivot changes only its A column and one survivor column.
 * Stream the two CSR/CSC views directly instead of sorting a complete COO. */
struct DtonSpec {
    int pivot, owner, target, retained;
    double alpha, beta;
};
__device__ static double doubleton_value(View a, int i, int owner, int pivot, int target,
                                    bool retained, double beta, int *error) {
    if (i == owner)
        return retained ? beta : 0.;
    Sum sum;
    double term = gpu_product2(a.get(i, pivot), beta, error), base = a.get(i, target);
    if (pivot < target) {
        sum.add(term);
        sum.add(base);
    } else {
        sum.add(base);
        sum.add(term);
    }
    return sum.value(error);
}
__global__ static void doubleton_a_counts(View a, const double *rl, const double *ru,
                                     const DtonSpec *spec, const double *source_l,
                                     const double *source_u, int *ptr, double *lower, double *upper,
                                     int *error) {
    int owner = spec->owner, pivot = spec->pivot, target = spec->target;
    bool retained = spec->retained;
    double alpha = spec->alpha, beta = spec->beta, vl = source_l[pivot], vu = source_u[pivot];

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= a.rows || (i == owner && !retained))
        return;
    int row = i - (!retained && i > owner), count = a.ptr[i + 1] - a.ptr[i];
    double lo = rl[i], hi = ru[i];
    if (i == owner) {
        count = 1;
        lo = vl - alpha;
        hi = vu - alpha;
    } else {
        double aik = a.get(i, pivot);
        if (aik != 0.) {
            double base = a.get(i, target);
            double value = doubleton_value(a, i, owner, pivot, target, retained, beta, error);
            count = count - 1 - (base != 0.) + (value != 0.);
            double shift = gpu_product2(aik, alpha, error);
            lo -= shift;
            hi -= shift;
        }
    }
    if ((isfinite(i == owner ? vl : rl[i]) && !isfinite(lo)) ||
        (isfinite(i == owner ? vu : ru[i]) && !isfinite(hi)))
        atomicExch(error, (int)NUMERICAL_ERROR);
    ptr[row + 1] = count;
    lower[row] = lo;
    upper[row] = hi;
}
__global__ static void doubleton_a_payload(View a, const DtonSpec *spec, const int *ptr, int *col,
                                      double *val, int *error) {
    int owner = spec->owner, pivot = spec->pivot, target = spec->target;
    bool retained = spec->retained;
    double beta = spec->beta;

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= a.rows || (i == owner && !retained))
        return;
    int row = i - (!retained && i > owner), pos = ptr[row];
    if (i == owner) {
        col[pos] = target - (target > pivot);
        val[pos] = beta;
        return;
    }
    double value = doubleton_value(a, i, owner, pivot, target, retained, beta, error);
    bool emitted = false;
    for (int k = a.ptr[i]; k < a.ptr[i + 1]; ++k) {
        int j = a.col[k];
        if (!emitted && j >= target) {
            if (value != 0.) {
                col[pos] = target - (target > pivot);
                val[pos++] = value;
            }
            emitted = true;
        }
        if (j != pivot && j != target) {
            col[pos] = j - (j > pivot);
            val[pos++] = a.val[k];
        }
    }
    if (!emitted && value != 0.) {
        col[pos] = target - (target > pivot);
        val[pos++] = value;
    }
    if (pos != ptr[row + 1])
        atomicExch(error, (int)INVALID_ARGUMENT);
}
__device__ static int doubleton_target_column(View a, View at, int owner, int pivot, int target,
                                         bool retained, double beta, int start, int *col,
                                         double *values, int *error) {
    int k = at.ptr[pivot], t = at.ptr[target], pos = start;
    while (k < at.ptr[pivot + 1] || t < at.ptr[target + 1]) {
        int i = k < at.ptr[pivot + 1] ? at.col[k] : INT_MAX;
        int j = t < at.ptr[target + 1] ? at.col[t] : INT_MAX;
        int row = i < j ? i : j;
        k += i == row;
        t += j == row;
        double value = doubleton_value(a, row, owner, pivot, target, retained, beta, error);
        if (value != 0.) {
            if (col) {
                col[pos] = row - (!retained && row > owner);
                values[pos] = value;
            }
            ++pos;
        }
    }
    return pos - start;
}
__global__ static void doubleton_at_counts(View a, View at, const DtonSpec *spec, int *ptr,
                                      int *error) {
    int owner = spec->owner, pivot = spec->pivot, target = spec->target;
    bool retained = spec->retained;
    double beta = spec->beta;

    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < at.rows && j != pivot) {
        int count = j == target ? doubleton_target_column(a, at, owner, pivot, target, retained, beta, 0,
                                                     nullptr, nullptr, error)
                                : at.ptr[j + 1] - at.ptr[j];
        ptr[j - (j > pivot) + 1] = count;
    }
}
__global__ static void doubleton_at_payload(View a, View at, const DtonSpec *spec,
                                       const int *ptr, int *col, double *val, int *error) {
    int owner = spec->owner, pivot = spec->pivot, target = spec->target;
    bool retained = spec->retained;
    double beta = spec->beta;

    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= at.rows || j == pivot)
        return;
    int row = j - (j > pivot), pos = ptr[row];
    if (j == target) {
        int count =
            doubleton_target_column(a, at, owner, pivot, target, retained, beta, pos, col, val, error);
        if (pos + count != ptr[row + 1])
            atomicExch(error, (int)INVALID_ARGUMENT);
    } else {
        for (int k = at.ptr[j]; k < at.ptr[j + 1]; ++k) {
            int i = at.col[k];
            col[pos] = i - (!retained && i > owner);
            val[pos++] = at.val[k];
        }
    }
}
__global__ static void doubleton_objective(View q, const DtonSpec *spec, const double *c,
                                      const double *l, const double *u, int *ptr, int *col,
                                      double *val, double *nc, double *nl, double *nu) {
    int pivot = spec->pivot;

    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j == q.rows)
        ptr[q.rows - 1] = q.ptr[q.rows];
    if (j >= q.rows || j == pivot)
        return;
    int row = j - (j > pivot);
    ptr[row] = q.ptr[j];
    nc[row] = c[j] == 0. ? 0. : c[j];
    nl[row] = l[j];
    nu[row] = u[j];
    for (int k = q.ptr[j]; k < q.ptr[j + 1]; ++k) {
        col[k] = q.col[k] - (q.col[k] > pivot);
        val[k] = q.val[k];
    }
}
__global__ static void doubleton_validate_at(View a, View at, int nnz, int *error) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (!j && (at.ptr[0] != 0 || at.ptr[at.rows] != nnz))
        atomicExch(error, (int)INVALID_ARGUMENT);
    if (j < at.rows) {
        if (!valid_device_row(at, j, nnz)) {
            atomicExch(error, (int)INVALID_ARGUMENT);
            return;
        }
        for (int k = at.ptr[j]; k < at.ptr[j + 1]; ++k)
            if (at.val[k] != a.get(at.col[k], j))
                atomicExch(error, (int)INVALID_ARGUMENT);
    }
}
__global__ static void doubleton_specification(View a, View q, View at,
                                          const Candidate *candidates, const int *selected,
                                          const double *c, const double *l, const double *u,
                                          DtonSpec *spec, int *eligible) {
    *eligible = 0;
    Candidate x = candidates[selected[0]];
    int pivot = x.j, owner = x.row;
    if (q.ptr[pivot] != q.ptr[pivot + 1] || c[pivot] != 0. || a.ptr[owner + 1] - a.ptr[owner] != 2)
        return;
    int first = a.ptr[owner], other = a.col[first] == pivot ? first + 1 : first;
    int target = a.col[other];
    double beta = -a.val[other] / a.get(owner, pivot);
    if (!isfinite(beta) || beta == 0. || at.ptr[pivot + 1] - at.ptr[pivot] > 4096 ||
        at.ptr[target + 1] - at.ptr[target] > 4096)
        return;
    *spec =
        DtonSpec{pivot, owner, target, isfinite(l[pivot]) || isfinite(u[pivot]), x.value, beta};
    *eligible = 1;
}
__global__ static void doubleton_stable_ids(int m, int n, const DtonSpec *spec, const int *rows,
                                       const int *cols, int *nr, int *nc) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < m && (i != spec->owner || spec->retained))
        nr[i - (!spec->retained && i > spec->owner)] = rows[i];
    if (i < n && i != spec->pivot)
        nc[i - (i > spec->pivot)] = cols[i];
}
static Status resident_doubleton(CudaPlan *p, int *used) {
    *used = 0;
    View a = view(p->A, p->m, p->n), at = view(p->AT, p->n, p->m);
    Buffer<DtonSpec> spec;
    Buffer<int> error;
    GPU_CHECK(spec.reserve(1));
    GPU_CHECK(error.reserve(1));
    doubleton_specification<<<1, 1>>>(a, view(p->Q, p->n, p->n), at, p->compact.data,
                                 p->selected.data, p->c.data, p->l.data, p->u.data, spec.data,
                                 error.data);
    int eligible;
    GPU_CHECK(error.download(&eligible, 1));
    if (!eligible)
        return OK;
    int nm = thrust::reduce(thrust::device, p->batch.kr.data, p->batch.kr.data + p->m, 0);
    int nn = p->n - 1;
    delete p->pending;
    p->pending = new (std::nothrow) CudaPlan(p->device, nm, nn, p->transfers);
    if (!p->pending)
        return OUT_OF_MEMORY;
    CudaPlan *next = p->pending;
    GPU_CHECK(cudaMemset(error.data, 0, sizeof(int)));
    GPU_CHECK(next->A.ptr.reserve((size_t)nm + 1));
    GPU_CHECK(next->AT.ptr.reserve((size_t)nn + 1));
    GPU_CHECK(cudaMemset(next->A.ptr.data, 0, ((size_t)nm + 1) * sizeof(int)));
    GPU_CHECK(cudaMemset(next->AT.ptr.data, 0, ((size_t)nn + 1) * sizeof(int)));
    GPU_CHECK(next->rl.reserve((size_t)nm + 1));
    GPU_CHECK(next->ru.reserve((size_t)nm + 1));
    doubleton_a_counts<<<((unsigned)p->m + 255) / 256, 256>>>(a, p->rl.data, p->ru.data, spec.data,
                                                         p->l.data, p->u.data, next->A.ptr.data,
                                                         next->rl.data, next->ru.data, error.data);
    doubleton_at_counts<<<((unsigned)p->n + 255) / 256, 256>>>(a, at, spec.data, next->AT.ptr.data,
                                                          error.data);
    thrust::inclusive_scan(thrust::device, next->A.ptr.data, next->A.ptr.data + nm + 1,
                           next->A.ptr.data);
    thrust::inclusive_scan(thrust::device, next->AT.ptr.data, next->AT.ptr.data + nn + 1,
                           next->AT.ptr.data);
    int annz, atnnz;
    GPU_CHECK(cuda_memcpy(&annz, next->A.ptr.data + nm, sizeof(int), cudaMemcpyDeviceToHost));
    GPU_CHECK(cuda_memcpy(&atnnz, next->AT.ptr.data + nn, sizeof(int), cudaMemcpyDeviceToHost));
    if (annz < 0 || annz > p->annz || atnnz != annz) {
        delete p->pending;
        p->pending = nullptr;
        return OK;
    }
    GPU_CHECK(next->A.col.reserve(annz ? annz : 1));
    GPU_CHECK(next->A.val.reserve(annz ? annz : 1));
    GPU_CHECK(next->AT.col.reserve(annz ? annz : 1));
    GPU_CHECK(next->AT.val.reserve(annz ? annz : 1));
    doubleton_a_payload<<<((unsigned)p->m + 255) / 256, 256>>>(
        a, spec.data, next->A.ptr.data, next->A.col.data, next->A.val.data, error.data);
    doubleton_at_payload<<<((unsigned)p->n + 255) / 256, 256>>>(
        a, at, spec.data, next->AT.ptr.data, next->AT.col.data, next->AT.val.data, error.data);
    GPU_CHECK(next->Q.ptr.reserve((size_t)nn + 1));
    GPU_CHECK(next->Q.col.reserve(p->qnnz ? p->qnnz : 1));
    GPU_CHECK(next->Q.val.reserve(p->qnnz ? p->qnnz : 1));
    GPU_CHECK(next->c.reserve((size_t)nn + 1));
    GPU_CHECK(next->l.reserve((size_t)nn + 1));
    GPU_CHECK(next->u.reserve((size_t)nn + 1));
    doubleton_objective<<<((unsigned)p->n + 256) / 256, 256>>>(
        view(p->Q, p->n, p->n), spec.data, p->c.data, p->l.data, p->u.data, next->Q.ptr.data,
        next->Q.col.data, next->Q.val.data, next->c.data, next->l.data, next->u.data);
    int width = std::max(nm, nn);
    validate_transformed<<<((unsigned)width + 256) / 256, 256>>>(
        view(next->A, nm, nn), view(next->Q, nn, nn), annz, p->qnnz, next->c.data,
        next->l.data, next->u.data, next->rl.data, next->ru.data, error.data);
    doubleton_validate_at<<<((unsigned)nn + 255) / 256, 256>>>(
        view(next->A, nm, nn), view(next->AT, nn, nm), annz, error.data);
    int code;
    GPU_CHECK(error.download(&code, 1));
    if (code) {
        delete p->pending;
        p->pending = nullptr;
        return OK;
    }
    GPU_CHECK(next->offset.reserve(1));
    GPU_CHECK(cuda_memcpy(next->offset.data, p->offset.data, sizeof(double),
                              cudaMemcpyDeviceToDevice));
    GPU_CHECK(next->row_ids.reserve((size_t)nm + 1));
    GPU_CHECK(next->col_ids.reserve((size_t)nn + 1));
    doubleton_stable_ids<<<((unsigned)std::max(p->m, p->n) + 255) / 256, 256>>>(
        p->m, p->n, spec.data, p->row_ids.data, p->col_ids.data, next->row_ids.data,
        next->col_ids.data);
    GPU_CHECK(cudaGetLastError());
    next->annz = annz;
    next->qnnz = p->qnnz;
    *used = 1;
    return OK;
}
#endif
