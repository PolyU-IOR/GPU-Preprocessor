#pragma once

__global__ static void pivot_rows(int m, const int *owner, int *pivot_row) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < m && owner[i] >= 0)
        pivot_row[owner[i]] = i;
}
__global__ static void affine_counts(View a, const int *keep, const int *pivot_row,
                                     unsigned long long *counts) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < a.cols)
        counts[j] = keep[j] ? 1
                    : pivot_row[j] >= 0
                        ? (unsigned)(a.ptr[pivot_row[j] + 1] - a.ptr[pivot_row[j]] - 1)
                        : 0;
}
__global__ static void affine_payload(View a, const int *keep, const int *pivot_row,
                                      const unsigned long long *offsets, int *ptr, int *col,
                                      double *val) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j > a.cols)
        return;
    int pos = ptr[j] = (int)offsets[j];
    if (j == a.cols)
        return;
    if (keep[j]) {
        col[pos] = j;
        val[pos] = 1.;
    } else if (pivot_row[j] >= 0) {
        int row = pivot_row[j];
        double pivot = a.get(row, j);
        for (int k = a.ptr[row]; k < a.ptr[row + 1]; ++k)
            if (a.col[k] != j) {
                col[pos] = a.col[k];
                val[pos++] = -a.val[k] / pivot;
            }
    }
}
static Status build_transform(CudaPlan *p, const Buffer<int> &keep,
                                 const Buffer<int> &owner, DeviceMatrix &t, int *nnz) {
    Buffer<int> rows;
    Buffer<unsigned long long> counts, offsets;
    GPU_CHECK(rows.reserve((size_t)p->n + 1));
    GPU_CHECK(cudaMemset(rows.data, 0xff, ((size_t)p->n + 1) * sizeof(int)));
    GPU_CHECK(counts.reserve((size_t)p->n + 1));
    GPU_CHECK(offsets.reserve((size_t)p->n + 1));
    GPU_CHECK(cudaMemset(counts.data, 0, ((size_t)p->n + 1) * sizeof(unsigned long long)));
    if (p->m)
        pivot_rows<<<((unsigned)p->m + 255) / 256, 256>>>(p->m, owner.data, rows.data);
    if (p->n)
        affine_counts<<<((unsigned)p->n + 255) / 256, 256>>>(view(p->A, p->m, p->n), keep.data,
                                                             rows.data, counts.data);
    thrust::exclusive_scan(thrust::device, counts.data, counts.data + p->n + 1, offsets.data);
    unsigned long long total;
    GPU_CHECK(cuda_memcpy(&total, offsets.data + p->n, sizeof(total), cudaMemcpyDeviceToHost));
    if (total > INT_MAX)
        return NUMERICAL_ERROR;
    *nnz = (int)total;
    GPU_CHECK(t.ptr.reserve((size_t)p->n + 1));
    GPU_CHECK(t.col.reserve(total ? (size_t)total : 1));
    GPU_CHECK(t.val.reserve(total ? (size_t)total : 1));
    affine_payload<<<((unsigned)p->n + 256) / 256, 256>>>(view(p->A, p->m, p->n), keep.data,
                                                          rows.data, offsets.data, t.ptr.data,
                                                          t.col.data, t.val.data);
    GPU_CHECK(cudaGetLastError());
    return OK;
}

__global__ static void transform_counts(View A, View Q, View T, const int *kr,
                                        const int *owner, unsigned long long *ac,
                                        unsigned long long *qc) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < A.rows) {
        unsigned long long count = 0;
        if (kr[i]) {
            if (owner[i] >= 0)
                count = T.ptr[owner[i] + 1] - T.ptr[owner[i]];
            else
                for (int t = A.ptr[i]; t < A.ptr[i + 1]; ++t)
                    count += (unsigned)(T.ptr[A.col[t] + 1] - T.ptr[A.col[t]]);
        }
        ac[i] = count;
    }
    if (i < Q.rows) {
        unsigned long long count = 0, ni = T.ptr[i + 1] - T.ptr[i];
        for (int k = Q.ptr[i]; k < Q.ptr[i + 1]; ++k) {
            int j = Q.col[k];
            if (j < i)
                continue;
            count +=
                i == j ? ni * (ni + 1) / 2 : ni * (unsigned long long)(T.ptr[j + 1] - T.ptr[j]);
        }
        qc[i] = count;
    }
}
__global__ static void emit_a(View A, View T, const int *kr, const int *owner,
                              const int *rmap, const int *cmap, const unsigned long long *offset,
                              int n, unsigned long long *keys, double *values, int *error) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= A.rows || !kr[i])
        return;
    unsigned long long pos = offset[i];
    if (owner[i] >= 0) {
        int j = owner[i];
        for (int t = T.ptr[j]; t < T.ptr[j + 1]; ++t) {
            keys[pos] = (unsigned long long)rmap[i] * n + cmap[T.col[t]];
            values[pos++] = T.val[t];
        }
    } else
        for (int k = A.ptr[i]; k < A.ptr[i + 1]; ++k) {
            int j = A.col[k];
            for (int t = T.ptr[j]; t < T.ptr[j + 1]; ++t) {
                keys[pos] = (unsigned long long)rmap[i] * n + cmap[T.col[t]];
                values[pos++] = gpu_product2(A.val[k], T.val[t], error);
            }
        }
}
__global__ static void emit_q(View Q, View T, const int *cmap,
                              const unsigned long long *offset, int n, unsigned long long *keys,
                              double *values, int *error) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= Q.rows)
        return;
    unsigned long long pos = offset[i];
    for (int k = Q.ptr[i]; k < Q.ptr[i + 1]; ++k) {
        int j = Q.col[k];
        if (j < i)
            continue;
        for (int t = T.ptr[i]; t < T.ptr[i + 1]; ++t)
            for (int u = i == j ? t : T.ptr[j]; u < T.ptr[j + 1]; ++u) {
                int a = cmap[T.col[t]], b = cmap[T.col[u]];
                double factors[] = {Q.val[k], T.val[t], T.val[u], i != j && a == b ? 2. : 1.},
                       v = 0.;
                if (!gpu_product(factors, 4, v))
                    atomicExch(error, (int)NUMERICAL_ERROR);
                if (a > b) {
                    int swap = a;
                    a = b;
                    b = swap;
                }
                keys[pos] = (unsigned long long)a * n + b;
                values[pos++] = v;
            }
    }
}
__global__ static void scalar_updates(View A, View Q, const double *c, const double *l,
                                      const double *u, const double *rl, const double *ru,
                                      const double *alpha, const int *kc, const int *kr,
                                      const int *owner, const int *rmap, double *g, double *out_l,
                                      double *out_u, double *delta, double *stage_c, int *error) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < Q.rows) {
        Sum shift, earlier;
        for (int k = Q.ptr[i]; k < Q.ptr[i + 1]; ++k) {
            int j = Q.col[k];
            double v = gpu_product2(Q.val[k], alpha[j], error);
            shift.add(v);
            if (j > i && !kc[j])
                earlier.add(v);
        }
        Sum total;
        total.add(c[i]);
        total.add(shift.value(error));
        g[i] = total.value(error);
        Sum stage;
        stage.add(c[i]);
        stage.add(earlier.value(error));
        stage_c[i] = stage.value(error);
        double f[] = {.5, alpha[i], shift.value(error)}, v = 0.;
        if (!gpu_product(f, 3, v))
            atomicExch(error, (int)NUMERICAL_ERROR);
        Sum d;
        d.add(v);
        d.add(gpu_product2(c[i], alpha[i], error));
        delta[i] = d.value(error);
    }
    if (i < A.rows && kr[i]) {
        int j = owner[i];
        double lo, hi;
        if (j >= 0) {
            lo = l[j] - alpha[j];
            hi = u[j] - alpha[j];
        } else {
            Sum sum;
            for (int k = A.ptr[i]; k < A.ptr[i + 1]; ++k)
                sum.add(gpu_product2(A.val[k], alpha[A.col[k]], error));
            double shift = sum.value(error);
            lo = rl[i] - shift;
            hi = ru[i] - shift;
        }
        if ((isfinite(j >= 0 ? l[j] : rl[i]) && !isfinite(lo)) ||
            (isfinite(j >= 0 ? u[j] : ru[i]) && !isfinite(hi)))
            atomicExch(error, (int)NUMERICAL_ERROR);
        out_l[rmap[i]] = lo;
        out_u[rmap[i]] = hi;
    }
}
__global__ static void emit_linear(View T, const int *map, const double *g,
                                   unsigned long long *keys, double *values, int *error) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= T.rows)
        return;
    for (int t = T.ptr[i]; t < T.ptr[i + 1]; ++t) {
        keys[t] = (unsigned long long)map[T.col[t]];
        values[t] = gpu_product2(T.val[t], g[i], error);
    }
}
__global__ static void group_sum(int count, const unsigned long long *keys, const double *values,
                                 int *keep, double *sums, int *error) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count)
        return;
    keep[i] = 0;
    if (i && keys[i] == keys[i - 1])
        return;
    Sum sum;
    int end = i;
    while (end < count && keys[end] == keys[i])
        sum.add(values[end++]);
    double v = sum.value(error);
    sums[i] = v;
    keep[i] = v != 0.;
}
__global__ static void compress_groups(int count, const unsigned long long *keys,
                                       const double *sums, const int *keep, const int *map,
                                       unsigned long long *out_keys, double *out_values) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count && keep[i]) {
        int dst = map[i];
        out_keys[dst] = keys[i];
        out_values[dst] = sums[i];
    }
}
__global__ static void mirror_q(int count, const unsigned long long *keys, const double *values,
                                const int *map, int n, unsigned long long *out_keys,
                                double *out_values) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count)
        return;
    int a = keys[i] / n, b = keys[i] % n, dst = map[i];
    out_keys[dst] = keys[i];
    out_values[dst] = values[i];
    if (a != b) {
        out_keys[dst + 1] = (unsigned long long)b * n + a;
        out_values[dst + 1] = values[i];
    }
}
__global__ static void mirror_counts(int count, const unsigned long long *keys, int n,
                                     int *counts) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count)
        counts[i] = (keys[i] / n == keys[i] % n) ? 1 : 2;
}
__global__ static void csr_payload(int count, int n, const unsigned long long *keys, int *ptr,
                                   int *col) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) {
        int row = keys[i] / n;
        atomicAdd(ptr + row + 1, 1);
        col[i] = keys[i] % n;
    }
}
struct DeviceCOO {
    Buffer<unsigned long long> keys, unique_keys;
    Buffer<double> values, sums, unique_values;
    Buffer<int> keep, map;
    int count = 0;
    Status allocate(unsigned long long size) {
        if (size > INT_MAX / 2)
            return NUMERICAL_ERROR;
        count = (int)size;
        GPU_CHECK(keys.reserve(size ? size : 1));
        GPU_CHECK(values.reserve(size ? size : 1));
        return OK;
    }
    Status reduce(int *error) {
        if (!count)
            return OK;
        thrust::stable_sort_by_key(thrust::device, keys.data, keys.data + count, values.data);
        GPU_CHECK(keep.reserve((size_t)count + 1));
        GPU_CHECK(map.reserve((size_t)count + 1));
        GPU_CHECK(cudaMemset(keep.data + count, 0, sizeof(int)));
        GPU_CHECK(sums.reserve(count));
        group_sum<<<((unsigned)count + 255) / 256, 256>>>(count, keys.data, values.data, keep.data,
                                                          sums.data, error);
        thrust::exclusive_scan(thrust::device, keep.data, keep.data + count + 1, map.data);
        int total;
        GPU_CHECK(cuda_memcpy(&total, map.data + count, sizeof(int), cudaMemcpyDeviceToHost));
        GPU_CHECK(unique_keys.reserve((size_t)total + 1));
        GPU_CHECK(unique_values.reserve((size_t)total + 1));
        compress_groups<<<((unsigned)count + 255) / 256, 256>>>(
            count, keys.data, sums.data, keep.data, map.data, unique_keys.data, unique_values.data);
        count = total;
        return OK;
    }
    Status finish(int rows, int cols, bool symmetric, DeviceMatrix *resident) {
        unsigned long long *k = unique_keys.data;
        double *v = unique_values.data;
        if (symmetric && count) {
            GPU_CHECK(keep.reserve((size_t)count + 1));
            GPU_CHECK(map.reserve((size_t)count + 1));
            GPU_CHECK(cudaMemset(keep.data + count, 0, sizeof(int)));
            mirror_counts<<<((unsigned)count + 255) / 256, 256>>>(count, k, cols, keep.data);
            thrust::exclusive_scan(thrust::device, keep.data, keep.data + count + 1, map.data);
            int total;
            GPU_CHECK(
                cuda_memcpy(&total, map.data + count, sizeof(int), cudaMemcpyDeviceToHost));
            GPU_CHECK(keys.reserve(total));
            GPU_CHECK(values.reserve(total));
            mirror_q<<<((unsigned)count + 255) / 256, 256>>>(count, k, v, map.data, cols, keys.data,
                                                             values.data);
            count = total;
            thrust::stable_sort_by_key(thrust::device, keys.data, keys.data + count, values.data);
            k = keys.data;
            v = values.data;
        }
        Buffer<int> ptr, col;
        GPU_CHECK(ptr.reserve((size_t)rows + 1));
        GPU_CHECK(col.reserve(count ? count : 1));
        GPU_CHECK(cudaMemset(ptr.data, 0, ((size_t)rows + 1) * sizeof(int)));
        if (count)
            csr_payload<<<((unsigned)count + 255) / 256, 256>>>(count, cols, k, ptr.data, col.data);
        thrust::inclusive_scan(thrust::device, ptr.data, ptr.data + rows + 1, ptr.data);
        resident->ptr.swap(ptr);
        resident->col.swap(col);
        if (symmetric && count)
            resident->val.swap(values);
        else
            resident->val.swap(unique_values);
        return OK;
    }
};
static Status download_matrix(const DeviceMatrix &device, Matrix *matrix, int count) {
    matrix->nnz = count;
    matrix->col_ind = count ? (int *)checked_calloc(count, sizeof(int)) : nullptr;
    matrix->values = count ? (double *)checked_calloc(count, sizeof(double)) : nullptr;
    if (count && (!matrix->col_ind || !matrix->values))
        return OUT_OF_MEMORY;
    GPU_CHECK(device.ptr.download(matrix->row_ptr, (size_t)matrix->rows + 1));
    GPU_CHECK(device.col.download(matrix->col_ind, count));
    GPU_CHECK(device.val.download(matrix->values, count));
    return OK;
}
__global__ static void scatter_linear(int count, const unsigned long long *keys,
                                      const double *values, double *c) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count)
        c[keys[i]] = values[i];
}
__global__ static void compact_bounds(int n, const int *keep, const int *map, const double *l,
                                      const double *u, double *nl, double *nu) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < n && keep[j]) {
        nl[map[j]] = l[j];
        nu[map[j]] = u[j];
    }
}
__device__ static bool valid_device_row(View a, int i, int nnz) {
    int first = a.ptr[i], last = a.ptr[i + 1];
    if (first < 0 || last < first || last > nnz)
        return false;
    int previous = -1;
    for (int k = first; k < last; ++k) {
        if (a.col[k] <= previous || a.col[k] >= a.cols || !isfinite(a.val[k]) || a.val[k] == 0.)
            return false;
        previous = a.col[k];
    }
    return true;
}
__global__ static void validate_transformed(View a, View q, int annz, int qnnz,
                                            const double *c, const double *l, const double *u,
                                            const double *rl, const double *ru, int *error) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    bool valid = true;
    if (!i)
        valid = a.ptr[0] == 0 && a.ptr[a.rows] == annz && q.ptr[0] == 0 && q.ptr[q.rows] == qnnz;
    if (i < a.rows)
        valid = valid && valid_device_row(a, i, annz) && !isnan(rl[i]) && !isnan(ru[i]) &&
                rl[i] != INFINITY && ru[i] != -INFINITY;
    if (i < q.rows) {
        valid = valid && valid_device_row(q, i, qnnz) && isfinite(c[i]) && !isnan(l[i]) &&
                !isnan(u[i]) && l[i] != INFINITY && u[i] != -INFINITY;
        if (valid) {
            for (int k = q.ptr[i]; k < q.ptr[i + 1]; ++k)
                if (q.val[k] != q.get(q.col[k], i))
                    valid = false;
            if (q.get(i, i) < 0.)
                valid = false;
        }
    }
    if (!valid)
        atomicExch(error, (int)INVALID_ARGUMENT);
}
__global__ static void resident_offset(const double *source, const double *delta, int n,
                                       bool shifted, double *out, int *error) {
    Sum sum;
    sum.add(*source);
    if (shifted)
        for (int j = 0; j < n; ++j)
            if (delta[j] != 0.)
                sum.add(delta[j]);
    *out = sum.value(error);
}
__global__ static void compact_stable_ids(int n, const int *keep, const int *map, const int *source,
                                          int *out) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < n && keep[j])
        out[map[j]] = source[j];
}
