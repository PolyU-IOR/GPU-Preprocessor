#ifndef CUDA_ROW_TRANSFORM_CUH
#define CUDA_ROW_TRANSFORM_CUH
/* These four rules delete one distinct row per selected candidate and may
 * intersect row/variable bounds, but never change columns or the objective.
 * Keep the original buffers intact until capture and commit have succeeded. */
__global__ static void retained_row_lengths(View a, const int *keep, int *lengths) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i <= a.rows)
        lengths[i] = i < a.rows && keep[i] ? a.ptr[i + 1] - a.ptr[i] : 0;
}
__global__ static void compact_row_payload(View a, const int *keep, const int *map,
                                           const int *offsets, const double *lo,
                                           const double *up, const int *ids, int *ptr,
                                           int *col, double *val, double *new_lo,
                                           double *new_up, int *new_ids) {
    for (size_t i = blockIdx.x; i <= (size_t)a.rows; i += gridDim.x) {
        if (i == (size_t)a.rows) {
            if (!threadIdx.x)
                ptr[map[i]] = offsets[i];
        } else if (keep[i]) {
            int dest = map[i], begin = a.ptr[i];
            if (!threadIdx.x) {
                ptr[dest] = offsets[i];
                new_lo[dest] = lo[i]; new_up[dest] = up[i]; new_ids[dest] = ids[i];
            }
            for (size_t k = (size_t)begin + threadIdx.x; k < (size_t)a.ptr[i + 1];
                 k += blockDim.x) {
                int pos = offsets[i] + (int)k - begin;
                col[pos] = a.col[k]; val[pos] = a.val[k];
            }
        }
    }
}
static Status cuda_transform_rows(CudaPlan *p) {
    int m = p->m, n = p->n, newm = m - p->batch.count;
    if (newm < 0)
        return INVALID_ARGUMENT;
    Buffer<int> map, offsets;
    GPU_CHECK(map.reserve((size_t)m + 1));
    GPU_CHECK(offsets.reserve((size_t)m + 1));
    GPU_CHECK(cudaMemset(p->batch.kr.data + m, 0, sizeof(int)));
    thrust::exclusive_scan(thrust::device, p->batch.kr.data, p->batch.kr.data + m + 1, map.data);
    retained_row_lengths<<<((unsigned)m + 256) / 256, 256>>>(
        view(p->A, m, n), p->batch.kr.data, offsets.data);
    thrust::exclusive_scan(thrust::device, offsets.data, offsets.data + m + 1, offsets.data);
    int nnz;
    GPU_CHECK(cuda_memcpy(&nnz, offsets.data + m, sizeof(int), cudaMemcpyDeviceToHost));
    delete p->pending;
    p->pending = new (std::nothrow) CudaPlan(p->device, newm, n, p->transfers);
    if (!p->pending)
        return OUT_OF_MEMORY;
    CudaPlan *next = p->pending;
    GPU_CHECK(next->A.ptr.reserve((size_t)newm + 1));
    GPU_CHECK(next->A.col.reserve((size_t)nnz + 1));
    GPU_CHECK(next->A.val.reserve((size_t)nnz + 1));
    GPU_CHECK(next->rl.reserve((size_t)newm + 1));
    GPU_CHECK(next->ru.reserve((size_t)newm + 1));
    GPU_CHECK(next->row_ids.reserve((size_t)newm + 1));
    compact_row_payload<<<(unsigned)std::min((size_t)m + 1, (size_t)65535), 256>>>(
        view(p->A, m, n), p->batch.kr.data, map.data, offsets.data,
        p->batch.rl.data, p->batch.ru.data, p->row_ids.data, next->A.ptr.data,
        next->A.col.data, next->A.val.data, next->rl.data, next->ru.data, next->row_ids.data);
    GPU_CHECK(cudaGetLastError());
    /* No floating-point arithmetic is performed on A/Q/c here. Input CSR was
     * validated and intersected bounds were checked by cuda_prepare_batch. */
    next->annz = nnz; next->qnnz = p->qnnz; next->row_only = true;
    return OK;
}
#endif
