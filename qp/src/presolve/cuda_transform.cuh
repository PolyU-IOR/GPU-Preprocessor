#pragma once

static Status cuda_transform(void *opaque, const Config *cfg, int *fill_rejected) {
    CudaPlan *p = (CudaPlan *)opaque;
    DeviceGuard guard(p->device);
    GPU_CHECK(guard.status);
    TransferScope traffic(p->transfers);
    *fill_rejected = 0;
    if (p->batch.rule == EMPTY_ROWS || p->batch.rule == SINGLETON_ROWS ||
        p->batch.rule == ACTIVITY || p->batch.rule == PARALLEL_ROWS)
        return cuda_transform_rows(p);
    int m = p->m, n = p->n;
    int newm = thrust::reduce(thrust::device, p->batch.kr.data, p->batch.kr.data + m, 0);
    int newn = thrust::reduce(thrust::device, p->batch.kc.data, p->batch.kc.data + n, 0);
    Status status;
    delete p->pending;
    p->pending = new (std::nothrow) CudaPlan(p->device, newm, newn, p->transfers);
    if (!p->pending)
        return OUT_OF_MEMORY;
    CudaPlan *next = p->pending;
    try {
        Buffer<int> &kr = p->batch.kr, &kc = p->batch.kc, &own = p->batch.owner;
        Buffer<double> &al = p->batch.alpha, &lower = p->batch.rl, &upper = p->batch.ru;
        Buffer<int> rmap, cmap, error;
        Buffer<double> g, delta, sc, ol, ou, nc;
        Buffer<unsigned long long> ac, qc, ao, qo;
        DeviceMatrix transform;
        int tnnz = 0;
        status = build_transform(p, kc, own, transform, &tnnz);
        if (status)
            return status;
        GPU_CHECK(rmap.reserve((size_t)m + 1));
        GPU_CHECK(cmap.reserve((size_t)n + 1));
        if (m)
            thrust::exclusive_scan(thrust::device, kr.data, kr.data + m, rmap.data);
        if (n)
            thrust::exclusive_scan(thrust::device, kc.data, kc.data + n, cmap.data);
        GPU_CHECK(error.reserve(1));
        GPU_CHECK(cudaMemset(error.data, 0, sizeof(int)));
        GPU_CHECK(ac.reserve((size_t)m + 1));
        GPU_CHECK(qc.reserve((size_t)n + 1));
        GPU_CHECK(ao.reserve((size_t)m + 1));
        GPU_CHECK(qo.reserve((size_t)n + 1));
        GPU_CHECK(cudaMemset(ac.data, 0, ((size_t)m + 1) * sizeof(unsigned long long)));
        GPU_CHECK(cudaMemset(qc.data, 0, ((size_t)n + 1) * sizeof(unsigned long long)));
        int width = m > n ? m : n;
        View a = view(p->A, m, n), q = view(p->Q, n, n), t = view(transform, n, n);
        if (width)
            transform_counts<<<((unsigned)width + 255) / 256, 256>>>(a, q, t, kr.data, own.data,
                                                                     ac.data, qc.data);
        thrust::exclusive_scan(thrust::device, ac.data, ac.data + m + 1, ao.data);
        thrust::exclusive_scan(thrust::device, qc.data, qc.data + n + 1, qo.data);
        unsigned long long an, qn;
        GPU_CHECK(cuda_memcpy(&an, ao.data + m, sizeof(an), cudaMemcpyDeviceToHost));
        GPU_CHECK(cuda_memcpy(&qn, qo.data + n, sizeof(qn), cudaMemcpyDeviceToHost));
        DeviceCOO amat, qmat, linear;
        status = amat.allocate(an);
        if (status)
            return status;
        status = qmat.allocate(qn);
        if (status)
            return status;
        status = linear.allocate(tnnz);
        if (status)
            return status;
        if (m)
            emit_a<<<((unsigned)m + 255) / 256, 256>>>(a, t, kr.data, own.data, rmap.data,
                                                       cmap.data, ao.data, newn, amat.keys.data,
                                                       amat.values.data, error.data);
        if (n)
            emit_q<<<((unsigned)n + 255) / 256, 256>>>(
                q, t, cmap.data, qo.data, newn, qmat.keys.data, qmat.values.data, error.data);
        GPU_CHECK(g.reserve((size_t)n + 1));
        GPU_CHECK(delta.reserve((size_t)n + 1));
        GPU_CHECK(sc.reserve((size_t)n + 1));
        GPU_CHECK(ol.reserve((size_t)newm + 1));
        GPU_CHECK(ou.reserve((size_t)newm + 1));
        GPU_CHECK(nc.reserve((size_t)newn + 1));
        GPU_CHECK(cudaMemset(nc.data, 0, ((size_t)newn + 1) * sizeof(double)));
        if (width)
            scalar_updates<<<((unsigned)width + 255) / 256, 256>>>(
                a, q, p->c.data, p->l.data, p->u.data, lower.data, upper.data, al.data, kc.data,
                kr.data, own.data, rmap.data, g.data, ol.data, ou.data, delta.data, sc.data,
                error.data);
        if (n)
            emit_linear<<<((unsigned)n + 255) / 256, 256>>>(t, cmap.data, g.data, linear.keys.data,
                                                            linear.values.data, error.data);
        status = qmat.reduce(error.data);
        if (status)
            return status;
        status = qmat.finish(newn, newn, true, &next->Q);
        if (status)
            return status;
        bool substitution = p->batch.rule == SINGLETON_EQ ||
                            p->batch.rule == DOUBLETON_EQ;
        if (substitution &&
            ((cfg->max_q_fill >= 0 &&
              (long long)qmat.count - p->qnnz > (long long)cfg->max_q_fill * p->batch.count) ||
             (isfinite(cfg->max_q_fill_ratio) &&
              qmat.count > cfg->max_q_fill_ratio * std::max(1, p->qnnz)))) {
            *fill_rejected = 1;
            return OK;
        }
        status = amat.reduce(error.data);
        if (status)
            return status;
        status = linear.reduce(error.data);
        if (status)
            return status;
        int code;
        GPU_CHECK(error.download(&code, 1));
        if (code)
            return (Status)code;
        status = amat.finish(newm, newn, false, &next->A);
        if (status)
            return status;
        if (linear.count)
            scatter_linear<<<((unsigned)linear.count + 255) / 256, 256>>>(
                linear.count, linear.unique_keys.data, linear.unique_values.data, nc.data);
        bool fixed = p->batch.rule == FIXED_COLS || p->batch.rule == EMPTY_COLS ||
                     p->batch.rule == DUAL_FIX;
        bool shifted = fixed || p->batch.rule == SINGLETON_EQ ||
                       p->batch.rule == DOUBLETON_EQ;
        {
            GPU_CHECK(next->offset.reserve(1));
            resident_offset<<<1, 1>>>(p->offset.data, delta.data, n, shifted, next->offset.data,
                                      error.data);
            GPU_CHECK(error.download(&code, 1));
            if (code)
                return (Status)code;
            if (fixed)
                p->stages.swap(sc);
        }
        next->c.swap(nc);
        next->rl.swap(ol);
        next->ru.swap(ou);
        GPU_CHECK(next->l.reserve((size_t)newn + 1));
        GPU_CHECK(next->u.reserve((size_t)newn + 1));
        if (n)
            compact_bounds<<<((unsigned)n + 255) / 256, 256>>>(n, kc.data, cmap.data,
                                                               p->batch.l.data, p->batch.u.data,
                                                               next->l.data, next->u.data);
        status = device_transpose(next->A, newm, newn, amat.count, next->AT);
        if (status)
            return status;
        GPU_CHECK(cudaMemset(error.data, 0, sizeof(int)));
        int check_width = std::max(newm, newn);
        validate_transformed<<<((unsigned)check_width + 256) / 256, 256>>>(
            view(next->A, newm, newn), view(next->Q, newn, newn), amat.count, qmat.count,
            next->c.data, next->l.data, next->u.data, next->rl.data, next->ru.data, error.data);
        GPU_CHECK(cudaGetLastError());
        GPU_CHECK(error.download(&code, 1));
        if (code)
            return (Status)code;
        next->annz = amat.count;
        next->qnnz = qmat.count;
        {
            GPU_CHECK(next->row_ids.reserve((size_t)newm + 1));
            GPU_CHECK(next->col_ids.reserve((size_t)newn + 1));
            if (m)
                compact_stable_ids<<<((unsigned)m + 255) / 256, 256>>>(
                    m, kr.data, rmap.data, p->row_ids.data, next->row_ids.data);
            if (n)
                compact_stable_ids<<<((unsigned)n + 255) / 256, 256>>>(
                    n, kc.data, cmap.data, p->col_ids.data, next->col_ids.data);
            GPU_CHECK(cudaGetLastError());
            return OK;
        }
    } catch (const std::bad_alloc &) {
        return OUT_OF_MEMORY;
    } catch (const thrust::system_error &) {
        return CUDA_ERROR;
    }
}
