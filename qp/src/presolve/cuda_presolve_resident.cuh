#ifndef CUDA_RESIDENT_CUH
#define CUDA_RESIDENT_CUH
/* Included after the device rule and transformation implementation. */
__global__ static void resident_identity(int n, int *ids) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < n)
        ids[j] = j;
}
__global__ static void resident_input_bounds(int m, int n, const double *l, const double *u,
                                             const double *rl, const double *ru,
                                             const double *offset, int *status) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if ((j < n && l[j] > u[j]) || (j < m && rl[j] > ru[j]))
        atomicExch(status + 1, (int)INFEASIBLE);
    if (!j && !isfinite(*offset))
        atomicExch(status, (int)INVALID_ARGUMENT);
}
static void resident_shape(const CudaPlan *p, Problem *out) {
    memset(out, 0, sizeof(*out));
    out->A.rows = p->m;
    out->A.cols = out->Q.rows = out->Q.cols = p->n;
    out->A.nnz = p->annz;
    out->Q.nnz = p->qnnz;
}
static Status resident_initialize(CudaPlan *p, State *s, Problem *shape) {
    try {
        auto owner = new TapeOwner(std::make_shared<DeviceTape>(p->device, p->m, p->n));
        s->device_tape = owner;
        GPU_CHECK(p->row_ids.reserve((size_t)p->m + 1));
        GPU_CHECK(p->col_ids.reserve((size_t)p->n + 1));
        if (p->m)
            resident_identity<<<((unsigned)p->m + 255) / 256, 256>>>(p->m, p->row_ids.data);
        if (p->n)
            resident_identity<<<((unsigned)p->n + 255) / 256, 256>>>(p->n, p->col_ids.data);
        Buffer<int> error;
        GPU_CHECK(error.reserve(2));
        GPU_CHECK(cudaMemset(error.data, 0, 2 * sizeof(int)));
        int width = std::max(1, std::max(p->m, p->n)), codes[2];
        resident_input_bounds<<<((unsigned)width + 255) / 256, 256>>>(
            p->m, p->n, p->l.data, p->u.data, p->rl.data, p->ru.data, p->offset.data, error.data);
        GPU_CHECK(error.download(codes, 2));
        if (codes[0] || codes[1])
            return (Status)(codes[0] ? codes[0] : codes[1]);
        resident_shape(p, shape);
        s->residency.resident = 1;
        return OK;
    } catch (const std::bad_alloc &) {
        return OUT_OF_MEMORY;
    }
}
extern "C" Status cuda_resident_create(const Problem *source, State *s, void **out,
                                              Problem *shape) {
    Status status = cuda_plan_create(source, s->device, &s->transfers, out);
    if (status)
        return status;
    CudaPlan *p = (CudaPlan *)*out;
    DeviceGuard guard(p->device);
    GPU_CHECK(guard.status);
    TransferScope traffic(&s->transfers);
    GPU_CHECK(p->offset.upload(&source->objective_offset, 1));
    return resident_initialize(p, s, shape);
}
static bool resident_input_pointer(const void *p, size_t count, int device) {
    if (!count)
        return true;
    cudaPointerAttributes attributes = {};
    if (cudaPointerGetAttributes(&attributes, p) != cudaSuccess) {
        cudaGetLastError();
        return false;
    }
    return attributes.type == cudaMemoryTypeManaged ||
           (attributes.type == cudaMemoryTypeDevice && attributes.device == device);
}
__global__ static void resident_input_csr(View a, View q, int annz, int qnnz, int *status) {
    if (!blockIdx.x && !threadIdx.x &&
        (a.ptr[0] != 0 || a.ptr[a.rows] != annz || q.ptr[0] != 0 || q.ptr[q.rows] != qnnz))
        atomicExch(status, (int)INVALID_ARGUMENT);
    for (int which = 0; which < 2; ++which) {
        View v = which ? q : a; int nnz = which ? qnnz : annz;
        for (size_t i = blockIdx.x; i < (size_t)v.rows; i += gridDim.x) {
            int begin = v.ptr[i], end = v.ptr[i + 1];
            if (begin < 0 || end < begin || end > nnz) {
                if (!threadIdx.x) atomicExch(status, (int)INVALID_ARGUMENT);
                continue;
            }
            for (size_t k = (size_t)begin + threadIdx.x; k < (size_t)end; k += blockDim.x)
                if (v.col[k] < 0 || v.col[k] >= v.cols ||
                    (k > (size_t)begin && v.col[k] <= v.col[k - 1]) ||
                    !isfinite(v.val[k]) || v.val[k] == 0.)
                    atomicExch(status, (int)INVALID_ARGUMENT);
        }
    }
}
__global__ static void resident_input_vectors(View q, const double *c, const double *l,
                                               const double *u, int m, const double *rl,
                                               const double *ru, int *status) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    bool valid = true;
    if (i < m) valid = !isnan(rl[i]) && !isnan(ru[i]) && rl[i] != INFINITY && ru[i] != -INFINITY;
    if (i < q.rows) {
        valid = valid && isfinite(c[i]) && !isnan(l[i]) && !isnan(u[i]) &&
                l[i] != INFINITY && u[i] != -INFINITY;
        for (int k = q.ptr[i]; k < q.ptr[i + 1]; ++k)
            if (q.val[k] != q.get(q.col[k], i)) valid = false;
        if (q.get(i, i) < 0.) valid = false;
    }
    if (!valid) atomicExch(status, (int)INVALID_ARGUMENT);
}
template <class T> static cudaError_t resident_copy(Buffer<T> &out, const T *in, size_t count) {
    cudaError_t code = out.reserve(count ? count : 1);
    return code != cudaSuccess
               ? code
               : cuda_memcpy(out.data, in, count * sizeof(T), cudaMemcpyDeviceToDevice);
}
static Status resident_copy_matrix(DeviceMatrix &out, const DeviceMatrixView &in) {
    GPU_CHECK(resident_copy(out.ptr, in.row_ptr, (size_t)in.rows + 1));
    GPU_CHECK(resident_copy(out.col, in.col_ind, in.nnz));
    GPU_CHECK(resident_copy(out.val, in.values, in.nnz));
    return OK;
}
extern "C" Status cuda_validate_device_input(const DeviceProblemView *source) {
    DeviceGuard guard(source->device);
    GPU_CHECK(guard.status);
    const auto &a = source->A;
    const auto &q = source->Q;
    int m = a.rows, n = a.cols;
    if (!resident_input_pointer(a.row_ptr, (size_t)m + 1, source->device) ||
        !resident_input_pointer(a.col_ind, a.nnz, source->device) ||
        !resident_input_pointer(a.values, a.nnz, source->device) ||
        !resident_input_pointer(q.row_ptr, (size_t)n + 1, source->device) ||
        !resident_input_pointer(q.col_ind, q.nnz, source->device) ||
        !resident_input_pointer(q.values, q.nnz, source->device) ||
        !resident_input_pointer(source->c, n, source->device) ||
        !resident_input_pointer(source->lower, n, source->device) ||
        !resident_input_pointer(source->upper, n, source->device) ||
        !resident_input_pointer(source->row_lower, m, source->device) ||
        !resident_input_pointer(source->row_upper, m, source->device) ||
        !resident_input_pointer(source->objective_offset, 1, source->device))
        return INVALID_ARGUMENT;
    try {
        Buffer<int> error;
        GPU_CHECK(error.reserve(1));
        GPU_CHECK(cudaMemset(error.data, 0, sizeof(int)));
        View av{m, n, a.row_ptr, a.col_ind, a.values};
        View qv{n, n, q.row_ptr, q.col_ind, q.values};
        int width = std::max(1, std::max(m, n)), code;
        resident_input_csr<<<std::min(width, 65535), 256>>>(av, qv, a.nnz, q.nnz,
                                                                   error.data);
        GPU_CHECK(error.download(&code, 1));
        if (code)
            return (Status)code;
        resident_input_vectors<<<((unsigned)width + 255) / 256, 256>>>(qv, source->c,
            source->lower, source->upper, m, source->row_lower, source->row_upper, error.data);
        GPU_CHECK(error.download(&code, 1));
        if (code)
            return (Status)code;
        return OK;
    } catch (const std::bad_alloc &) {
        return OUT_OF_MEMORY;
    }
}
extern "C" Status cuda_resident_create_device(const DeviceProblemView *source,
                                                     State *s, void **out, Problem *shape) {
    DeviceGuard guard(s->device);
    GPU_CHECK(guard.status);
    TransferScope traffic(&s->transfers);
    Status validated = cuda_validate_device_input(source);
    if (validated) return validated;
    const auto &a = source->A;
    const auto &q = source->Q;
    int m = a.rows, n = a.cols;
    try {
        std::unique_ptr<CudaPlan> p(new CudaPlan(s->device, m, n, &s->transfers));
        Status status = resident_copy_matrix(p->A, a);
        if (status)
            return status;
        status = resident_copy_matrix(p->Q, q);
        if (status)
            return status;
        GPU_CHECK(resident_copy(p->c, source->c, n));
        GPU_CHECK(resident_copy(p->l, source->lower, n));
        GPU_CHECK(resident_copy(p->u, source->upper, n));
        GPU_CHECK(resident_copy(p->rl, source->row_lower, m));
        GPU_CHECK(resident_copy(p->ru, source->row_upper, m));
        GPU_CHECK(resident_copy(p->offset, source->objective_offset, 1));
        p->annz = a.nnz;
        p->qnnz = q.nnz;
        *out = p.release();
        return resident_initialize((CudaPlan *)*out, s, shape);
    } catch (const std::bad_alloc &) {
        return OUT_OF_MEMORY;
    } catch (const thrust::system_error &) {
        return CUDA_ERROR;
    }
}
__global__ static void resident_candidate_status(const Candidate *c, int n, int *status) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < n && c[j].status) {
        if (c[j].status == NUMERICAL_ERROR)
            atomicAdd(status + 1, 1);
        else
            atomicExch(status, c[j].status);
    }
}
struct ResidentHealthy {
    const Candidate *c;
    __host__ __device__ bool operator()(int j) const {
        return !c[j].status && c[j].action != GPU_NONE;
    }
};
/* The same deterministic greedy ownership policy as the reference. Numerical
 * data and compatibility decisions stay on the device. Independent families
 * use a parallel stable selection instead of this structural transaction. */
__global__ static void resident_select(View a, View q, const Candidate *c, int found,
                                       int limit, int *blocked, int *rows, int *cols, int *owner,
                                       int *selected, int *result) {
    int count = 0;
    for (int t = 0; t < found && count < limit; ++t) {
        Candidate x = c[t];
        if (x.status || !x.action)
            continue;
        if (x.action == GPU_SUB) {
            bool safe = cols[x.j] && !blocked[x.j] && owner[x.row] < 0;
            for (int k = a.ptr[x.row]; k < a.ptr[x.row + 1] && safe; ++k)
                if (!cols[a.col[k]])
                    safe = false;
            if (!safe)
                continue;
            for (int k = a.ptr[x.row]; k < a.ptr[x.row + 1]; ++k)
                blocked[a.col[k]] = 1;
            for (int k = q.ptr[x.j]; k < q.ptr[x.j + 1]; ++k)
                blocked[q.col[k]] = 1;
            cols[x.j] = 0;
            owner[x.row] = x.j;
        } else if (x.action == GPU_PAR_COL) {
            if (!cols[x.k] || !cols[x.j] || blocked[x.j])
                continue;
            blocked[x.k] = 1;
            cols[x.j] = 0;
        } else if (x.action == GPU_PAR_ROW) {
            if (!rows[x.k] || !rows[x.row])
                continue;
            rows[x.row] = 0;
        } else if (x.action == GPU_DUAL) {
            if (owner[x.row] >= 0)
                continue;
            owner[x.row] = x.j;
        }
        selected[count++] = t;
    }
    *result = count;
}
__global__ static void resident_independent_selection(int count, int found, bool reverse, int *out) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < count) out[j] = reverse ? found - j - 1 : j;
}
static Status resident_selection(CudaPlan *p, uint32_t rule, int found, int limit, int *count) {
    GPU_CHECK(p->selected.reserve(found));
    if (rule == PROPAGATION || rule == EMPTY_ROWS || rule == SINGLETON_ROWS ||
        rule == ACTIVITY || rule == FIXED_COLS || rule == EMPTY_COLS ||
        rule == DUAL_FIX) {
        /* Detection compacted active candidates and the caller checked every
         * status. Every independent candidate is healthy here; preserve the
         * same forward/reverse order without another filtering scan/sync. */
        *count = std::min(found, limit);
        if (*count > 0)
            resident_independent_selection<<<((unsigned)*count + 255) / 256, 256>>>(*count, found,
                rule == FIXED_COLS || rule == EMPTY_COLS || rule == DUAL_FIX, p->selected.data);
        GPU_CHECK(cudaGetLastError());
        return OK;
    }
    Buffer<int> blocked, rows, cols, owners, result;
    GPU_CHECK(blocked.reserve((size_t)p->n + 1));
    GPU_CHECK(cols.reserve((size_t)p->n + 1));
    GPU_CHECK(rows.reserve((size_t)p->m + 1));
    GPU_CHECK(owners.reserve((size_t)p->m + 1));
    GPU_CHECK(result.reserve(1));
    GPU_CHECK(cudaMemset(blocked.data, 0, (size_t)p->n * sizeof(int)));
    GPU_CHECK(cudaMemset(owners.data, 0xff, (size_t)p->m * sizeof(int)));
    thrust::fill(thrust::device, rows.data, rows.data + p->m, 1);
    thrust::fill(thrust::device, cols.data, cols.data + p->n, 1);
    resident_select<<<1, 1>>>(view(p->A, p->m, p->n), view(p->Q, p->n, p->n),
                              p->compact.data, found, limit, blocked.data, rows.data, cols.data,
                              owners.data, p->selected.data, result.data);
    GPU_CHECK(result.download(count, 1));
    return OK;
}
__device__ static DeviceRecord
resident_record(View a, View q, View at, const Candidate &c, int z, const int *keep,
                const int *rows, const double *linear, const double *stages, const double *l,
                const double *u, const NewBounds *previous, const NewBounds *latest) {
    DeviceRecord r = {};
    r.m = a.rows;
    r.n = a.cols;
    r.j = c.j;
    r.k = c.k;
    r.row = c.row;
    if (c.action == GPU_DUAL) {
        r.type = -1;
        return r;
    }
    if (c.action == GPU_ROW)
        r.type = REC_ROW;
    else if (c.action == GPU_BOUND || c.action == GPU_SINGLE_ROW) {
        r.type = c.action == GPU_BOUND ? REC_BOUND : REC_SINGLE_ROW;
        r.lower = c.action == GPU_BOUND ? l[c.j] : previous[z].lower;
        r.upper = c.action == GPU_BOUND ? u[c.j] : previous[z].upper;
        r.lower2 = latest[z].lower;
        r.upper2 = latest[z].upper;
        r.pivot = a.get(c.row, c.j);
        r.gsize = a.ptr[c.row + 1] - a.ptr[c.row];
    } else if (c.action == GPU_PAR_ROW) {
        r.type = REC_PAR_ROW;
        r.ratio = c.ratio;
        r.lower = previous[z].lower;
        r.upper = previous[z].upper;
        r.gsize = a.ptr[c.k + 1] - a.ptr[c.k];
    } else if (c.action == GPU_PAR_COL) {
        r.type = REC_PAR_COL;
        r.ratio = c.ratio;
        r.lower = previous[z].lower;
        r.upper = previous[z].upper;
        r.lower2 = l[c.j];
        r.upper2 = u[c.j];
    } else {
        r.type = c.action == GPU_FIXED ? REC_FIXED : REC_SUB;
        r.alpha = c.value;
        r.c = c.action == GPU_FIXED ? stages[c.j] : linear[c.j];
        r.asize = at.ptr[c.j + 1] - at.ptr[c.j];
        if (c.action == GPU_FIXED) {
            for (int k = q.ptr[c.j]; k < q.ptr[c.j + 1]; ++k)
                if (!(q.col[k] > c.j && !keep[q.col[k]]))
                    ++r.qsize;
        } else {
            r.qsize = q.ptr[c.j + 1] - q.ptr[c.j];
            r.retained = rows[c.row];
            r.pivot = a.get(c.row, c.j);
            r.gsize = a.ptr[c.row + 1] - a.ptr[c.row] - 1;
        }
    }
    return r;
}
struct CaptureSize { unsigned long long entries, records; };
struct AddCaptureSize {
    __host__ __device__ CaptureSize operator()(CaptureSize a, CaptureSize b) const {
        return {a.entries + b.entries, a.records + b.records};
    }
};
__global__ static void
resident_record_counts(View a, View q, View at, const Candidate *c, const int *selected,
                       int count, const int *keep, const int *rows, const double *linear,
                       const double *stages, const double *l, const double *u,
                       const NewBounds *previous, const NewBounds *latest, DeviceRecord *out,
                       CaptureSize *sizes) {
    int z = blockIdx.x * blockDim.x + threadIdx.x;
    if (z < count) {
        DeviceRecord r = resident_record(a, q, at, c[selected[z]], z, keep, rows, linear, stages, l,
                                         u, previous, latest);
        out[z] = r;
        sizes[z] = {(unsigned long long)r.qsize + r.asize + r.gsize,
                    (unsigned long long)(r.type >= 0)};
    }
}
template <int lanes> __global__ static void
resident_record_payload(View a, View q, View at, const Candidate *candidates,
                        const int *selected, int count, const int *keep, const int *row_ids,
                        const int *col_ids, const DeviceRecord *input,
                        const CaptureSize *offsets, DeviceRecord *records, int *index,
                        double *values, int record_base, int value_base) {
    size_t z = ((size_t)blockIdx.x * blockDim.x + threadIdx.x) / lanes;
    int lane = threadIdx.x % lanes;
    if (z >= (size_t)count || input[z].type < 0)
        return;
    DeviceRecord r = input[z];
    Candidate c = candidates[selected[z]];
    int pos = value_base + (int)offsets[z].entries;
    r.qoffset = pos;
    if (r.qsize && !lane) {
        for (int k = q.ptr[c.j]; k < q.ptr[c.j + 1]; ++k) {
            if (r.type == REC_FIXED && q.col[k] > c.j && !keep[q.col[k]])
                continue;
            index[pos] = col_ids[q.col[k]];
            values[pos++] = q.val[k];
        }
    }
    r.aoffset = r.qoffset + r.qsize;
    if (r.asize)
        for (size_t k = (size_t)at.ptr[c.j] + lane; k < (size_t)at.ptr[c.j + 1]; k += lanes) {
            pos = r.aoffset + (int)(k - (size_t)at.ptr[c.j]);
            index[pos] = row_ids[at.col[k]];
            values[pos] = at.val[k];
        }
    r.goffset = r.aoffset + r.asize;
    if (r.gsize) {
        int row = r.type == REC_PAR_ROW ? c.k : c.row;
        for (size_t k = (size_t)a.ptr[row] + lane; k < (size_t)a.ptr[row + 1]; k += lanes) {
            if (r.type == REC_SUB && a.col[k] == c.j)
                continue;
            /* Sorted CSR omits exactly the substitution pivot. */
            pos = r.goffset + (int)(k - (size_t)a.ptr[row]) -
                  (r.type == REC_SUB && a.col[k] > c.j);
            index[pos] = col_ids[a.col[k]];
            values[pos] = r.type == REC_SUB ? -a.val[k] / r.pivot : a.val[k];
        }
    }
    if (lane) return;
    bool has_col = r.type == REC_FIXED || r.type == REC_SUB || r.type == REC_BOUND ||
                   r.type == REC_SINGLE_ROW || r.type == REC_PAR_COL;
    bool has_row = r.type == REC_SUB || r.type == REC_ROW || r.type == REC_SINGLE_ROW ||
                   r.type == REC_BOUND || r.type == REC_PAR_ROW;
    if (has_col)
        r.j = col_ids[c.j];
    if (has_row)
        r.row = row_ids[c.row];
    if (r.type == REC_PAR_COL)
        r.k = col_ids[c.k];
    if (r.type == REC_PAR_ROW)
        r.k = row_ids[c.k];
    records[record_base + (int)offsets[z].records] = r;
}
template <int lanes>
static void capture_payload(CudaPlan *p, const DeviceRecord *records,
                            const CaptureSize *offsets, DeviceTapeSegment *tail) {
    int count = p->batch.count;
    resident_record_payload<lanes><<<(unsigned)(((size_t)count * lanes + 255) / 256), 256>>>(
        view(p->A, p->m, p->n), view(p->Q, p->n, p->n), view(p->AT, p->n, p->m),
        p->compact.data, p->selected.data, count, p->batch.kc.data, p->row_ids.data,
        p->col_ids.data, records, offsets, tail->records.data,
        tail->indices.data, tail->values.data, tail->count, tail->entries);
}
static Status resident_capture(CudaPlan *p, State *s) {
    int count = p->batch.count;
    Buffer<DeviceRecord> records;
    Buffer<CaptureSize> offsets;
    GPU_CHECK(records.reserve(count));
    GPU_CHECK(offsets.reserve((size_t)count + 1));
    GPU_CHECK(cudaMemset(offsets.data + count, 0, sizeof(CaptureSize)));
    resident_record_counts<<<((unsigned)count + 255) / 256, 256>>>(
        view(p->A, p->m, p->n), view(p->Q, p->n, p->n), view(p->AT, p->n, p->m),
        p->compact.data, p->selected.data, count, p->batch.kc.data, p->batch.kr.data, p->c.data,
        p->stages.data, p->l.data, p->u.data, p->batch.previous.data, p->batch.latest.data,
        records.data, offsets.data);
    /* Own and initialize the scratch: CUB's multiword look-back state may load
     * a tile payload before its ready flag selects that payload. */
    size_t scratch_bytes = 0;
    GPU_CHECK(cub::DeviceScan::ExclusiveScan(nullptr, scratch_bytes, offsets.data, offsets.data,
                                            AddCaptureSize{}, CaptureSize{0, 0}, count + 1));
    Buffer<unsigned char> scratch;
    GPU_CHECK(scratch.reserve(scratch_bytes));
    GPU_CHECK(cudaMemset(scratch.data, 0, scratch_bytes));
    GPU_CHECK(cub::DeviceScan::ExclusiveScan(scratch.data, scratch_bytes, offsets.data, offsets.data,
                                            AddCaptureSize{}, CaptureSize{0, 0}, count + 1));
    CaptureSize shape;
    GPU_CHECK(cuda_memcpy(&shape, offsets.data + count, sizeof(shape), cudaMemcpyDeviceToHost));
    int added = (int)shape.records;
    unsigned long long entries = shape.entries;
    if (!added)
        return OK;
    if (entries > INT_MAX)
        return OUT_OF_MEMORY;
    DeviceTape &tape = **(TapeOwner *)s->device_tape;
    DeviceTapeSegment *tail = tape.segments.empty() ? nullptr : tape.segments.back().get();
    if (!tail || tail->folding || (size_t)tail->count + added > tail->records.capacity ||
        (size_t)tail->entries + entries > tail->indices.capacity) {
        auto next = std::unique_ptr<DeviceTapeSegment>(new DeviceTapeSegment());
        size_t rc = std::max(added, std::min(65536, std::max(64, tape.cols)));
        size_t ec =
            std::max((size_t)entries, std::min((size_t)524288, std::max((size_t)256, rc * 8)));
        GPU_CHECK(next->records.reserve(rc));
        GPU_CHECK(next->indices.reserve(ec));
        GPU_CHECK(next->values.reserve(ec));
        tape.bytes += rc * sizeof(DeviceRecord) + ec * (sizeof(int) + sizeof(double));
        tape.segments.push_back(std::move(next));
        tail = tape.segments.back().get();
    }
    if (entries >= (unsigned long long)32 * count)
        capture_payload<32>(p, records.data, offsets.data, tail);
    else
        capture_payload<1>(p, records.data, offsets.data, tail);
    GPU_CHECK(cudaGetLastError());
    tail->count += added;
    tail->entries += (int)entries;
    tape.count += added;
    tape.entries += (size_t)entries;
    s->count = tape.count;
    return OK;
}
extern "C" int cuda_resident_apply(State *s, const Config *cfg, uint32_t rule, int limit,
                                       void **plan, double deadline) {
    if (!plan || !*plan)
        return -(int)INVALID_ARGUMENT;
    CudaPlan *p = (CudaPlan *)*plan;
    DeviceGuard guard(p->device);
    if (guard.status != cudaSuccess)
        return -(int)CUDA_ERROR;
    TransferScope traffic(&s->transfers);
    try {
        int cap = rule == EMPTY_ROWS || rule == SINGLETON_ROWS || rule == ACTIVITY ||
                          rule == PARALLEL_ROWS || rule == SINGLETON_EQ ||
                          rule == DOUBLETON_EQ
                      ? p->m
                      : p->n;
        int found;
        Status status = cuda_detect(p, cfg, rule, cap, &found);
        ++s->execution.gpu_candidate_passes;
        if (status || !found)
            return status ? -(int)status : 0;
        if (rule == FIXED_COLS) {
            status = ensure_transpose(p);
            if (status) return -(int)status;
        }
        Buffer<int> flags;
        if (flags.reserve(2) != cudaSuccess ||
            cudaMemset(flags.data, 0, 2 * sizeof(int)) != cudaSuccess)
            return -(int)CUDA_ERROR;
        resident_candidate_status<<<((unsigned)found + 255) / 256, 256>>>(p->compact.data, found,
                                                                          flags.data);
        int codes[2];
        if (flags.download(codes, 2) != cudaSuccess)
            return -(int)CUDA_ERROR;
        if (codes[0])
            return -codes[0];
        s->batches.numerical_rejections += (uint64_t)codes[1];
        if (codes[1])
            return -(int)NUMERICAL_ERROR;
        limit = cuda_batch_limit(p, rule, limit);
        for (;;) {
            if (monotonic_seconds() >= deadline)
                return -(int)TIME_LIMIT;
            int count;
            status = resident_selection(p, rule, found, limit, &count);
            if (status || !count)
                return status ? -(int)status : 0;
            status = cuda_prepare_batch(p, rule, count);
            int rejected = 0;
            if (!status && rule != PROPAGATION && rule != SINGLETON_DUAL) {
                int used = 0;
                if (rule == DOUBLETON_EQ && count == 1)
                    status = resident_doubleton(p, &used);
                if (!status && !used)
                    status = cuda_transform(p, cfg, &rejected);
            }
            if (status == NUMERICAL_ERROR || status == INVALID_ARGUMENT || rejected) {
                if (rejected)
                    ++s->batches.fill_rejections;
                else
                    ++s->batches.numerical_rejections;
                cuda_plan_reject(p, rule, count > 1 ? count / 2 : 0);
                if (count == 1)
                    return rejected ? 0 : -(int)NUMERICAL_ERROR;
                limit = count / 2;
                ++s->batches.retries;
                continue;
            }
            if (status)
                return -(int)status;
            status = resident_capture(p, s);
            if (status)
                return -(int)status;
            if (rule == PROPAGATION || rule == SINGLETON_DUAL)
                status = cuda_publish_bounds(p);
            else
                cuda_plan_commit(plan);
            if (status)
                return -(int)status;
            p = (CudaPlan *)*plan;
            s->reduced_m = p->m;
            s->reduced_n = p->n;
            ++s->execution.gpu_batches;
            s->execution.gpu_reductions += count;
            s->execution.largest_batch = std::max(s->execution.largest_batch, (size_t)count);
            return count;
        }
    } catch (const std::bad_alloc &) {
        return -(int)OUT_OF_MEMORY;
    } catch (const thrust::system_error &) {
        return -(int)CUDA_ERROR;
    }
}
template <class T> static void resident_release(Buffer<T> &buffer) {
    Buffer<T> empty;
    buffer.swap(empty);
}
static void resident_release_scratch(CudaPlan *p) {
    resident_release(p->candidates);
    resident_release(p->compact);
    resident_release(p->row_hash);
    resident_release(p->col_hash);
    resident_release(p->row_order);
    resident_release(p->col_order);
    resident_release(p->selected);
    resident_release(p->stages);
    resident_release(p->propagation_queue);
    resident_release(p->propagation_seen);
    resident_release(p->propagation_count);
    resident_release(p->propagation_dirty_rows);
    resident_release(p->propagation_dirty_flags);
    resident_release(p->propagation_dirty_count);
    resident_release(p->propagation_long_rows);
    resident_release(p->propagation_activity);
    DeviceBatch &b = p->batch;
    resident_release(b.kr);
    resident_release(b.kc);
    resident_release(b.owner);
    resident_release(b.order);
    resident_release(b.error);
    resident_release(b.groups);
    resident_release(b.keys);
    resident_release(b.alpha);
    resident_release(b.l);
    resident_release(b.u);
    resident_release(b.rl);
    resident_release(b.ru);
    resident_release(b.latest);
    resident_release(b.previous);
    resident_release(b.proposals);
    resident_release(b.prefix);
}
extern "C" Status cuda_resident_finish(void **plan, State *s, Problem *shape) {
    CudaPlan *p = (CudaPlan *)*plan;
    DeviceGuard guard(p->device);
    GPU_CHECK(guard.status);
    GPU_CHECK(cudaStreamSynchronize(0));
    DeviceTape &tape = **(TapeOwner *)s->device_tape;
    tape.row_ids.swap(p->row_ids);
    tape.col_ids.swap(p->col_ids);
    tape.reduced_m = p->m;
    tape.reduced_n = p->n;
    tape.bytes += (tape.row_ids.capacity + tape.col_ids.capacity) * sizeof(int);
    resident_release_scratch(p);
    s->residency.iterations = s->transfers;
    s->residency.tape_records = tape.count;
    s->residency.tape_entries = tape.entries;
    s->residency.tape_device_bytes = tape.bytes;
    resident_shape(p, shape);
    s->device_plan = p;
    *plan = nullptr;
    return OK;
}
extern "C" void cuda_resident_tape_free(void **opaque) {
    if (!opaque || !*opaque)
        return;
    auto owner = (TapeOwner *)*opaque;
    DeviceGuard guard((*owner)->device);
    delete owner;
    *opaque = nullptr;
}
extern "C" Status cuda_resident_download(void *opaque, Problem *out, uint64_t *bytes) {
    CudaPlan *p = (CudaPlan *)opaque;
    DeviceGuard guard(p->device);
    GPU_CHECK(guard.status);
    TransferScope traffic(p->transfers);
    Problem model = {};
    Status status = problem_alloc(&model, p->m, p->n);
    if (!status)
        status = download_matrix(p->A, &model.A, p->annz);
    if (!status)
        status = download_matrix(p->Q, &model.Q, p->qnnz);
    if (!status && (p->c.download(model.c, p->n) != cudaSuccess ||
                    p->l.download(model.lower, p->n) != cudaSuccess ||
                    p->u.download(model.upper, p->n) != cudaSuccess ||
                    p->rl.download(model.row_lower, p->m) != cudaSuccess ||
                    p->ru.download(model.row_upper, p->m) != cudaSuccess ||
                    p->offset.download(&model.objective_offset, 1) != cudaSuccess))
        status = CUDA_ERROR;
    if (status)
        problem_free(&model);
    else {
        problem_free(out);
        *out = model;
        *bytes += ((uint64_t)p->m + p->n + 2) * sizeof(int) +
                  ((uint64_t)p->annz + p->qnnz) * (sizeof(int) + sizeof(double)) +
                  ((uint64_t)3 * p->n + 2 * p->m + 1) * sizeof(double);
    }
    return status;
}
extern "C" Status cuda_resident_view_no_transpose(void *opaque, DeviceProblemView *out) {
    CudaPlan *p = (CudaPlan *)opaque;
    out->A = DeviceMatrixView{p->m, p->n, p->annz, p->A.ptr.data, p->A.col.data, p->A.val.data};
    out->Q = DeviceMatrixView{p->n, p->n, p->qnnz, p->Q.ptr.data, p->Q.col.data, p->Q.val.data};
    out->AT =
        DeviceMatrixView{p->n, p->m, p->AT.ptr.data ? p->annz : 0, p->AT.ptr.data, p->AT.col.data, p->AT.val.data};
    out->c = p->c.data;
    out->lower = p->l.data;
    out->upper = p->u.data;
    out->row_lower = p->rl.data;
    out->row_upper = p->ru.data;
    out->objective_offset = p->offset.data;
    out->device = p->device;
    return OK;
}
extern "C" Status cuda_resident_view(void *opaque, DeviceProblemView *out) {
    CudaPlan *p = (CudaPlan *)opaque;
    DeviceGuard guard(p->device); GPU_CHECK(guard.status);
    Status status = ensure_transpose(p);
    return status ? status : cuda_resident_view_no_transpose(opaque, out);
}
extern "C" Status cuda_state_host_clone(const State *s, State **out) {
    if (!s->device_tape)
        return INVALID_ARGUMENT;
    const DeviceTape &tape = **(TapeOwner *)s->device_tape;
    DeviceGuard guard(tape.device);
    GPU_CHECK(guard.status);
    std::unique_ptr<State, void (*)(State *)> copy(
        state_create(s->original_m, s->original_n), [](State *p) { state_free(&p); });
    if (!copy)
        return OUT_OF_MEMORY;
    copy->reduced_m = s->reduced_m;
    copy->reduced_n = s->reduced_n;
    copy->tolerance = s->tolerance;
    copy->capacity = s->count;
    copy->records = (Record *)checked_calloc(s->count, sizeof(Record));
    if (!copy->records)
        return OUT_OF_MEMORY;
    GPU_CHECK(tape.row_ids.download(copy->row_ids, s->reduced_m));
    GPU_CHECK(tape.col_ids.download(copy->col_ids, s->reduced_n));
    uint64_t bytes = ((uint64_t)s->reduced_m + s->reduced_n) * sizeof(int);
    try {
        for (const auto &seg : tape.segments) {
            std::vector<DeviceRecord> records(seg->count);
            std::vector<int> index(seg->entries);
            std::vector<double> values(seg->entries);
            GPU_CHECK(seg->records.download(records.data(), records.size()));
            GPU_CHECK(seg->indices.download(index.data(), index.size()));
            GPU_CHECK(seg->values.download(values.data(), values.size()));
            bytes += records.size() * sizeof(DeviceRecord) +
                     index.size() * (sizeof(int) + sizeof(double));
            for (const auto &d : records) {
                Record &r = copy->records[copy->count++];
                r.type = (RecordType)d.type;
                r.m = d.m;
                r.n = d.n;
                r.j = d.j;
                r.k = d.k;
                r.row = d.row;
                r.retained_row = d.retained;
                r.alpha = d.alpha;
                r.pivot = d.pivot;
                r.ratio = d.ratio;
                r.c = d.c;
                r.lower = d.lower;
                r.upper = d.upper;
                r.lower2 = d.lower2;
                r.upper2 = d.upper2;
                auto vector = [&](Vector &v, int start, int count) {
                    v.size = count;
                    if (!count)
                        return true;
                    v.index = (int *)checked_calloc(count, sizeof(int));
                    v.value = (double *)checked_calloc(count, sizeof(double));
                    if (!v.index || !v.value)
                        return false;
                    memcpy(v.index, index.data() + start, (size_t)count * sizeof(int));
                    memcpy(v.value, values.data() + start, (size_t)count * sizeof(double));
                    return true;
                };
                if (!vector(r.qrow, d.qoffset, d.qsize) || !vector(r.acol, d.aoffset, d.asize) ||
                    !vector(r.coefficients, d.goffset, d.gsize))
                    return OUT_OF_MEMORY;
            }
        }
        copy->residency.tape_export_bytes = bytes;
        *out = copy.release();
        return OK;
    } catch (const std::bad_alloc &) {
        return OUT_OF_MEMORY;
    }
}
extern "C" Status cuda_state_copy_ids(const State *s, int *rows, int *cols) {
    const auto &tape = **(TapeOwner *)s->device_tape;
    DeviceGuard guard(tape.device);
    GPU_CHECK(guard.status);
    GPU_CHECK(tape.row_ids.download(rows, s->reduced_m));
    GPU_CHECK(tape.col_ids.download(cols, s->reduced_n));
    return OK;
}
#endif
