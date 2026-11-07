#ifndef CUDA_RESIDENT_RECOVERY_CUH
#define CUDA_RESIDENT_RECOVERY_CUH
struct ResidentAccess {
    int *last_read, *last_write;
    int level = 0;
    bool update = false;
    __device__ void read(int id) {
        if (update)
            last_read[id] = max(last_read[id], level);
        else
            level = max(level, last_write[id] + 1);
    }
    __device__ void write(int id) {
        if (update)
            last_write[id] = level;
        else
            level = max(level, max(last_read[id], last_write[id]) + 1);
    }
};
__device__ static void resident_access(const DeviceRecord &r, const int *index, int n,
                                       ResidentAccess &a) {
    if (r.type == REC_FOLD_ROW || r.type == REC_FOLD_COL) {
        bool col = r.type == REC_FOLD_COL;
        if (col) {
            a.read(r.j);
            a.read(n + r.j);
        } else
            a.read(2 * n + r.row);
        for (int k = 0; k < r.gsize; ++k) {
            int id = index[r.goffset + k];
            if (col) {
                a.write(id);
                a.write(n + id);
            } else
                a.write(2 * n + id);
        }
    } else if (r.type == REC_FIXED || r.type == REC_SUB) {
        a.write(r.j);
        a.write(n + r.j);
        for (int k = 0; k < r.gsize; ++k)
            a.read(index[r.goffset + k]);
        for (int k = 0; k < r.qsize; ++k)
            a.read(index[r.qoffset + k]);
        for (int k = 0; k < r.asize; ++k)
            if (index[r.aoffset + k] != r.row)
                a.read(2 * n + index[r.aoffset + k]);
        if (r.type == REC_SUB) {
            a.write(2 * n + r.row);
            if (r.retained)
                a.read(2 * n + r.row);
        }
    } else if (r.type == REC_ROW)
        a.write(2 * n + r.row);
    else if (r.type == REC_BOUND || r.type == REC_SINGLE_ROW) {
        a.read(r.j);
        a.read(n + r.j);
        a.read(2 * n + r.row);
        a.write(2 * n + r.row);
        for (int k = 0; k < r.gsize; ++k) {
            a.read(n + index[r.goffset + k]);
            a.write(n + index[r.goffset + k]);
        }
    } else if (r.type == REC_PAR_ROW) {
        for (int k = 0; k < r.gsize; ++k)
            a.read(index[r.goffset + k]);
        a.read(2 * n + r.k);
        a.write(2 * n + r.k);
        a.write(2 * n + r.row);
    } else {
        a.read(r.k);
        a.read(n + r.k);
        a.write(r.j);
        a.write(r.k);
        a.write(n + r.j);
    }
}
struct ResidentLevelJob {
    const DeviceRecord *records;
    const int *indices;
    int count, reserved;
    int *levels;
};
__global__ static void resident_level_job(ResidentLevelJob *jobs, int slot,
                                          const DeviceRecord *records, const int *indices,
                                          int count, int *levels) {
    jobs[slot] = ResidentLevelJob{records, indices, count, 0, levels};
}
/* Segments are replay barriers, so their dependency plans are independent.
 * Plan several at once on different SMs; preserve the original exact joint
 * x/y/z hazard levels within each segment. No record data goes through CPU. */
__global__ static void resident_levels(const ResidentLevelJob *jobs, int n, int *last_read,
                                       int *last_write, size_t stride) {
    const auto &job = jobs[blockIdx.x];
    for (int k = job.count; k > 0; --k) {
        ResidentAccess access;
        access.last_read = last_read + (size_t)blockIdx.x * stride;
        access.last_write = last_write + (size_t)blockIdx.x * stride;
        resident_access(job.records[k - 1], job.indices, n, access);
        job.levels[k - 1] = access.level;
        access.update = true;
        resident_access(job.records[k - 1], job.indices, n, access);
    }
}
struct ResidentLevelScratch {
    Buffer<int> reads, writes, levels;
    Buffer<ResidentLevelJob> jobs;
    cudaError_t reserve(size_t slots, size_t coordinates, size_t records) {
        cudaError_t status = reads.reserve(slots * coordinates);
        if (!status)
            status = writes.reserve(slots * coordinates);
        if (!status)
            status = levels.reserve(slots * records);
        if (!status)
            status = jobs.reserve(slots);
        return status;
    }
};
__global__ static void resident_level_counts(int count, const int *levels, int *starts, int *order,
                                             int *range) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k < count) {
        order[k] = k;
        atomicAdd(starts + levels[k] + 1, 1);
        atomicMax(range, levels[k] + 1);
    }
}
__global__ static void resident_level_width(int count, const int *starts, int *range) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k <= count)
        atomicMax(range + 1, starts[k]);
}
static Status prepare_resident_recovery(const State *s, CudaRecovery *w) {
    const auto &owner = *(TapeOwner *)s->device_tape;
    if (w->device != owner->device)
        return INVALID_ARGUMENT;
    if ((size_t)s->original_n > (INT_MAX - (size_t)s->original_m) / 2)
        return OUT_OF_MEMORY;
    try {
        size_t coordinates = (size_t)2 * s->original_n + s->original_m;
        size_t stride = coordinates + 1, record_stride = 1;
        // Keep tape ordinals for reverse barriers; folding segments have no
        // dependency plan and must not enlarge scratch or trigger allocations.
        w->resident_plans.resize(owner->segments.size());
        std::vector<size_t> planned;
        for (size_t k = 0; k < owner->segments.size(); ++k) {
            const auto &segment = owner->segments[k];
            if (!segment->folding) {
                planned.push_back(k);
                record_stride = std::max(record_stride, (size_t)segment->count + 1);
            }
        }
        size_t slots = 0;
        std::unique_ptr<ResidentLevelScratch> scratch;
        if (!planned.empty()) {
            size_t free_bytes, total_bytes;
            GPU_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
            /* Transient, per-workspace scratch: at most 1/8 of free memory and
             * 4 GiB, with a one-segment floor. Shrink on allocation pressure. */
            size_t budget = std::min((size_t)4 * 1024 * 1024 * 1024, free_bytes / 8);
            size_t per_slot = (2 * stride + record_stride) * sizeof(int);
            slots = std::max((size_t)1,
                             std::min(planned.size(), std::min((size_t)128, budget / per_slot)));
            for (;;) {
                auto candidate = std::unique_ptr<ResidentLevelScratch>(new ResidentLevelScratch());
                cudaError_t status = candidate->reserve(slots, stride, record_stride);
                if (status == cudaSuccess) {
                    scratch = std::move(candidate);
                    break;
                }
                if (status != cudaErrorMemoryAllocation || slots == 1)
                    return status == cudaErrorMemoryAllocation ? OUT_OF_MEMORY : CUDA_ERROR;
                cudaGetLastError();
                slots = std::max((size_t)1, slots / 2);
            }
        }
        Buffer<int> range;
        if (!planned.empty())
            GPU_CHECK(range.reserve(2));
        for (size_t base = 0; base < planned.size(); base += slots) {
            size_t batch = std::min(slots, planned.size() - base);
            GPU_CHECK(cudaMemset(scratch->reads.data, 0xff, batch * stride * sizeof(int)));
            GPU_CHECK(cudaMemset(scratch->writes.data, 0xff, batch * stride * sizeof(int)));
            for (size_t slot = 0; slot < batch; ++slot) {
                const auto &segment = *owner->segments[planned[base + slot]];
                auto plan = std::unique_ptr<ResidentReplaySegment>(new ResidentReplaySegment());
                GPU_CHECK(plan->order.reserve((size_t)segment.count + 1));
                GPU_CHECK(plan->starts.reserve((size_t)segment.count + 1));
                resident_level_job<<<1, 1>>>(scratch->jobs.data, (int)slot, segment.records.data,
                                             segment.indices.data, segment.count,
                                             scratch->levels.data + slot * record_stride);
                w->resident_plans[planned[base + slot]] = std::move(plan);
            }
            resident_levels<<<(unsigned)batch, 1>>>(scratch->jobs.data, s->original_n,
                                                    scratch->reads.data, scratch->writes.data,
                                                    stride);
            for (size_t slot = 0; slot < batch; ++slot) {
                int count = owner->segments[planned[base + slot]]->count;
                auto &plan = *w->resident_plans[planned[base + slot]];
                int *levels = scratch->levels.data + slot * record_stride;
                GPU_CHECK(cudaMemset(plan.starts.data, 0, ((size_t)count + 1) * sizeof(int)));
                GPU_CHECK(cudaMemset(range.data, 0, 2 * sizeof(int)));
                resident_level_counts<<<((unsigned)count + 255) / 256, 256>>>(
                    count, levels, plan.starts.data, plan.order.data, range.data);
                resident_level_width<<<((unsigned)count + 256) / 256, 256>>>(
                    count, plan.starts.data, range.data);
                thrust::stable_sort_by_key(thrust::device, levels, levels + count, plan.order.data);
                thrust::inclusive_scan(thrust::device, plan.starts.data,
                                       plan.starts.data + count + 1, plan.starts.data);
                int stats[2];
                GPU_CHECK(range.download(stats, 2));
                plan.levels = stats[0];
                plan.max_width = stats[1];
            }
        }
        GPU_CHECK(w->data.reserve(coordinates + 1));
        GPU_CHECK(w->input.reserve((size_t)2 * s->reduced_n + s->reduced_m + 1));
        GPU_CHECK(w->status.reserve(2));
        w->resident = owner;
        w->rows = s->original_m;
        w->cols = s->original_n;
        w->generation = s->generation;
        return OK;
    } catch (const std::bad_alloc &) {
        return OUT_OF_MEMORY;
    } catch (const thrust::system_error &) {
        return CUDA_ERROR;
    }
}
#endif
