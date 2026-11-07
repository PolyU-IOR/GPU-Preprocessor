#include "cuda_common.cuh"
#include "cuda_tape.cuh"
#include <algorithm>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/system_error.h>
#include <vector>

struct ResidentReplaySegment {
    int levels = 0, max_width = 0;
    Buffer<int> order, starts;
};
struct CudaRecovery {
    int device, rows = 0, cols = 0, max_width = 0;
    uint64_t generation = 0;
    std::vector<int> levels;
    Buffer<DeviceRecord> records;
    Buffer<int> index, row_ids, col_ids, order, starts, status;
    Buffer<double> values, data, input;
    TapeOwner resident;
    std::vector<std::unique_ptr<ResidentReplaySegment>> resident_plans;
    explicit CudaRecovery(int d) : device(d) {}
};
static Status prepare_recovery(const State *s, CudaRecovery *w);
extern "C" Status cuda_postsolve_create(const State *s, int device, void **out) {
    DeviceGuard guard(device);
    GPU_CHECK(guard.status);
    CudaRecovery *w = new (std::nothrow) CudaRecovery(device);
    if (!w)
        return OUT_OF_MEMORY;
    Status status = prepare_recovery(s, w);
    if (status)
        delete w;
    else
        *out = w;
    return status;
}
extern "C" void cuda_postsolve_free(void **out) {
    if (!out || !*out)
        return;
    CudaRecovery *w = (CudaRecovery *)*out;
    DeviceGuard guard(w->device);
    delete w;
    *out = nullptr;
}
template <class Read, class Write>
static void accesses(const Record &r, int n, Read read, Write write) {
    auto xread = [&](const Vector &v) {
        for (int k = 0; k < v.size; ++k)
            read(v.index[k]);
    };
    if (r.type == REC_FOLD_ROW || r.type == REC_FOLD_COL) {
        bool col = r.type == REC_FOLD_COL;
        if (col) {
            read(r.j);
            read(n + r.j);
        } else
            read(2 * n + r.row);
        for (int k = 0; k < r.coefficients.size; ++k) {
            int id = r.coefficients.index[k];
            if (col) {
                write(id);
                write(n + id);
            } else
                write(2 * n + id);
        }
    } else if (r.type == REC_FIXED || r.type == REC_SUB) {
        write(r.j);
        write(n + r.j);
        xread(r.coefficients);
        xread(r.qrow);
        for (int k = 0; k < r.acol.size; ++k)
            if (r.acol.index[k] != r.row)
                read(2 * n + r.acol.index[k]);
        if (r.type == REC_SUB) {
            write(2 * n + r.row);
            if (r.retained_row)
                read(2 * n + r.row);
        }
    } else if (r.type == REC_ROW)
        write(2 * n + r.row);
    else if (r.type == REC_BOUND || r.type == REC_SINGLE_ROW) {
        read(r.j);
        read(n + r.j);
        read(2 * n + r.row);
        write(2 * n + r.row);
        for (int k = 0; k < r.coefficients.size; ++k) {
            read(n + r.coefficients.index[k]);
            write(n + r.coefficients.index[k]);
        }
    } else if (r.type == REC_PAR_ROW) {
        xread(r.coefficients);
        read(2 * n + r.k);
        write(2 * n + r.k);
        write(2 * n + r.row);
    } else {
        read(r.k);
        read(n + r.k);
        write(r.j);
        write(r.k);
        write(n + r.j);
    }
}
#include "cuda_postsolve_prepare.cuh"
static Status prepare_recovery(const State *s, CudaRecovery *w) {
    if (w->generation == s->generation)
        return OK;
    w->generation = 0; /* Failed replacement must never reuse a partially overwritten tape. */
    w->resident.reset();
    w->resident_plans.clear();
    if (s->device_tape)
        return prepare_resident_recovery(s, w);
    if (s->count > INT_MAX || (size_t)s->original_n > (INT_MAX - (size_t)s->original_m) / 2)
        return OUT_OF_MEMORY;
    try {
        std::vector<DeviceRecord> records;
        std::vector<int> indices, levels(s->count),
            last_read((size_t)2 * s->original_n + s->original_m, -1), last_write(last_read);
        std::vector<double> values;
        records.reserve(s->count);
        auto pack = [&](const Vector &v, int &offset, int &size) {
            if (indices.size() + (size_t)v.size > INT_MAX)
                throw std::bad_alloc();
            offset = (int)indices.size();
            size = v.size;
            if (v.size) {
                indices.insert(indices.end(), v.index, v.index + v.size);
                values.insert(values.end(), v.value, v.value + v.size);
            }
        };
        for (size_t k = 0; k < s->count; ++k) {
            const Record &r = s->records[k];
            DeviceRecord d = {};
            d.type = r.type;
            d.j = r.j;
            d.k = r.k;
            d.row = r.row;
            d.retained = r.retained_row;
            d.alpha = r.alpha;
            d.pivot = r.pivot;
            d.ratio = r.ratio;
            d.c = r.c;
            d.lower = r.lower;
            d.upper = r.upper;
            d.lower2 = r.lower2;
            d.upper2 = r.upper2;
            pack(r.qrow, d.qoffset, d.qsize);
            pack(r.acol, d.aoffset, d.asize);
            pack(r.coefficients, d.goffset, d.gsize);
            records.push_back(d);
        }
        int max_level = -1;
        /* A level contains only records with disjoint write/read-write sets.
         * Read-after-write, write-after-read and write-after-write all preserve
         * chronological reverse order; no cross-block synchronization is assumed. */
        for (size_t step = s->count; step > 0; --step) {
            size_t k = step - 1;
            int level = 0;
            accesses(
                s->records[k], s->original_n,
                [&](int id) { level = std::max(level, last_write[id] + 1); },
                [&](int id) {
                    level = std::max(level, std::max(last_write[id], last_read[id]) + 1);
                });
            levels[k] = level;
            max_level = std::max(max_level, level);
            accesses(
                s->records[k], s->original_n,
                [&](int id) { last_read[id] = std::max(last_read[id], level); },
                [&](int id) { last_write[id] = level; });
        }
        std::vector<int> starts((size_t)max_level + 2, 0), order(s->count);
        for (int level : levels)
            ++starts[level + 1];
        for (size_t k = 1; k < starts.size(); ++k)
            starts[k] += starts[k - 1];
        std::vector<int> cursor(starts);
        for (size_t step = s->count; step > 0; --step) {
            size_t k = step - 1;
            order[cursor[levels[k]]++] = (int)k;
        }
        GPU_CHECK(w->records.upload(records.data(), records.size()));
        GPU_CHECK(w->index.upload(indices.data(), indices.size()));
        GPU_CHECK(w->values.upload(values.data(), values.size()));
        GPU_CHECK(w->order.upload(order.data(), order.size()));
        GPU_CHECK(w->starts.upload(starts.data(), starts.size()));
        GPU_CHECK(w->row_ids.upload(s->row_ids, s->reduced_m));
        GPU_CHECK(w->col_ids.upload(s->col_ids, s->reduced_n));
        GPU_CHECK(w->data.reserve((size_t)2 * s->original_n + s->original_m + 1));
        GPU_CHECK(w->input.reserve((size_t)2 * s->reduced_n + s->reduced_m + 1));
        GPU_CHECK(w->status.reserve(2));
        w->max_width = 0;
        for (size_t k = 1; k < starts.size(); ++k)
            w->max_width = std::max(w->max_width, starts[k] - starts[k - 1]);
        w->levels = std::move(starts);
        w->rows = s->original_m;
        w->cols = s->original_n;
        w->generation = s->generation;
        return OK;
    } catch (const std::bad_alloc &) {
        return OUT_OF_MEMORY;
    }
}
__global__ static void scatter_reduced(int nr, int mr, const int *cols, const int *rows,
                                       const double *xr, const double *yr, const double *zr,
                                       double *x, double *y, double *z, int *status) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < nr) {
        x[cols[i]] = xr[i];
        z[cols[i]] = zr[i];
        if (!isfinite(xr[i]) || !isfinite(zr[i]))
            atomicExch(status, (int)INVALID_ARGUMENT);
    }
    if (i < mr) {
        y[rows[i]] = yr[i];
        if (!isfinite(yr[i]))
            atomicExch(status, (int)INVALID_ARGUMENT);
    }
}
__device__ static double replay_dot(int offset, int size, const int *indices, const double *values,
                                    const double *x) {
    double sum = 0.;
    for (int k = offset; k < offset + size; ++k)
        sum += values[k] * x[indices[k]];
    return sum;
}
__device__ static bool replay_at(double x, double bound, double tol) {
    return isfinite(bound) && fabs(x - bound) <= tol * (1. + fabs(bound));
}
__device__ static bool replay_valid(double x, double z, double lo, double hi, double tol) {
    return z == 0. || (z > 0. && replay_at(x, lo, tol)) || (z < 0. && replay_at(x, hi, tol));
}
__device__ static void replay_record(const DeviceRecord &r, const int *index, const double *v,
                                     double *x, double *y, double *z, double tol, int *status) {
    if (r.type == REC_FOLD_COL) {
        double value = x[r.j], dual = z[r.j] / r.gsize;
        for (int k = 0; k < r.gsize; ++k) {
            int j = index[r.goffset + k];
            x[j] = value;
            z[j] = dual;
        }
    } else if (r.type == REC_FOLD_ROW) {
        double dual = y[r.row] / r.gsize;
        for (int k = 0; k < r.gsize; ++k)
            y[index[r.goffset + k]] = dual;
    } else if (r.type == REC_FIXED || r.type == REC_SUB) {
        x[r.j] = r.alpha + replay_dot(r.goffset, r.gsize, index, v, x);
        if (r.type == REC_SUB && !r.retained)
            y[r.row] = 0.;
        double gradient = r.c + replay_dot(r.qoffset, r.qsize, index, v, x), other = 0.;
        for (int k = r.aoffset; k < r.aoffset + r.asize; ++k)
            if (index[k] != r.row)
                other += v[k] * y[index[k]];
        if (r.type == REC_FIXED)
            z[r.j] = gradient - other;
        else {
            z[r.j] = r.retained ? y[r.row] : 0.;
            y[r.row] = (gradient - other - z[r.j]) / r.pivot;
        }
    } else if (r.type == REC_ROW)
        y[r.row] = 0.;
    else if (r.type == REC_BOUND || r.type == REC_SINGLE_ROW) {
        if (r.type == REC_SINGLE_ROW)
            y[r.row] = 0.;
        bool unchanged =
            (z[r.j] > 0. && r.lower2 == r.lower) || (z[r.j] < 0. && r.upper2 == r.upper);
        if (!unchanged && !replay_valid(x[r.j], z[r.j], r.lower, r.upper, tol)) {
            double delta = z[r.j] / r.pivot;
            y[r.row] += delta;
            for (int k = r.goffset; k < r.goffset + r.gsize; ++k)
                if (index[k] != r.j)
                    z[index[k]] -= v[k] * delta;
            z[r.j] = 0.;
        }
    } else if (r.type == REC_PAR_ROW) {
        y[r.row] = 0.;
        double activity = replay_dot(r.goffset, r.gsize, index, v, x), dual = y[r.k];
        if (!replay_valid(activity, dual, r.lower, r.upper, tol)) {
            y[r.row] = dual / r.ratio;
            y[r.k] = 0.;
        }
    } else {
        double sum = x[r.k], lo = r.lower2, hi = r.upper2;
        double a = (sum - r.upper) / r.ratio, b = (sum - r.lower) / r.ratio;
        if (r.ratio < 0.) {
            double t = a;
            a = b;
            b = t;
        }
        lo = fmax(lo, a);
        hi = fmin(hi, b);
        if (lo > hi + tol * (1. + fabs(sum))) {
            atomicExch(status, (int)NUMERICAL_ERROR);
            return;
        }
        double value = isfinite(lo) && isfinite(hi) ? .5 * lo + .5 * hi : fmin(fmax(0., lo), hi);
        x[r.j] = value;
        x[r.k] = sum - r.ratio * value;
        z[r.j] = r.ratio * z[r.k];
    }
}
__global__ static void unfold_classes(const DeviceRecord *records, const int *index, double *x,
                                      double *y, double *z, const int *status) {
    if (status[1])
        return;
    const DeviceRecord &r = records[blockIdx.x];
    __shared__ double value, dual;
    if (!threadIdx.x) {
        value = r.type == REC_FOLD_COL ? x[r.j] : 0.;
        dual = (r.type == REC_FOLD_COL ? z[r.j] : y[r.row]) / r.gsize;
    }
    __syncthreads();
    for (int k = threadIdx.x; k < r.gsize; k += blockDim.x) {
        int id = index[r.goffset + k];
        if (r.type == REC_FOLD_COL) {
            x[id] = value;
            z[id] = dual;
        } else
            y[id] = dual;
    }
}
__global__ static void replay_level(const DeviceRecord *records, const int *indices,
                                    const double *values, const int *order, int begin, int end,
                                    double *x, double *y, double *z, double tol, int *status) {
    int pos = begin + blockIdx.x * blockDim.x + threadIdx.x;
    if (pos < end && !status[1])
        replay_record(records[order[pos]], indices, values, x, y, z, tol, status);
}
__global__ static void replay_chain(const DeviceRecord *records, const int *indices,
                                    const double *values, const int *order, const int *starts,
                                    int levels, double *x, double *y, double *z, double tol,
                                    int *status) {
    if (status[1])
        return;
    for (int level = 0; level < levels; ++level) {
        for (int pos = starts[level] + threadIdx.x; pos < starts[level + 1]; pos += blockDim.x)
            replay_record(records[order[pos]], indices, values, x, y, z, tol, status);
        if (blockDim.x > 1)
            __syncthreads();
    }
}
__global__ static void replay_device_level(const DeviceRecord *records, const int *indices,
                                           const double *values, const int *order,
                                           const int *starts, int level, double *x, double *y,
                                           double *z, double tol, int *status) {
    int pos = starts[level] + blockIdx.x * blockDim.x + threadIdx.x;
    if (pos < starts[level + 1] && !status[1])
        replay_record(records[order[pos]], indices, values, x, y, z, tol, status);
}
__global__ static void recovery_finite(int n, int m, const double *x, const double *y,
                                       const double *z, int *status) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (status[1])
        return;
    if ((i < n && (!isfinite(x[i]) || !isfinite(z[i]))) || (i < m && !isfinite(y[i])))
        atomicExch(status, (int)NUMERICAL_ERROR);
}
static Status postsolve_impl(const State *s, void *opaque, const double *xr, const double *yr,
                                const double *zr, double *packed, double *output_x,
                                double *output_y, double *output_z, bool device_inputs) {
    CudaRecovery *w = (CudaRecovery *)opaque;
    DeviceGuard guard(w->device);
    GPU_CHECK(guard.status);
    Status code = prepare_recovery(s, w);
    if (code)
        return code;
    size_t input_size = (size_t)2 * s->reduced_n + s->reduced_m;
    size_t output_size = (size_t)2 * s->original_n + s->original_m;
    if (device_inputs) {
        GPU_CHECK(cuda_memcpy(w->input.data, xr, (size_t)s->reduced_n * sizeof(double),
                                  cudaMemcpyDeviceToDevice));
        GPU_CHECK(cuda_memcpy(w->input.data + s->reduced_n, zr,
                                  (size_t)s->reduced_n * sizeof(double), cudaMemcpyDeviceToDevice));
        GPU_CHECK(cuda_memcpy(w->input.data + (size_t)2 * s->reduced_n, yr,
                                  (size_t)s->reduced_m * sizeof(double), cudaMemcpyDeviceToDevice));
    } else {
        if (s->reduced_n) {
            memcpy(packed, xr, (size_t)s->reduced_n * sizeof(double));
            memcpy(packed + s->reduced_n, zr, (size_t)s->reduced_n * sizeof(double));
        }
        if (s->reduced_m)
            memcpy(packed + (size_t)2 * s->reduced_n, yr, (size_t)s->reduced_m * sizeof(double));
        GPU_CHECK(w->input.upload(packed, input_size));
    }
    GPU_CHECK(cudaMemset(w->status.data, 0, 2 * sizeof(int)));
    GPU_CHECK(cudaMemset(w->data.data, 0, (output_size + 1) * sizeof(double)));
    double *x = w->data.data, *z = x + s->original_n, *y = z + s->original_n;
    double *dx = w->input.data, *dz = dx + s->reduced_n, *dy = dz + s->reduced_n;
    int width = std::max(s->reduced_n, s->reduced_m);
    if (width)
        scatter_reduced<<<((unsigned)width + 255) / 256, 256>>>(
            s->reduced_n, s->reduced_m, w->resident ? w->resident->col_ids.data : w->col_ids.data,
            w->resident ? w->resident->row_ids.data : w->row_ids.data, dx, dy, dz, x, y, z,
            w->status.data + 1);
    if (w->resident) {
        for (size_t step = w->resident->segments.size(); step > 0; --step) {
            const auto &tape = *w->resident->segments[step - 1];
            if (tape.folding) {
                unfold_classes<<<tape.count, 256>>>(tape.records.data, tape.indices.data, x, y, z,
                                                    w->status.data);
                continue;
            }
            const auto &plan = *w->resident_plans[step - 1];
            if (plan.levels > 64) {
                int threads = plan.max_width == 1 ? 1 : plan.max_width <= 32 ? 32 : 256;
                replay_chain<<<1, threads>>>(tape.records.data, tape.indices.data, tape.values.data,
                                             plan.order.data, plan.starts.data, plan.levels, x, y,
                                             z, s->tolerance, w->status.data);
            } else
                for (int level = 0; level < plan.levels; ++level)
                    replay_device_level<<<((unsigned)plan.max_width + 255) / 256, 256>>>(
                        tape.records.data, tape.indices.data, tape.values.data, plan.order.data,
                        plan.starts.data, level, x, y, z, s->tolerance, w->status.data);
        }
    } else {
        int levels = (int)w->levels.size() - 1;
        if (levels > 64) {
            int threads = w->max_width == 1 ? 1 : w->max_width <= 32 ? 32 : 256;
            replay_chain<<<1, threads>>>(w->records.data, w->index.data, w->values.data,
                                         w->order.data, w->starts.data, levels, x, y, z,
                                         s->tolerance, w->status.data);
        } else
            for (int level = 0; level < levels; ++level) {
                int begin = w->levels[level], end = w->levels[level + 1];
                replay_level<<<((unsigned)(end - begin) + 255) / 256, 256>>>(
                    w->records.data, w->index.data, w->values.data, w->order.data, begin, end, x, y,
                    z, s->tolerance, w->status.data);
            }
    }
    width = std::max(s->original_n, s->original_m);
    if (width)
        recovery_finite<<<((unsigned)width + 255) / 256, 256>>>(s->original_n, s->original_m, x, y,
                                                                z, w->status.data);
    GPU_CHECK(cudaGetLastError());
    int status[2];
    GPU_CHECK(w->status.download(status, 2));
    if (status[1])
        return INVALID_ARGUMENT;
    if (status[0])
        return (Status)status[0];
    if (device_inputs) {
        GPU_CHECK(cuda_memcpy(output_x, x, (size_t)s->original_n * sizeof(double),
                                  cudaMemcpyDeviceToDevice));
        GPU_CHECK(cuda_memcpy(output_z, z, (size_t)s->original_n * sizeof(double),
                                  cudaMemcpyDeviceToDevice));
        GPU_CHECK(cuda_memcpy(output_y, y, (size_t)s->original_m * sizeof(double),
                                  cudaMemcpyDeviceToDevice));
    } else
        GPU_CHECK(w->data.download(packed, output_size));
    return OK;
}
extern "C" Status cuda_postsolve(const State *s, void *opaque, const double *xr,
                                        const double *yr, const double *zr, double *packed) {
    return postsolve_impl(s, opaque, xr, yr, zr, packed, nullptr, nullptr, nullptr, false);
}
static bool recovery_device_pointer(const void *p, int count, int device) {
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
extern "C" Status cuda_postsolve_device(const State *s, void *opaque, const double *xr,
                                               const double *yr, const double *zr, double *x,
                                               double *y, double *z) {
    DeviceGuard guard(s->device);
    GPU_CHECK(guard.status);
    if (!recovery_device_pointer(xr, s->reduced_n, s->device) ||
        !recovery_device_pointer(zr, s->reduced_n, s->device) ||
        !recovery_device_pointer(yr, s->reduced_m, s->device) ||
        !recovery_device_pointer(x, s->original_n, s->device) ||
        !recovery_device_pointer(z, s->original_n, s->device) ||
        !recovery_device_pointer(y, s->original_m, s->device))
        return INVALID_ARGUMENT;
    return postsolve_impl(s, opaque, xr, yr, zr, nullptr, x, y, z, true);
}
